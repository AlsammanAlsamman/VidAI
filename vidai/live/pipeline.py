"""The live pipeline: capture -> sensors + processors (every frame) -> encoder, steered by rules, voice and Claude.

    ffmpeg A (capture: screen/camera/PiP + mic)
        ├─ video rgb24 ──► Python frame loop ──► processors ──► ffmpeg B (encode MKV) ──► video.mkv
        │                     ├─ preview (GUI), MotionSensor, learners, OCR sampler
        ├─ audio 48 kHz ──────────────────────────────────────► ffmpeg B
        └─ audio 16 kHz ──► AudioSensor ──► SpeechToText ──► transcript / voice_command
    bus ◄── every stat and action;  live.jsonl ◄── bus (Claude reads)
    control.jsonl ──► commands (Claude writes)  ──► processors / rules / learners / markers

Timeline: everything uses *video time* (frame index / fps), audio uses sample count; both come from the
same capture process, so anchors line up with the recorded file.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np
from pydantic import BaseModel, Field

from .. import ffmpeg
from ..capture import CaptureConfig, canvas_size, capture_inputs, ffmpeg_has_pulse
from .bus import LiveBus
from .commands import CommandsMixin
from .common import PREVIEW_EVERY, PREVIEW_H, PREVIEW_W  # noqa: F401  (re-exported)
from .dialog import DialogMixin, _match_option  # noqa: F401  (re-exported)
from .governor import GovernorMixin
from .history import UndoMixin
from .learn import LiveLearner
from .memory import MemoryMixin
from .processors import Context, LiveProcessor, ProcessorChain
from .rules import RuleEngine
from .sensors import SR, WAKE, AudioSensor, MotionSensor, ScreenText, SpeechToText


class LiveConfig(BaseModel):
    """What runs live. Claude sets this from the brief; everything can be changed during recording."""
    stt: bool = True  # speech to text + voice commands
    stt_model: str = ""  # "" = auto: tiny.en for English (fastest), small for Arabic, base otherwise
    utterance_gap: float = 0.35  # quiet time that ends a spoken command (anchors use min_silence separately)
    stt_language: str | None = "en"  # English only for now; "ar", or None = auto among stt_languages
    stt_languages: list[str] = Field(default_factory=lambda: ["en"])  # auto-detect only among these
    wake_words: list[str] = Field(default_factory=list)  # extra wake words ("vidai" variants are built in)
    ocr: bool = False  # text on screen
    ocr_interval: float = 2.0
    ocr_langs: str = "eng+ara"  # tesseract languages (if installed)
    processors: list[dict] = Field(default_factory=list)  # [{"name", "type"|"file", "params", "enabled"}]
    rules: list[dict] = Field(default_factory=list)
    learners: list[dict] = Field(default_factory=list)  # [{"name", "labels", "region"}]
    voice_actions: bool = True  # built-in voice commands (mark, zoom, captions, ...) act immediately
    thinking: str = "video"  # animated VidAI icon while Claude works on a request: video | preview | off
    speak: bool = True  # VidAI talks (spd-say) when it needs permission
    address: str = "Master"  # how VidAI addresses the user
    thinking_position: str = "top-right"
    thinking_timeout: float = 120.0  # hide the icon if Claude has not answered after this many seconds
    governor: bool = True  # keep 30 fps: lighten tracking/preview under load, warn before switching effects off
    hw_encode: bool = True  # use the GPU encoder when available


class LivePipeline(CommandsMixin, DialogMixin, UndoMixin, MemoryMixin, GovernorMixin):
    """One live run (preview or recording). This file owns capture, the frame/audio/control loops and stop;
    the behaviour lives in the mixins: commands.py (every command), dialog.py (voice, requests, questions,
    suggest, VidAI's voice), history.py (undo/redo), memory.py (learning), governor.py (performance)."""

    def __init__(self, cfg: CaptureConfig, live: LiveConfig | None = None, output: str | Path | None = None,
                 session_dir: str | Path | None = None, on_frame: Callable[[np.ndarray], None] | None = None,
                 min_silence: float = 0.5, silence_db: float | None = None, scene_threshold: float = 0.12,
                 on_stop_request: Callable[[], None] | None = None,
                 on_start_request: Callable[[], None] | None = None) -> None:
        self.cfg, self.live = cfg, live or LiveConfig()
        self.output = str(output) if output else None
        self.dir = Path(session_dir) if session_dir else None
        self.on_frame, self.on_stop_request, self.on_start_request = on_frame, on_stop_request, on_start_request
        self.W, self.H = canvas_size(cfg)
        self.fps = cfg.fps
        self.frames = 0
        self.clock = lambda: self.frames / self.fps
        log = (self.dir / "live.jsonl") if self.dir else None  # preview events are logged too
        self.bus = LiveBus(log, clock=self.clock)
        self.ctx = Context(self.bus, self.W, self.H)
        self.chain = ProcessorChain(self.ctx)
        self.rules = RuleEngine(self.bus, self._rule_action)
        self.learners: dict[str, LiveLearner] = {}
        self.audio = AudioSensor(self.bus, min_silence=min_silence, silence_db=silence_db,
                                 on_utterance=self._on_utterance, utterance_gap=self.live.utterance_gap)
        self.motion = MotionSensor(self.bus, scene_threshold=scene_threshold)
        self.stt: SpeechToText | None = None
        self.ocr: ScreenText | None = None
        self.level_db = -90.0
        self.error: str | None = None
        self.loop_ms = 0.0
        self.preview_frame: np.ndarray | None = None
        self._latest_full: np.ndarray | None = None
        self._procs: list[subprocess.Popen] = []
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        # command() is called from the control loop, bus subscribers (STT, audio, frame thread), the GUI and
        # worker threads: one re-entrant lock keeps the chain / history / questions consistent
        self._cmd_lock = threading.RLock()
        self._err_count: dict[str, int] = {}
        self._tmp = tempfile.mkdtemp(prefix="vidai_live_")
        self._n_tmp = 0
        self._control_pos = 0
        self.stopped = False
        self.pending: dict[int, dict] = {}  # Claude requests waiting for an answer (seq -> event)
        # performance governor
        self.level = 0  # 0 normal, 1 light, 2 minimal
        self.lag_frames = 0.0
        self.frame_ms_ema = 0.0
        self._t0_wall: float | None = None
        self._level_since = time.monotonic()
        self._pressure_since: float | None = None
        self.preview_every = PREVIEW_EVERY
        self.encoder_name = ""
        # a request to Claude is collected until the user stops talking ("... saying" <pause> "an award")
        self._req: dict | None = None
        self.asks: dict[str, str] = {}  # permission questions waiting for the user (id -> text)
        self.questions: dict[str, dict] = {}  # Claude's questions to the user (id -> {text, options, t})
        self.speaking_until = -1.0  # VidAI is talking until this AUDIO time: ignore the mic meanwhile
        from .voice import Voice

        self.voice = Voice(Path(session_dir) / "voice" if session_dir else None)
        self.voice_clips: list[tuple[float, str, float]] = []  # (start on the recording clock, wav, seconds)
        self._talk = False  # "VidAI talk": the next thing the user says is a question for Claude
        self.suggesting: dict | None = None  # "VidAI suggest": {"items", "i", "qid"}
        # undo / redo: each request is a group of reversible steps
        self._txn = 0
        self.history: list[dict] = []  # {"txn", "undo": [cmds], "redo": [cmds]}
        self.redo_stack: list[dict] = []
        # learning (vidai.profile): what each request did, so mistakes and good solutions are remembered
        from ..profile import Profile

        self.profile = Profile()
        self.requests_log: list[dict] = []  # {msg, t, via, names, cmds, removed}
        self._claude_req: dict | None = None  # the request Claude is answering now
        self._probation: list[dict] = []  # Claude's solutions kept for a while -> become macros
        self.learned: list[dict] = []
        self.request_gap = 1.0  # seconds of quiet after the last words before the request is sent
        self._thinking: LiveProcessor | None = None

    # ------------------------------------------------------------------ start
    @property
    def running(self) -> bool:
        return bool(self._procs) and self._procs[0].poll() is None

    def start(self) -> None:
        try:
            self._start()
        except BaseException:
            self._abort_start()
            raise

    def _abort_start(self) -> None:
        """start() failed half-way: kill what was started and unblock threads waiting on the audio FIFO."""
        self._stop.set()
        for p in [getattr(self, n, None) for n in ("cap", "enc", "mic")]:
            if p is not None and p.poll() is None:
                p.kill()
                p.wait()
        atap = os.path.join(self._tmp, "atap.fifo")
        if os.path.exists(atap):
            try:  # a reader blocked in open() needs a writer to come and go
                fd = os.open(atap, os.O_WRONLY | os.O_NONBLOCK)
                os.close(fd)
            except OSError:
                pass
        self.chain.close_all()
        self.bus.close()
        shutil.rmtree(self._tmp, ignore_errors=True)
        self.stopped = True

    def _start(self) -> None:
        use_parec = self.cfg.mic and self.cfg.mode != "test" and not ffmpeg_has_pulse()
        if use_parec and not shutil.which("parec"):
            raise RuntimeError("cannot record the microphone: install pulseaudio-utils (parec) or turn the mic off")
        has_audio = self.cfg.mic
        atap = os.path.join(self._tmp, "atap.fifo")
        os.mkfifo(atap)
        # while recording, audio and video go to separate files (no pipe coupling between the two ffmpeg
        # processes); they are joined losslessly at stop, or by recovery after a crash
        out = Path(self.output) if self.output else None
        self.video_part = out.with_name(out.stem + ".part-video.mkv") if out else None
        self.audio_part = out.with_name(out.stem + ".part-audio.mka") if (out and has_audio) else None

        # sensors that need workers
        recording = bool(self.output)  # preview runs voice (so "VidAI record" works) but not OCR
        if self.live.stt and has_audio:
            lang = self.live.stt_language
            model = self.live.stt_model or ("tiny.en" if lang == "en" else "small" if lang and "ar" in lang
                                            else "base")
            self.stt = SpeechToText(self.bus, model, lang if lang and "+" not in lang else None,
                                    wake_words=(self.live.wake_words + WAKE) if self.live.wake_words else None,
                                    allowed_languages=self.live.stt_languages)
            self.stt.profile = self.profile
        if self.live.ocr and recording:
            self.ocr = ScreenText(self.bus, self.live.ocr_interval, self.live.ocr_langs)
        self.bus.subscribe(self._on_voice, {"voice_command"})
        self.bus.subscribe(self._on_speech_for_request, {"speech_start", "speech_end", "transcript"})
        self.bus.subscribe(self._on_claude_request, {"claude"})
        for p in self.live.processors:
            self.command({"cmd": "add", **p}, source="config")
        for r in self.live.rules:
            self.command({"cmd": "rule", "rule": r}, source="config")
        for l in self.live.learners:
            self.command({"cmd": "learn", **l}, source="config")

        # ffmpeg A: capture
        in_args, graph, inputs = capture_inputs(self.cfg, use_parec)
        a_cmd = [ffmpeg.ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y"] + in_args
        if "mic" in inputs:
            split = "asplit[arec][atapin]" if (self.output) else "anull[atapin]"
            graph.append(f"[mic]{split}")
            graph.append(f"[atapin]aresample={SR},aformat=sample_fmts=flt:channel_layouts=mono[atap]")
        a_cmd += ["-filter_complex", ";".join(graph), "-map", "[main]", "-f", "rawvideo", "pipe:1"]
        if "mic" in inputs:
            if self.output:
                a_cmd += ["-map", "[arec]", "-c:a", "aac", "-b:a", "192k", "-flush_packets", "1",
                          "-cluster_time_limit", "1000", str(self.audio_part)]
            a_cmd += ["-map", "[atap]", "-f", "f32le", atap]

        # readers first (FIFO opens block until both ends exist)
        if "mic" in inputs:
            self._spawn(self._audio_loop, atap)
        # ffmpeg B: video encoder
        if self.output:
            hw = ffmpeg.hw_encoder() if self.live.hw_encode else None
            enc = hw or ["-c:v", "libx264", "-preset", self.cfg.preset, "-crf", str(self.cfg.crf), "-pix_fmt", "yuv420p"]
            self.encoder_name = "h264_vaapi (GPU)" if hw else f"libx264 {self.cfg.preset}"
            b_cmd = [ffmpeg.ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                     "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{self.W}x{self.H}", "-r", str(self.fps),
                     "-i", "pipe:0", *enc, "-g", str(self.fps * 2),
                     "-flush_packets", "1", "-cluster_time_limit", "1000", str(self.video_part)]
            self.enc = subprocess.Popen(b_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            self.enc = None
        if use_parec:
            pa = ["parec", "--raw", "--format=s16le", "--rate=48000", "--channels=2", "--latency-msec=20"]
            if self.cfg.mic_source:
                pa.append(f"--device={self.cfg.mic_source}")
            self.mic = subprocess.Popen(pa, stdout=subprocess.PIPE)
            a_stdin = self.mic.stdout
        else:
            self.mic = None
            a_stdin = subprocess.PIPE
        self.cap = subprocess.Popen(a_cmd, stdin=a_stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self._procs = [self.cap] + ([self.enc] if self.enc else [])
        for p, name in ((self.cap, "capture"), (self.enc, "encoder")):
            if p:
                self._spawn(self._watch_stderr, p, name)
        self._spawn(self._frame_loop)
        if self.dir:
            self._spawn(self._control_loop)
        self.bus.publish("action", {"what": "started", "recording": bool(self.output), "size": [self.W, self.H],
                                    "stt": bool(self.stt), "ocr": bool(self.ocr)})

    def _spawn(self, fn, *args) -> None:
        th = threading.Thread(target=fn, args=args, daemon=True)
        th.start()
        self._threads.append(th)

    def _watch_stderr(self, proc: subprocess.Popen, name: str) -> None:
        err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        if err.strip() and not self._stop.is_set():
            self.error = f"{name}: {err[-1500:]}"
            self.bus.publish("error", {"where": name, "error": err[-500:]})

    # ------------------------------------------------------------------ loops
    def _guard(self, where: str, fn, *args) -> None:
        """Run optional per-frame work; an error is reported (rate-limited) and never stops the recording."""
        try:
            fn(*args)
        except Exception as e:
            n = self._err_count[where] = self._err_count.get(where, 0) + 1
            if n <= 3 or n % 300 == 0:
                self.bus.publish("error", {"where": where, "error": repr(e)[:200], "count": n})

    def _frame_loop(self) -> None:
        try:
            self._frame_loop_inner()
        except Exception as e:  # never leave capture blocked on a full pipe: report, close, drain
            self.error = f"frame loop: {e!r}"[:500]
            self.bus.publish("error", {"where": "frame_loop", "error": repr(e)[:300]})
            if self.enc and self.enc.stdin:
                try:
                    self.enc.stdin.close()
                except OSError:
                    pass
            out = self.cap.stdout
            while out and out.read(1 << 20):
                pass

    def _frame_loop_inner(self) -> None:
        size = self.W * self.H * 3
        out = self.cap.stdout
        enc_in = self.enc.stdin if self.enc else None
        import cv2

        t_perf, n_perf, ms_acc = time.monotonic(), 0, 0.0
        while True:
            buf = out.read(size)
            if len(buf) < size:
                break
            t0 = time.perf_counter()
            t = self.frames / self.fps
            if self._t0_wall is None:
                self._t0_wall = time.monotonic()
            self.lag_frames = (time.monotonic() - self._t0_wall) * self.fps - self.frames  # >0 = behind real time
            frame = np.frombuffer(buf, np.uint8).reshape(self.H, self.W, 3)
            if any(p.enabled for p in self.chain.items):
                frame = frame.copy()
                frame = self.chain.run(frame, t)
                if frame.shape[:2] != (self.H, self.W):
                    frame = cv2.resize(frame, (self.W, self.H))
                frame = np.ascontiguousarray(frame, np.uint8)
            if enc_in:
                try:
                    enc_in.write(frame.data if frame.flags.c_contiguous else frame.tobytes())
                except (BrokenPipeError, OSError):
                    enc_in = None
                    self.bus.publish("error", {"where": "encoder", "error": "encoder closed its input"})
            if self.frames % self.preview_every == 0:
                small = cv2.resize(frame, (PREVIEW_W, PREVIEW_H), interpolation=cv2.INTER_AREA)
                if self.pending and self.live.thinking == "preview" and self._thinking:
                    small = self._thinking.process(np.ascontiguousarray(small), t, self.ctx)
                self.preview_frame = small
                if self.on_frame:
                    self._guard("preview", self.on_frame, small)
                if self.frames % (self.preview_every * 3) == 0:  # 2 Hz
                    self._guard("motion", self.motion.feed, small, t)
                    if self.ocr and self.motion.values:
                        self.ocr.notify_motion(self.motion.values[-1])
                for l in list(self.learners.values()):
                    self._guard(f"learner:{l.name}", l.feed, small, t)
            if self.ctx.tracks is not None and self.frames % (1 + self.level) == 0:  # lighter under load
                self._guard("tracking", self.ctx.tracks.feed, frame)
            if self.ocr and self.ocr.want(t):
                self._guard("ocr", self.ocr.submit, frame, t)
            self.frames += 1
            ms = (time.perf_counter() - t0) * 1000
            self.frame_ms_ema = 0.9 * self.frame_ms_ema + 0.1 * ms
            ms_acc += ms
            n_perf += 1
            if time.monotonic() - t_perf >= 5.0:
                self.loop_ms = ms_acc / n_perf
                self.bus.publish("perf", {"fps": round(n_perf / (time.monotonic() - t_perf), 1),
                                          "frame_ms": round(self.loop_ms, 2), "lag_frames": round(self.lag_frames),
                                          "level": self.level, "encoder": self.encoder_name,
                                          "active_processors": [p.name for p in self.chain.items if p.enabled]})
                t_perf, n_perf, ms_acc = time.monotonic(), 0, 0.0
        if enc_in:
            try:
                enc_in.close()
            except OSError:
                pass

    def _audio_loop(self, path: str) -> None:
        nbytes = SR // 10 * 4
        with open(path, "rb") as f:
            while True:
                buf = f.read(nbytes)
                if len(buf) < nbytes:
                    break
                block = np.frombuffer(buf, np.float32)
                self._guard("audio", self.audio.feed, block)  # keep draining the FIFO whatever happens
                if self.audio.levels:
                    self.level_db = self.audio.levels[-1]

    def _control_loop(self) -> None:
        path = self.dir / "control.jsonl"
        pos_file = self.dir / "control.pos"
        size = path.stat().st_size if path.exists() else 0
        try:  # continue where the previous pipeline (preview) stopped: nothing sent in between is lost
            self._control_pos = min(int(pos_file.read_text()), size)
        except (OSError, ValueError):
            self._control_pos = size  # only new commands
        while not self._stop.is_set():
            try:
                if path.exists() and path.stat().st_size > self._control_pos:
                    with open(path, "rb") as f:
                        f.seek(self._control_pos)
                        data = f.read()
                    end = data.rfind(b"\n") + 1  # a half-written last line waits for the next round
                    self._control_pos += end
                    for line in data[:end].decode("utf-8", errors="replace").splitlines():
                        if line.strip():
                            try:
                                self.command(json.loads(line), source="claude")
                            except json.JSONDecodeError as e:
                                self.bus.publish("error", {"where": "control", "error": f"bad JSON: {e}"})
            except OSError:
                pass
            self._flush_request()
            self._check_probation()
            for qid, q in list(self.questions.items()):  # nobody answered for 90 s: close the question
                if time.monotonic() - q.get("t", time.monotonic()) > 90:
                    self.command({"cmd": "cancel_question", "question": qid}, source="timeout")
            if self.live.governor:
                self._govern()
            if self.pending:
                oldest = min(p["t"] for p in self.pending.values())
                if self.clock() - oldest > self.live.thinking_timeout:
                    self.bus.publish("error", {"where": "claude", "error": "no answer from Claude; hiding the icon",
                                               "requests": [p["message"] for p in self.pending.values()]})
                    self.command({"cmd": "done"}, source="timeout")
            self._stop.wait(0.1)
        try:
            pos_file.write_text(str(self._control_pos))
        except OSError:
            pass

    # ------------------------------------------------------------------ stop
    def stop(self, timeout: float = 20.0) -> None:
        if self.stopped:
            return
        self.stopped = True
        self._flush_request(force=True)
        self._stop.set()
        cap = getattr(self, "cap", None)
        if cap and cap.poll() is None:
            try:
                if cap.stdin:
                    cap.stdin.write(b"q")
                    cap.stdin.flush()
                else:
                    cap.send_signal(signal.SIGINT)
            except (BrokenPipeError, OSError):
                cap.send_signal(signal.SIGINT)
            if self.mic:
                self.mic.terminate()
            try:  # with the mic piped in (parec), ffmpeg catches SIGINT but keeps going; SIGTERM ends it cleanly
                cap.wait(2)
            except subprocess.TimeoutExpired:
                cap.send_signal(signal.SIGTERM)
        for p in self._procs:
            try:
                p.wait(timeout)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        if self.mic and self.mic.poll() is None:
            self.mic.kill()
        for th in self._threads:
            th.join(3)
        if self.output:
            try:
                mux_parts(self.video_part, self.audio_part, self.output, self.voice_clips)
            except Exception as e:
                self.error = f"mux: {e}"
                self.bus.publish("error", {"where": "mux", "error": str(e)[:300]})
        self.audio.finish()
        if self.stt:
            if self.output:  # the recording's last words become anchors; the preview's don't matter
                self.stt.flush()
            self.stt.close(wait=bool(self.output))
        if self.ocr:
            self.ocr.close()
        for l in self.learners.values():
            l.finish(self.clock())
        self.chain.close_all()
        if self.ctx.tracks is not None:
            self.ctx.tracks.close()
        if self.output:
            self._remember_session()
        self.bus.publish("action", {"what": "stopped", "duration": round(self.clock(), 2)})
        self.bus.close()
        shutil.rmtree(self._tmp, ignore_errors=True)


def mux_parts(video_part: str | Path | None, audio_part: str | Path | None, output: str | Path,
              voice_clips: list[tuple[float, str, float]] | None = None) -> Path:
    """Join the separately recorded video and audio tracks into the final file, then delete the parts.
    Video is always copied (lossless). With VidAI voice clips, they are mixed into the audio at their times and
    the microphone is turned down under them (no echo). Tolerates truncated parts (crash recovery)."""
    output = Path(output)
    vp = Path(video_part) if video_part else None
    ap = Path(audio_part) if audio_part else None
    if not vp or not vp.exists() or vp.stat().st_size < 1024:
        raise FileNotFoundError(f"no recorded video in {vp}")
    args = ["-err_detect", "ignore_err", "-i", str(vp)]
    has_a = bool(ap and ap.exists() and ap.stat().st_size > 256)
    if has_a:
        args += ["-err_detect", "ignore_err", "-i", str(ap)]
    clips = [c for c in (voice_clips or []) if Path(c[1]).exists()]
    tmp = output.with_name(output.stem + ".muxing.mkv")
    if clips:
        first = 2 if has_a else 1
        for _, path, _ in clips:
            args += ["-i", path]
        g = []
        mix = []
        if has_a:
            duck = "+".join(f"between(t,{t0:.3f},{t0 + d + 0.3:.3f})" for t0, _, d in clips)
            g.append(f"[1:a]volume=0.2:enable='{duck}'[mic]")
            mix.append("[mic]")
        for i, (t0, _, _) in enumerate(clips):
            g.append(f"[{first + i}:a]aresample=48000,aformat=channel_layouts=stereo,"
                     f"adelay={int(t0 * 1000)}:all=1[v{i}]")
            mix.append(f"[v{i}]")
        g.append(f"{''.join(mix)}amix=inputs={len(mix)}:normalize=0:duration={'first' if has_a else 'longest'}[aout]")
        args += ["-filter_complex", ";".join(g), "-map", "0:v", "-map", "[aout]", "-c:v", "copy",
                 "-c:a", "aac", "-b:a", "192k", str(tmp)]
    else:
        args += ["-map", "0:v"] + (["-map", "1:a"] if has_a else []) + ["-c", "copy", str(tmp)]
    ffmpeg.run(args)
    os.replace(tmp, output)
    vp.unlink(missing_ok=True)
    if ap:
        ap.unlink(missing_ok=True)
    return output
