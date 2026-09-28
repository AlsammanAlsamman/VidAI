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

import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from pydantic import BaseModel, Field

from .. import ffmpeg
from ..capture import CaptureConfig, capture_inputs, canvas_size, ffmpeg_has_pulse
from .bus import LiveBus
from .learn import LiveLearner
from .processors import REGISTRY, Context, LiveProcessor, ProcessorChain
from .rules import RuleEngine
from .sensors import SR, WAKE, AudioSensor, MotionSensor, ScreenText, SpeechToText

PREVIEW_W, PREVIEW_H = 320, 180
PREVIEW_EVERY = 5  # frames (6 Hz at 30 fps)


class LiveConfig(BaseModel):
    """What runs live. Claude sets this from the brief; everything can be changed during recording."""
    stt: bool = True  # speech to text + voice commands
    stt_model: str = ""  # tiny | base | small; "" = small for Arabic, base otherwise
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


class LivePipeline:
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
                                 on_utterance=self._on_utterance)
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
        self._tmp = tempfile.mkdtemp(prefix="vidai_live_")
        self._n_tmp = 0
        self._control_pos = 0
        self.stopped = False
        self.pending: dict[int, dict] = {}  # Claude requests waiting for an answer (seq -> event)
        # a request to Claude is collected until the user stops talking ("... saying" <pause> "an award")
        self._req: dict | None = None
        self.asks: dict[str, str] = {}  # permission questions waiting for the user (id -> text)
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
            model = self.live.stt_model or ("small" if lang and "ar" in lang else "base")
            self.stt = SpeechToText(self.bus, model, lang if lang and "+" not in lang else None,
                                    wake_words=(self.live.wake_words + WAKE) if self.live.wake_words else None,
                                    allowed_languages=self.live.stt_languages)
            self.stt.profile = self.profile
        if self.live.ocr and recording:
            self.ocr = ScreenText(self.bus, self.live.ocr_interval, self.live.ocr_langs)
        self.bus.subscribe(self._on_voice, {"voice_command"})
        self.bus.subscribe(self._on_speech_for_request, {"speech_start", "transcript"})
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
            b_cmd = [ffmpeg.ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                     "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{self.W}x{self.H}", "-r", str(self.fps),
                     "-i", "pipe:0", "-c:v", "libx264", "-preset", self.cfg.preset, "-crf", str(self.cfg.crf),
                     "-pix_fmt", "yuv420p", "-g", str(self.fps * 2),
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
    def _frame_loop(self) -> None:
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
            if self.frames % PREVIEW_EVERY == 0:
                small = cv2.resize(frame, (PREVIEW_W, PREVIEW_H), interpolation=cv2.INTER_AREA)
                if self.pending and self.live.thinking == "preview" and self._thinking:
                    small = self._thinking.process(np.ascontiguousarray(small), t, self.ctx)
                self.preview_frame = small
                if self.on_frame:
                    self.on_frame(small)
                if self.frames % (PREVIEW_EVERY * 3) == 0:  # 2 Hz
                    self.motion.feed(small, t)
                    if self.ocr:
                        self.ocr.notify_motion(self.motion.values[-1])
                for l in list(self.learners.values()):
                    l.feed(small, t)
            if self.ctx.tracks is not None:
                self.ctx.tracks.feed(frame)
            if self.ocr and self.ocr.want(t):
                self.ocr.submit(frame, t)
            self.frames += 1
            ms_acc += (time.perf_counter() - t0) * 1000
            n_perf += 1
            if time.monotonic() - t_perf >= 5.0:
                self.loop_ms = ms_acc / n_perf
                self.bus.publish("perf", {"fps": round(n_perf / (time.monotonic() - t_perf), 1),
                                          "frame_ms": round(self.loop_ms, 2),
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
                self.audio.feed(block)
                self.level_db = self.audio.levels[-1]

    def _control_loop(self) -> None:
        path = self.dir / "control.jsonl"
        if path.exists():
            self._control_pos = path.stat().st_size  # only new commands
        while not self._stop.is_set():
            try:
                if path.exists() and path.stat().st_size > self._control_pos:
                    with open(path, encoding="utf-8") as f:
                        f.seek(self._control_pos)
                        data = f.read()
                        self._control_pos = f.tell()
                    for line in data.splitlines():
                        if line.strip():
                            try:
                                self.command(json.loads(line), source="claude")
                            except json.JSONDecodeError as e:
                                self.bus.publish("error", {"where": "control", "error": f"bad JSON: {e}"})
            except OSError:
                pass
            self._flush_request()
            self._check_probation()
            if self.pending:
                oldest = min(p["t"] for p in self.pending.values())
                if self.clock() - oldest > self.live.thinking_timeout:
                    self.bus.publish("error", {"where": "claude", "error": "no answer from Claude; hiding the icon",
                                               "requests": [p["message"] for p in self.pending.values()]})
                    self.command({"cmd": "done"}, source="timeout")
            self._stop.wait(0.1)

    # ------------------------------------------------------------------ events -> actions
    def _on_claude_request(self, ev: dict) -> None:
        if ev["data"].get("source") not in ("voice", "user", "gui"):
            return  # rules' notify_claude does not show the icon
        self.pending[ev["seq"]] = {"t": ev["t"], "message": ev["data"].get("message", "")}
        self._show_thinking(True)

    def _show_thinking(self, on: bool) -> None:
        mode = self.live.thinking
        if mode == "off":
            return
        if self._thinking is None:
            self._thinking = REGISTRY["thinking"]("_thinking", {"position": self.live.thinking_position}, False)
            for w in (self.W, PREVIEW_W):  # build the animation here (bus thread), not in the frame loop
                self._thinking._sprites[w] = self._thinking._build(w)
            if mode == "video":
                self.chain.add(self._thinking)
        if on and not self._thinking.enabled:
            self._thinking.shown_at = None
        self._thinking.enabled = on if mode == "video" else False
        self.bus.publish("action", {"what": "thinking_on" if on else "thinking_off", "pending": len(self.pending)})

    def _on_utterance(self, samples: np.ndarray, start: float, end: float) -> None:
        if self.stt:
            self.stt.submit(samples, start, end)

    def _on_voice(self, ev: dict) -> None:
        d = ev["data"]
        cmd, args = d["command"], d.get("args", "")
        if cmd == "claude":
            if args.strip():  # an empty request is never sent to Claude
                self._req = {"parts": [args.strip()], "due": time.monotonic() + self.request_gap,
                             "waiting": False, "cap": time.monotonic() + 20.0, "seq": ev["seq"]}
                self.bus.publish("action", {"what": "listening_request", "so_far": args.strip()})
                self._flush_if_obvious()
            return
        if not self.live.voice_actions:
            return
        if cmd in ("marker", "mistake", "section", "important"):
            self.command({"cmd": "mark", "type": cmd, "note": args}, source="voice")
            if cmd == "important":
                self.command({"cmd": "shape", "shape": "box", "x": 0.5, "y": 0.5, "w": 0.96, "h": 0.94,
                              "color": "#FFCC00", "thickness": 0.006, "for": 1.5}, source="voice")
        elif cmd == "zoom_in":
            self.command({"cmd": "zoom", "x": 0.25, "y": 0.25, "w": 0.5, "h": 0.5, "for": 8}, source="voice")
        elif cmd == "zoom_out":
            self.command({"cmd": "remove", "name": "zoom"}, source="voice")
        elif cmd == "captions_on":
            self.command({"cmd": "add", "name": "captions", "type": "captions"}, source="voice")
        elif cmd == "captions_off":
            self.command({"cmd": "remove", "name": "captions"}, source="voice")
        elif cmd == "stop":
            self.command({"cmd": "stop"}, source="voice")
        elif cmd in ("confirm", "deny"):
            if self.asks:
                self.command({"cmd": cmd}, source="voice")
        elif cmd == "full_access":
            self.command({"cmd": "mode", "mode": "full"}, source="voice")
        elif cmd == "ask_first":
            self.command({"cmd": "mode", "mode": "ask"}, source="voice")
        elif cmd == "record":
            if self.on_start_request and not self.output:
                self.on_start_request()
                self.bus.publish("action", {"what": "record_requested", "source": "voice"})
        elif cmd == "learn":
            name = args.split()[0] if args else f"learner{len(self.learners) + 1}"
            self.command({"cmd": "learn", "name": name}, source="voice")
        elif cmd == "label":
            parts = args.split()
            if len(parts) >= 2 and parts[0] in self.learners:
                self.command({"cmd": "label", "name": parts[0], "value": " ".join(parts[1:])}, source="voice")
            elif parts and self.learners:
                self.command({"cmd": "label", "name": list(self.learners)[-1], "value": " ".join(parts)},
                             source="voice")
        elif cmd == "wrong":
            last = self.requests_log[-1] if self.requests_log else None
            if last and last["names"] and self.clock() - last["t"] < 20:  # undo what VidAI just did
                for n in last["names"]:
                    self.command({"cmd": "remove", "name": n}, source="voice")
                self.say("Sorry. What did you mean?")
            elif self.learners:
                self.command({"cmd": "wrong", "name": list(self.learners)[-1]}, source="voice")

    def _on_speech_for_request(self, ev: dict) -> None:
        r = self._req
        if r is None:
            return
        if ev["kind"] == "speech_start":  # the user keeps talking: wait for that transcript
            r["waiting"] = True
        elif ev["kind"] == "transcript" and not ev["data"].get("is_command") and ev["seq"] > r["seq"]:
            r["parts"].append(ev["data"]["text"].strip())
            r["waiting"] = False
            r["due"] = time.monotonic() + self.request_gap
            self._flush_if_obvious()

    def _flush_if_obvious(self) -> None:
        """Act at once when the fast path understands a finished sentence (no need to wait for the pause)."""
        import re as _re

        from .intents import match

        r = self._req
        if r is None:
            return
        msg = " ".join(" ".join(p.rstrip(".…") for p in r["parts"]).split())
        unfinished = _re.search(r"\b(saying|says|say|with|and|the|a|an|to|on|in|my|of|that|above|over)\s*$",
                                msg.lower())
        if not unfinished and match(msg, self.chain):
            self._flush_request(force=True)

    def _flush_request(self, force: bool = False) -> None:
        r = self._req
        if r is None:
            return
        now = time.monotonic()
        if force or now >= r["cap"] or (not r["waiting"] and now >= r["due"]) or \
                (r["waiting"] and now >= r["due"] + 3.0):  # speech started but no transcript came
            self._req = None
            msg = " ".join(" ".join(p.rstrip(".…") for p in r["parts"]).replace("...", " ").split())
            self.request(msg, source="voice")

    def request(self, msg: str, source: str = "voice") -> dict:
        """A request in plain words: handled locally when VidAI understands it (fast path, < 1 s),
        otherwise sent to Claude (thinking icon until Claude answers)."""
        from .intents import match

        fixed = self.profile.correct(msg)
        if fixed != msg:
            self.bus.publish("action", {"what": "corrected", "heard": msg, "meant": fixed})
            msg = fixed
        via, cmds = "memory", None
        macro = self.profile.find_macro(msg)
        if macro:
            cmds = [dict(c) for c in macro["commands"]]
            self.profile.used_macro(macro["phrase"])
        else:
            via = "fast"
            try:
                cmds = match(msg, self.chain)
            except Exception as e:
                self.bus.publish("error", {"where": "fast_path", "error": repr(e)[:200]})
            cmds = [self._apply_prefs(c) for c in cmds] if cmds else None
        rec = {"msg": msg, "t": self.clock(), "via": via if cmds else "claude", "names": [], "cmds": [],
               "removed": False}
        self._finish_request_log(rec)
        if cmds:
            for c in cmds:
                self.command(c, source=via)
                if c.get("cmd") in ("add", "text", "zoom", "shape", "image", "blur"):
                    rec["names"].append(c.get("name") or c.get("cmd"))
            self.bus.publish("action", {"what": "fast_request", "message": msg, "commands": cmds, "via": via})
            import re as _re

            if _re.search(r"\bswap\b|\bswitch hands\b|\bother way\b", msg.lower()):  # learn once per swap
                self.profile.set_pref("hands_swapped", not self.profile.pref("hands_swapped", False))
                self._learned("preference", hands_swapped=self.profile.pref("hands_swapped"))
            return {"handled": via, "commands": cmds}
        self._claude_req = rec
        self.bus.publish("claude", {"message": msg, "source": source})
        return {"handled": "claude"}

    # ------------------------------------------------------------------ learning
    def _apply_prefs(self, c: dict) -> dict:
        """Use what the user chose before (sizes, which hand) for a new effect."""
        c = dict(c)
        prm = dict(c.get("params") or {})
        if c.get("type") == "attach":
            sc = self.profile.pref(f"scale:{prm.get('what')}")
            if sc and "scale" not in (c.get("params") or {}):
                prm["scale"] = sc
            if self.profile.pref("hands_swapped") and prm.get("to") in ("right_hand", "left_hand"):
                prm["to"] = {"right_hand": "left_hand", "left_hand": "right_hand"}[prm["to"]]
            c["params"] = prm
        return c

    def _finish_request_log(self, new: dict) -> None:
        """A new request right after one whose result was removed = that one was misunderstood."""
        prev = self.requests_log[-1] if self.requests_log else None
        self.requests_log.append(new)
        if prev and prev["removed"] and new["t"] - prev["t"] < 25:
            import difflib

            r = difflib.SequenceMatcher(None, prev["msg"].lower(), new["msg"].lower()).ratio()
            if 0.5 <= r < 1.0 and self.profile.learn_correction(prev["msg"], new["msg"]):
                self._learned("vocabulary", heard=prev["msg"], meant=new["msg"])

    def _learned(self, kind: str, **info) -> None:
        self.learned.append({"kind": kind, **info})
        self.bus.publish("action", {"what": "learned", "kind": kind, **info})

    def _track_learning(self, cmd: str | None, c: dict, source: str) -> None:
        name = c.get("name")
        # an effect removed soon after its request -> that request went wrong
        if cmd == "remove" and name:
            for rec in reversed(self.requests_log[-5:]):
                if name in rec["names"] and self.clock() - rec["t"] < 20 and not rec["removed"]:
                    rec["removed"] = True
                    self._probation = [p for p in self._probation if p["msg"] != rec["msg"]]
                    self.profile.add_lesson(f"The request '{rec['msg']}' was answered with {rec['names']} "
                                            f"({rec['via']}) and the user removed it right away.",
                                            ["mistake", rec["via"]], source="auto")
                    if rec["via"] == "memory":
                        self.profile.forget_macro(rec["msg"])
                    self._learned("mistake", request=rec["msg"])
        # size / hand adjustments become defaults
        if cmd == "set" and name:
            p = self.chain.get(name)
            prm = c.get("params") or {}
            if p is not None and "scale" in prm and p.params.get("what"):
                self.profile.nudge_pref(f"scale:{p.params['what']}", float(prm["scale"]))
        # what Claude does for a request -> candidate macro
        if source == "claude" and self._claude_req is not None and cmd in (
                "add", "text", "zoom", "shape", "image", "blur", "set", "remove", "enable", "disable", "rule"):
            self._claude_req["cmds"].append({"cmd": cmd, **{k: v for k, v in c.items() if k != "for"}})
            if cmd in ("add", "text", "zoom", "shape", "image", "blur"):
                self._claude_req["names"].append(name or cmd)
        if cmd == "done" and self._claude_req is not None:
            if self._claude_req["cmds"]:
                self._probation.append({**self._claude_req, "t_done": self.clock()})
            self._claude_req = None

    def _check_probation(self) -> None:
        """Claude's answer kept for 20 s (not removed) -> remember it: next time the request is instant."""
        now = self.clock()
        for p in list(self._probation):
            if now - p["t_done"] >= 20:
                self._probation.remove(p)
                alive = [self.chain.get(n) for n in p["names"]]
                if (not p["names"]) or any(q is not None and q.enabled for q in alive):
                    self.profile.add_macro(p["msg"], p["cmds"])
                    self._learned("macro", request=p["msg"])

    def _rule_action(self, action: dict, ev: dict, rule: dict) -> None:
        a = dict(action)
        if "show_text" in a:
            self.command({"cmd": "text", "text": a.pop("show_text"), **a}, source=f"rule:{rule['id']}")
        elif "zoom" in a:
            self.command({"cmd": "zoom", **a.pop("zoom"), **a}, source=f"rule:{rule['id']}")
        elif "shape" in a:
            self.command({"cmd": "shape", **a.pop("shape"), **a}, source=f"rule:{rule['id']}")
        elif "enable" in a:
            self.command({"cmd": "enable", "name": a["enable"], "for": a.get("for")}, source=f"rule:{rule['id']}")
        elif "disable" in a:
            self.command({"cmd": "disable", "name": a["disable"]}, source=f"rule:{rule['id']}")
        elif "set" in a:
            self.command({"cmd": "set", "name": a["set"], "params": a.get("params", {})}, source=f"rule:{rule['id']}")
        elif "mark" in a:
            self.command({"cmd": "mark", "type": a["mark"], "note": a.get("note", "")}, source=f"rule:{rule['id']}")
        elif "notify_claude" in a:
            self.bus.publish("claude", {"message": a["notify_claude"], "source": f"rule:{rule['id']}"})
        elif "label" in a:
            self.command({"cmd": "label", "name": a["label"], "value": a.get("value", "yes")},
                         source=f"rule:{rule['id']}")
        else:
            self.bus.publish("error", {"where": f"rule:{rule['id']}", "error": f"unknown action {action}"})

    # ------------------------------------------------------------------ commands
    def _temp_name(self, base: str) -> str:
        self._n_tmp += 1
        return f"{base}_{self._n_tmp}"

    def _load_file(self, path: str) -> list[str]:
        before = set(REGISTRY.items())
        mod_name = f"vidai_live_user_{Path(path).stem}_{int(time.time() * 1000)}"
        spec = importlib.util.spec_from_file_location(mod_name, path)
        module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        new = [k for k, v in REGISTRY.items() if (k, v) not in before]
        if not new:  # also accept plain subclasses without @register
            for obj in vars(module).values():
                if isinstance(obj, type) and issubclass(obj, LiveProcessor) and obj is not LiveProcessor \
                        and obj.__module__ == mod_name:
                    REGISTRY[obj.__name__.lower()] = obj
                    new.append(obj.__name__.lower())
        return new

    def command(self, c: dict, source: str = "claude") -> dict:
        """Apply one command (from Claude's control file, a rule, a voice command or the GUI)."""
        c = dict(c)
        cmd = c.pop("cmd", None)
        cid = c.pop("id", None)  # echoed back so the sender can match replies exactly
        tag = {"id": cid} if cid else {}
        t = self.clock()
        try:
            self._track_learning(cmd, c, source)
        except Exception as e:
            self.bus.publish("error", {"where": "learning", "error": repr(e)[:200]})
        try:
            res = self._do(cmd, c, t)
            self.bus.publish("ack", {"command": cmd, "source": source, **tag, **(res or {})})
            return res or {}
        except Exception as e:
            self.bus.publish("error", {"where": "command", "command": cmd, "source": source, **tag,
                                       "error": repr(e)[:300]})
            return {"error": repr(e)}

    def _do(self, cmd: str | None, c: dict, t: float) -> dict | None:
        dur = c.pop("for", None)
        until = t + float(dur) if dur else None
        if cmd == "add":
            if c.get("file"):
                types = self._load_file(c["file"])
                ptype = c.get("type") or types[-1]
            else:
                ptype = c.get("type", c.get("name"))
            cls = REGISTRY.get(str(ptype).lower())
            if cls is None:
                raise KeyError(f"unknown processor type {ptype!r}; known: {sorted(set(REGISTRY))}")
            p = self.chain.add(cls(c.get("name") or ptype, c.get("params", {}), c.get("enabled", True), until))
            p.spec = {"cmd": "add", "name": p.name, "params": p.params,  # to carry it from preview into recording
                      **({"file": c["file"]} if c.get("file") else {"type": ptype})}
            self.bus.publish("action", {"what": "processor_added", "name": p.name, "type": cls.__name__,
                                        "params": p.params, "until": until})
            return {"name": p.name}
        if cmd in ("text", "zoom", "shape", "image", "blur"):
            name = c.pop("name", None) or (cmd if cmd == "zoom" else self._temp_name(cmd))
            p = self.chain.add(REGISTRY[cmd](name, c, True, until))
            self.bus.publish("action", {"what": cmd, "name": name, "params": p.params, "until": until})
            return {"name": name}
        if cmd == "enable":
            p = self.chain.get(c["name"])
            if not p:
                raise KeyError(f"no processor {c['name']!r}")
            p.enabled, p.until, p.slow_frames = True, until, 0
            return {"name": p.name}
        if cmd == "disable":
            p = self.chain.get(c["name"])
            if p:
                p.enabled = False
            return {"name": c["name"]}
        if cmd == "remove":
            return {"removed": self.chain.remove(c["name"])}
        if cmd == "set":
            p = self.chain.get(c["name"])
            if not p:
                raise KeyError(f"no processor {c['name']!r}")
            p.configure(c.get("params", {}))
            return {"name": p.name, "params": p.params}
        if cmd == "rule":
            return {"rule": self.rules.add(c["rule"])}
        if cmd == "unrule":
            return {"removed": self.rules.remove(c["id"])}
        if cmd == "mark":
            self.bus.publish("marker", {"type": c.get("type", "marker"), "note": c.get("note", ""),
                                        "source": c.get("source", "")})
            return {"type": c.get("type", "marker")}
        if cmd == "learn":
            name = c.get("name") or f"learner{len(self.learners) + 1}"
            self.learners[name] = LiveLearner(self.bus, name, c.get("labels"), c.get("region"), c.get("k", 5))
            return {"learner": name, "labels": self.learners[name].model.labels}
        if cmd == "label":
            return self.learners[c["name"]].label(c.get("value", "yes"))
        if cmd == "wrong":
            return self.learners[c["name"]].wrong()
        if cmd == "forget":
            return {"removed": self.learners.pop(c["name"], None) is not None}
        if cmd == "stt":
            if self.stt:
                self.stt.enabled = bool(c.get("on", True))
                if c.get("language"):
                    self.stt.language = c["language"]
            return {"stt": bool(self.stt and self.stt.enabled)}
        if cmd == "ocr":
            if c.get("on", True) and not self.ocr:
                self.ocr = ScreenText(self.bus, c.get("interval", 2.0), c.get("langs", self.live.ocr_langs))
            elif self.ocr:
                self.ocr.enabled = bool(c.get("on", True))
            return {"ocr": bool(self.ocr and self.ocr.enabled)}
        if cmd == "stop":
            if self.on_stop_request:
                self.on_stop_request()
            return {"stopping": True}
        if cmd == "done":  # Claude finished a request (or all of them)
            req = c.get("request")
            if req is None:
                self.pending.clear()
            else:
                self.pending.pop(int(req), None)
            if not self.pending:
                self._show_thinking(False)
            return {"pending": len(self.pending)}
        if cmd == "ask":  # VidAI needs permission for an action (install, download, create...)
            from .. import actions

            rid, text = str(c["id_ask"]), c["text"]
            if actions.get_mode(self.dir) == "full":
                self.bus.publish("permission", {"request": rid, "state": "approved", "by": "full_access"})
                return {"request": rid, "state": "approved"}
            self.asks[rid] = text
            self.say(f"{self.live.address}, I need to {text}. Say VidAI confirm, or VidAI deny.")
            self.bus.publish("action", {"what": "asking", "request": rid, "text": text})
            if self.stt:  # a bare "yes" / "confirm" right after the question is enough
                self.stt.armed_until = self.clock() + 25
            return {"request": rid, "state": "pending"}
        if cmd in ("confirm", "deny"):
            rid = str(c.get("request") or (list(self.asks)[-1] if self.asks else ""))
            if rid not in self.asks:
                return {"state": "nothing to answer"}
            text = self.asks.pop(rid)
            state = "approved" if cmd == "confirm" else "denied"
            self.bus.publish("permission", {"request": rid, "state": state, "by": c.get("by", "user"), "text": text})
            self.say("Done, working on it." if state == "approved" else "Okay, I won't.")
            return {"request": rid, "state": state}
        if cmd == "mode":  # "VidAI, take all actions" / "VidAI, ask me first"
            from .. import actions

            mode = "full" if c.get("mode") == "full" else "ask"
            actions.set_mode(self.dir, mode)
            self.bus.publish("permission_mode", {"mode": mode})
            self.say("Full access. I will take all actions needed." if mode == "full"
                     else "Okay, I will ask you first.")
            if mode == "full":  # anything already waiting is approved too
                for rid in list(self.asks):
                    self.asks.pop(rid)
                    self.bus.publish("permission", {"request": rid, "state": "approved", "by": "full_access"})
            return {"mode": mode}
        if cmd == "thinking":  # Claude shows the icon itself while working on something longer
            self._show_thinking(bool(c.get("on", True)))
            return {"thinking": bool(c.get("on", True))}
        if cmd == "status":
            return self.status()
        raise ValueError(f"unknown command {cmd!r}")

    def _remember_session(self) -> None:
        import collections
        import re as _re

        for p in list(self._probation):  # keep what was not removed by the end
            p["t_done"] = -1e9
        self._check_probation()
        stop = {"that", "this", "with", "have", "from", "will", "what", "your", "they", "about", "there", "then",
                "vidai", "here", "just", "like", "going", "want", "make", "more", "some", "into", "them"}
        words = collections.Counter(w for tr in (self.stt.transcripts if self.stt else [])
                                    for w in _re.findall(r"[A-Za-z][A-Za-z0-9\-]{3,}", tr["text"])
                                    if w.lower() not in stop)
        self.profile.add_words([w for w, n in words.items() if n >= 2 or w.isupper()])
        reqs = self.requests_log
        self.profile.add_history({
            "session": str(self.dir or ""), "duration": round(self.clock(), 1),
            "requests": [r["msg"] for r in reqs],
            "answered_by": dict(collections.Counter(r["via"] for r in reqs)),
            "removed_quickly": [r["msg"] for r in reqs if r["removed"]],
            "errors": sum(1 for e in self.bus.history if e["kind"] == "error"),
            "learned": self.learned,
        })

    def say(self, text: str) -> None:
        """VidAI speaks (text-to-speech) and logs it, so the editor can cut these moments later."""
        self.bus.publish("action", {"what": "vidai_said", "text": text})
        if self.live.speak and shutil.which("spd-say"):
            try:
                subprocess.Popen(["spd-say", "-r", "5", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except OSError:
                pass

    def carry_specs(self) -> list[dict]:
        """Lasting effects that are on now (added in preview) -> re-added when the recording starts."""
        out = []
        for p in self.chain.items:
            spec = getattr(p, "spec", None)
            if spec and p.enabled and p.until is None and not p.name.startswith("_"):
                out.append({**spec, "params": dict(p.params)})
        return out

    def status(self) -> dict:
        return {"t": round(self.clock(), 2), "frames": self.frames, "loop_ms": round(self.loop_ms, 2),
                "processors": [p.describe() for p in self.chain.items], "rules": list(self.rules.rules.values()),
                "learners": [l.describe() for l in self.learners.values()],
                "stt": bool(self.stt and self.stt.enabled), "ocr": bool(self.ocr and self.ocr.enabled),
                "pending_requests": [{"request": k, **v} for k, v in self.pending.items()]}

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
                mux_parts(self.video_part, self.audio_part, self.output)
            except Exception as e:
                self.error = f"mux: {e}"
                self.bus.publish("error", {"where": "mux", "error": str(e)[:300]})
        self.audio.finish()
        if self.stt:
            self.stt.flush()
            self.stt.close()
        if self.ocr:
            self.ocr.close()
        for l in self.learners.values():
            l.finish(self.clock())
        if self.ctx.tracks is not None:
            self.ctx.tracks.close()
        if self.output:
            self._remember_session()
        self.bus.publish("action", {"what": "stopped", "duration": round(self.clock(), 2)})
        self.bus.close()
        shutil.rmtree(self._tmp, ignore_errors=True)


def mux_parts(video_part: str | Path | None, audio_part: str | Path | None, output: str | Path) -> Path:
    """Join the separately recorded video and audio tracks into the final file (stream copy, lossless),
    then delete the parts. Tolerates truncated parts (crash recovery)."""
    output = Path(output)
    vp = Path(video_part) if video_part else None
    ap = Path(audio_part) if audio_part else None
    if not vp or not vp.exists() or vp.stat().st_size < 1024:
        raise FileNotFoundError(f"no recorded video in {vp}")
    args = ["-err_detect", "ignore_err", "-i", str(vp)]
    has_a = bool(ap and ap.exists() and ap.stat().st_size > 256)
    if has_a:
        args += ["-err_detect", "ignore_err", "-i", str(ap)]
    tmp = output.with_name(output.stem + ".muxing.mkv")
    args += ["-map", "0:v"] + (["-map", "1:a"] if has_a else []) + ["-c", "copy", str(tmp)]
    ffmpeg.run(args)
    os.replace(tmp, output)
    vp.unlink(missing_ok=True)
    if ap:
        ap.unlink(missing_ok=True)
    return output
