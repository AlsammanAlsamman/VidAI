"""VidAI's own recorder: camera, screen, or screen + camera (picture-in-picture), with microphone.

One ffmpeg process writes the recording (MKV, crash-safe) and at the same time streams two small taps
to Python through FIFOs:
  - video tap: small RGB frames (GUI preview + live motion / scene-change anchors)
  - audio tap: 16 kHz mono float (live level meter + audio-level / silence anchors)
So the anchors are computed *while* recording; nothing needs to be re-decoded afterwards.

Audio: the bundled static ffmpeg cannot use PulseAudio/PipeWire directly, so the mic is read with
`parec` (PipeWire/Pulse) and piped into ffmpeg. A system ffmpeg with pulse support is used directly.
"""
from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Literal

import numpy as np
from pydantic import BaseModel

from . import ffmpeg, native
from .anchors import Series

Mode = Literal["camera", "screen", "screen+camera", "test"]

TAP_W, TAP_H, TAP_FPS = 320, 180, 6.0
TAP_SR = 16000
LEVEL_HZ = 10.0
MOTION_HZ = 2.0


class CaptureConfig(BaseModel):
    mode: Mode = "screen+camera"
    camera: str = "/dev/video0"
    camera_size: str = "1280x720"
    camera_format: str = "mjpeg"
    display: str = ""  # default: $DISPLAY
    screen_size: str = ""  # default: detected
    fps: int = 30
    out_height: int = 1080
    pip_width: float = 0.22  # camera width as a fraction of the output (screen+camera)
    pip_position: Literal["bottom-right", "bottom-left", "top-right", "top-left"] = "bottom-right"
    mic: bool = True
    mic_source: str = ""  # pulse source name, "" = default
    audio_offset: float = 0.0  # seconds, + delays audio (fix lip sync)
    preset: str = "veryfast"
    crf: int = 21


# ---------------- devices ----------------
def list_cameras() -> list[dict]:
    cams = []
    for dev in sorted(Path("/dev").glob("video*")):
        r = subprocess.run([ffmpeg.ffmpeg_exe(), "-hide_banner", "-f", "v4l2", "-list_formats", "all", "-i", str(dev)],
                           capture_output=True, text=True, timeout=10)
        fmts = {}
        for line in r.stderr.splitlines():
            m = re.search(r"(Raw|Compressed)\s*:\s*(\w+)\s*:", line)
            if m:
                fmts[m.group(2)] = re.findall(r"\d+x\d+", line)
        if fmts:
            cams.append({"device": str(dev), "formats": fmts})
    return cams


def screen_size(display: str = "") -> str:
    env = dict(os.environ, DISPLAY=display or os.environ.get("DISPLAY", ":0"))
    if shutil.which("xdpyinfo"):
        r = subprocess.run(["xdpyinfo"], capture_output=True, text=True, env=env)
        m = re.search(r"dimensions:\s+(\d+x\d+)", r.stdout)
        if m:
            return m.group(1)
    return "1920x1080"


def ffmpeg_has_pulse() -> bool:
    r = subprocess.run([ffmpeg.ffmpeg_exe(), "-hide_banner", "-devices"], capture_output=True, text=True)
    return bool(re.search(r"\bpulse\b", r.stdout))


def devices() -> dict:
    return {"cameras": list_cameras(), "screen": screen_size(), "display": os.environ.get("DISPLAY"),
            "mic_via": "ffmpeg-pulse" if ffmpeg_has_pulse() else ("parec" if shutil.which("parec") else None)}


# ---------------- command building ----------------
def _even(x: float) -> int:
    return int(x) // 2 * 2


def build_command(cfg: CaptureConfig, output: str | None, vtap: str, atap: str, audio_from_stdin: bool) -> list[str]:
    exe = ffmpeg.ffmpeg_exe()
    args = [exe, "-hide_banner", "-loglevel", "error", "-y"]
    tq = ["-thread_queue_size", "1024"]
    inputs: dict[str, int] = {}

    def add(name: str, a: list[str]) -> None:
        inputs[name] = len(inputs)
        args.extend(a)

    if cfg.mode == "test":
        add("screen", ["-re", "-f", "lavfi", "-i", f"testsrc2=s=1280x720:r={cfg.fps}"])
        if cfg.mic:
            add("mic", ["-re", "-f", "lavfi", "-i",
                        "aevalsrc='0.3*sin(2*PI*330*t)*gt(mod(t,4),1.5)':s=48000"])
    else:
        if cfg.mode in ("screen", "screen+camera"):
            add("screen", tq + ["-f", "x11grab", "-framerate", str(cfg.fps),
                                "-video_size", cfg.screen_size or screen_size(cfg.display),
                                "-draw_mouse", "1", "-i", cfg.display or os.environ.get("DISPLAY", ":0")])
        if cfg.mode in ("camera", "screen+camera"):
            add("camera", tq + ["-f", "v4l2", "-input_format", cfg.camera_format, "-video_size", cfg.camera_size,
                                "-framerate", str(cfg.fps), "-i", cfg.camera])
        if cfg.mic:
            if audio_from_stdin:
                add("mic", tq + ["-f", "s16le", "-ar", "48000", "-ac", "2", "-i", "pipe:0"])
            else:
                add("mic", tq + ["-f", "pulse", "-i", cfg.mic_source or "default"])

    H = cfg.out_height
    g = []
    if "screen" in inputs and "camera" in inputs:
        pw = _even(cfg.pip_width * H * 16 / 9)
        m = 24
        x = f"W-w-{m}" if "right" in cfg.pip_position else str(m)
        y = f"H-h-{m}" if "bottom" in cfg.pip_position else str(m)
        g.append(f"[{inputs['screen']}:v]scale=-2:{H},setsar=1[s]")
        g.append(f"[{inputs['camera']}:v]scale={pw}:-2,setsar=1[c]")
        g.append(f"[s][c]overlay={x}:{y}:shortest=0,format=yuv420p[main]")
    else:
        src = inputs.get("screen", inputs.get("camera"))
        g.append(f"[{src}:v]scale=-2:{H},setsar=1,format=yuv420p[main]")
    g.append("[main]split[rec][tapin]")
    g.append(f"[tapin]fps={TAP_FPS},scale={TAP_W}:{TAP_H},format=rgb24[vtap]")
    if "mic" in inputs:
        delay = f"adelay={int(cfg.audio_offset * 1000)}:all=1," if cfg.audio_offset > 0 else ""
        split = "asplit[arec][atapin]" if output else "anull[atapin]"  # preview: no recorded audio branch
        g.append(f"[{inputs['mic']}:a]{delay}aresample=48000,aformat=channel_layouts=stereo,{split}")
        g.append(f"[atapin]aresample={TAP_SR},aformat=sample_fmts=flt:channel_layouts=mono[atap]")
    args += ["-filter_complex", ";".join(g)]

    if output:
        args += ["-map", "[rec]"] + (["-map", "[arec]"] if "mic" in inputs else [])
        args += ["-c:v", "libx264", "-preset", cfg.preset, "-crf", str(cfg.crf), "-r", str(cfg.fps),
                 "-g", str(cfg.fps * 2)]
        if "mic" in inputs:
            args += ["-c:a", "aac", "-b:a", "192k"]
        # crash safety: write every packet and close a Matroska cluster at least once per second,
        # so a crash loses at most ~1 s and the file stays repairable
        args += ["-flush_packets", "1", "-cluster_time_limit", "1000", output]
    else:
        args += ["-map", "[rec]", "-f", "null", "-"]
    args += ["-map", "[vtap]", "-f", "rawvideo", vtap]
    if "mic" in inputs:
        args += ["-map", "[atap]", "-f", "f32le", atap]
    return args


# ---------------- live capture ----------------
class LiveCapture:
    """Start/stop a capture. With output=None it only previews (nothing is saved)."""

    def __init__(self, cfg: CaptureConfig, output: str | Path | None = None,
                 on_frame: Callable[[np.ndarray], None] | None = None) -> None:
        self.cfg = cfg
        self.output = str(output) if output else None
        self.on_frame = on_frame
        self.proc: subprocess.Popen | None = None
        self.mic_proc: subprocess.Popen | None = None
        self.t_first: float | None = None  # monotonic time of the first tapped frame (≈ video t=0)
        self.level_db = -90.0
        self.error: str | None = None
        self._levels: list[float] = []
        self._motion: list[float] = []
        self._prev_small: np.ndarray | None = None
        self._frames = 0
        self._threads: list[threading.Thread] = []
        self._tmp = tempfile.mkdtemp(prefix="vidai_cap_")

    # clock shared with live samplers (markers, input activity, ...)
    def clock(self) -> float:
        return time.monotonic() - self.t_first if self.t_first else 0.0

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self) -> None:
        vtap, atap = os.path.join(self._tmp, "v.fifo"), os.path.join(self._tmp, "a.fifo")
        os.mkfifo(vtap)
        os.mkfifo(atap)
        use_parec = self.cfg.mic and self.cfg.mode != "test" and not ffmpeg_has_pulse()
        if use_parec and not shutil.which("parec"):
            raise RuntimeError("no way to record the microphone: install pulseaudio-utils (parec) or set mic=False")
        cmd = build_command(self.cfg, self.output, vtap, atap, use_parec)
        has_audio = self.cfg.mic

        self._threads = [threading.Thread(target=self._read_video, args=(vtap,), daemon=True)]
        if has_audio:
            self._threads.append(threading.Thread(target=self._read_audio, args=(atap,), daemon=True))
        for t in self._threads:
            t.start()
        if use_parec:
            pa = ["parec", "--raw", "--format=s16le", "--rate=48000", "--channels=2", "--latency-msec=20"]
            if self.cfg.mic_source:
                pa.append(f"--device={self.cfg.mic_source}")
            self.mic_proc = subprocess.Popen(pa, stdout=subprocess.PIPE)
            stdin = self.mic_proc.stdout
        else:
            stdin = subprocess.PIPE
        self.proc = subprocess.Popen(cmd, stdin=stdin, stderr=subprocess.PIPE)
        threading.Thread(target=self._watch_stderr, daemon=True).start()

    def _watch_stderr(self) -> None:
        err = self.proc.stderr.read().decode(errors="replace") if self.proc and self.proc.stderr else ""
        if err.strip():
            self.error = err[-2000:]

    def _read_video(self, path: str) -> None:
        size = TAP_W * TAP_H * 3
        step = max(1, int(round(TAP_FPS / MOTION_HZ)))
        with open(path, "rb") as f:
            while True:
                buf = f.read(size)
                if len(buf) < size:
                    break
                if self.t_first is None:
                    self.t_first = time.monotonic()
                frame = np.frombuffer(buf, np.uint8).reshape(TAP_H, TAP_W, 3)
                if self._frames % step == 0:
                    small = np.ascontiguousarray(frame[::2, ::2].mean(axis=2).astype(np.uint8))
                    if self._prev_small is None:
                        self._motion.append(0.0)
                    else:
                        self._motion.append(float(native.frame_mad(np.stack([self._prev_small, small]))[1]))
                    self._prev_small = small
                self._frames += 1
                if self.on_frame:
                    self.on_frame(frame)

    def _read_audio(self, path: str) -> None:
        hop = int(TAP_SR / LEVEL_HZ)
        nbytes = hop * 4
        with open(path, "rb") as f:
            while True:
                buf = f.read(nbytes)
                if len(buf) < nbytes:
                    break
                db = float(native.rms_db(np.frombuffer(buf, np.float32), hop)[0])
                self._levels.append(round(db, 1))
                self.level_db = db

    def stop(self, timeout: float = 15.0) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                if self.proc.stdin:
                    self.proc.stdin.write(b"q")
                    self.proc.stdin.flush()
                else:
                    self.proc.send_signal(signal.SIGINT)
            except (BrokenPipeError, OSError):
                self.proc.send_signal(signal.SIGINT)
            if self.mic_proc:
                self.mic_proc.terminate()  # closes ffmpeg's audio input -> ffmpeg finishes
            try:
                self.proc.wait(timeout)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        if self.mic_proc and self.mic_proc.poll() is None:
            self.mic_proc.kill()
        for t in self._threads:
            t.join(3)
        shutil.rmtree(self._tmp, ignore_errors=True)

    def live_series(self) -> dict[str, Series]:
        out = {}
        if self._levels:
            out["audio_level"] = Series(rate_hz=LEVEL_HZ, values=list(self._levels))
        if self._motion:
            out["motion"] = Series(rate_hz=MOTION_HZ, values=[round(v, 4) for v in self._motion])
        return out


# ---------------- shared by the live pipeline ----------------
def canvas_size(cfg: CaptureConfig) -> tuple[int, int]:
    """Fixed 16:9 output canvas (e.g. 1920x1080), whatever the sources are."""
    H = _even(cfg.out_height)
    return _even(H * 16 / 9), H


def capture_inputs(cfg: CaptureConfig, audio_from_stdin: bool) -> tuple[list[str], list[str], dict[str, int]]:
    """ffmpeg input args + filter graph producing [main] (W x H, rgb24, constant fps) and [mic] audio."""
    args: list[str] = []
    tq = ["-thread_queue_size", "1024"]
    inputs: dict[str, int] = {}

    def add(name: str, a: list[str]) -> None:
        inputs[name] = len(inputs)
        args.extend(a)

    if cfg.mode == "test":
        add("screen", ["-re", "-f", "lavfi", "-i", f"testsrc2=s=1280x720:r={cfg.fps}"])
        if cfg.mic:
            src = cfg.mic_source or "aevalsrc='0.3*sin(2*PI*330*t)*gt(mod(t,4),1.5)':s=48000"
            if src.endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg")):  # test speech file (looped)
                add("mic", ["-re", "-stream_loop", "-1", "-i", src])
            else:
                add("mic", ["-re", "-f", "lavfi", "-i", src])
    else:
        if cfg.mode in ("screen", "screen+camera"):
            add("screen", tq + ["-f", "x11grab", "-framerate", str(cfg.fps),
                                "-video_size", cfg.screen_size or screen_size(cfg.display),
                                "-draw_mouse", "1", "-i", cfg.display or os.environ.get("DISPLAY", ":0")])
        if cfg.mode in ("camera", "screen+camera"):
            add("camera", tq + ["-f", "v4l2", "-input_format", cfg.camera_format, "-video_size", cfg.camera_size,
                                "-framerate", str(cfg.fps), "-i", cfg.camera])
        if cfg.mic:
            if audio_from_stdin:
                add("mic", tq + ["-f", "s16le", "-ar", "48000", "-ac", "2", "-i", "pipe:0"])
            else:
                add("mic", tq + ["-f", "pulse", "-i", cfg.mic_source or "default"])

    W, H = canvas_size(cfg)
    fit = f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1"
    g = []
    if "screen" in inputs and "camera" in inputs:
        pw = _even(cfg.pip_width * W)
        m = 24
        x = f"W-w-{m}" if "right" in cfg.pip_position else str(m)
        y = f"H-h-{m}" if "bottom" in cfg.pip_position else str(m)
        g.append(f"[{inputs['screen']}:v]{fit}[s]")
        g.append(f"[{inputs['camera']}:v]scale={pw}:-2,setsar=1[c]")
        g.append(f"[s][c]overlay={x}:{y}:shortest=0,fps={cfg.fps},format=rgb24[main]")
    else:
        src = inputs.get("screen", inputs.get("camera"))
        g.append(f"[{src}:v]{fit},fps={cfg.fps},format=rgb24[main]")
    if "mic" in inputs:
        delay = f"adelay={int(cfg.audio_offset * 1000)}:all=1," if cfg.audio_offset > 0 else ""
        g.append(f"[{inputs['mic']}:a]{delay}aresample=48000,aformat=channel_layouts=stereo[mic]")
    return args, g, inputs
