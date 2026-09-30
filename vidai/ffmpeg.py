"""Locate ffmpeg and run it. Prefers a system ffmpeg, falls back to the bundled imageio-ffmpeg binary."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def ffmpeg_exe() -> str:
    env = os.environ.get("VIDAI_FFMPEG")
    if env:
        return env
    system = shutil.which("ffmpeg")
    if system:
        return system
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


@lru_cache(maxsize=1)
def available_filters() -> frozenset[str]:
    out = subprocess.run([ffmpeg_exe(), "-hide_banner", "-filters"], capture_output=True, text=True).stdout
    names = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 3 and "->" in parts[2]:
            names.add(parts[1])
    return frozenset(names)


class FFmpegError(RuntimeError):
    pass


def run(args: list[str], quiet: bool = True) -> subprocess.CompletedProcess:
    cmd = [ffmpeg_exe(), "-hide_banner", "-y"]
    if quiet:
        cmd += ["-loglevel", "error"]
    proc = subprocess.run(cmd + args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise FFmpegError(f"ffmpeg failed ({proc.returncode}): {' '.join(args)}\n{proc.stderr[-2000:]}")
    return proc


@dataclass
class MediaInfo:
    duration: float
    width: int = 0
    height: int = 0
    fps: float = 0.0
    has_video: bool = False
    has_audio: bool = False
    sample_rate: int = 0


def _scan_duration(path: str) -> float:
    """Duration by reading every packet (no decoding), for files whose header has none."""
    proc = subprocess.run([ffmpeg_exe(), "-hide_banner", "-nostdin", "-i", path, "-map", "0", "-c", "copy",
                           "-f", "null", "-"], capture_output=True, text=True)
    ts = re.findall(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)", proc.stderr)
    dur = int(ts[-1][0]) * 3600 + int(ts[-1][1]) * 60 + float(ts[-1][2]) if ts else 0.0
    if dur <= 0:
        raise FFmpegError(f"cannot probe {path}: the file has no duration and no readable packets "
                          f"(damaged or empty recording?)\n{proc.stderr[-1000:]}")
    return dur


def probe(path: str) -> MediaInfo:
    """Read basic media info from `ffmpeg -i` output (ffprobe is not always available)."""
    proc = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", path], capture_output=True, text=True)
    err = proc.stderr
    m = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", err)
    if m:
        h, mi, s = m.groups()
        info = MediaInfo(duration=int(h) * 3600 + int(mi) * 60 + float(s))
    elif "Duration: N/A" in err:  # e.g. a recording that crashed before its index/duration was written
        info = MediaInfo(duration=_scan_duration(path))
    else:
        raise FFmpegError(f"cannot probe {path}:\n{err[-1000:]}")
    v = re.search(r"Stream #.*Video:.*?(\d{2,5})x(\d{2,5}).*?(\d+(?:\.\d+)?) fps", err)
    if v:
        info.has_video = True
        info.width, info.height, info.fps = int(v.group(1)), int(v.group(2)), float(v.group(3))
    elif "Video:" in err:
        info.has_video = True
    a = re.search(r"Stream #.*Audio:.*?(\d+) Hz", err)
    if a:
        info.has_audio = True
        info.sample_rate = int(a.group(1))
    return info


def read_audio(path: str, sr: int = 16000, start: float = 0.0, duration: float | None = None) -> np.ndarray:
    """Decode mono float32 PCM."""
    args = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error"]
    if start:
        args += ["-ss", f"{start:.3f}"]
    args += ["-i", path]
    if duration is not None:
        args += ["-t", f"{duration:.3f}"]
    args += ["-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    proc = subprocess.run(args, capture_output=True)
    if proc.returncode != 0:
        raise FFmpegError(proc.stderr.decode(errors="replace")[-1000:])
    return np.frombuffer(proc.stdout, dtype=np.float32)


def read_frames(path: str, fps: float = 2.0, width: int = 160, height: int = 90,
                start: float = 0.0, duration: float | None = None, keyframes_only: bool = False) -> np.ndarray:
    """Decode small grayscale frames as an array (n, height, width) of uint8.

    keyframes_only: decode only keyframes (~10x faster; the fps filter repeats them on the regular grid,
    so changes are located to the keyframe interval, ~1-2 s for OBS recordings)."""
    args = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error"]
    if keyframes_only:
        args += ["-skip_frame", "nokey"]
    if start:
        args += ["-ss", f"{start:.3f}"]
    args += ["-i", path]
    if duration is not None:
        args += ["-t", f"{duration:.3f}"]
    args += ["-an", "-vf", f"fps={fps},scale={width}:{height}", "-pix_fmt", "gray", "-f", "rawvideo", "-"]
    proc = subprocess.run(args, capture_output=True)
    if proc.returncode != 0:
        raise FFmpegError(proc.stderr.decode(errors="replace")[-1000:])
    buf = np.frombuffer(proc.stdout, dtype=np.uint8)
    n = buf.size // (width * height)
    return buf[: n * width * height].reshape(n, height, width)


def extract_frame(path: str, t: float, out_png: str, width: int = 640) -> str:
    run(["-ss", f"{t:.3f}", "-i", path, "-frames:v", "1", "-vf", f"scale={width}:-2", out_png])
    return out_png


@lru_cache(maxsize=1)
def hw_encoder() -> list[str] | None:
    """Intel/AMD GPU H.264 encoding (VAAPI) when this ffmpeg supports it and a render device exists.
    The bundled static ffmpeg does not; a system ffmpeg (`sudo apt install ffmpeg`) usually does."""
    import os as _os

    dev = "/dev/dri/renderD128"
    if not _os.path.exists(dev):
        return None
    out = subprocess.run([ffmpeg_exe(), "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    if "h264_vaapi" not in out:
        return None
    probe = subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-vaapi_device", dev, "-f", "lavfi",
                            "-i", "color=s=320x240:d=0.2", "-vf", "format=nv12,hwupload", "-c:v", "h264_vaapi",
                            "-f", "null", "-"], capture_output=True)
    if probe.returncode != 0:
        return None
    return ["-vaapi_device", dev, "-vf", "format=nv12,hwupload", "-c:v", "h264_vaapi", "-qp", "21"]
