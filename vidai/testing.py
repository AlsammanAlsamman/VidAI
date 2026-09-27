"""Synthetic test media (also handy for demos)."""
from __future__ import annotations

from pathlib import Path

from . import ffmpeg

# tone on: 0-5, 8-14, 16-24, 27-30  ->  silences: 5-8, 14-16, 24-27
SILENCES = [(5.0, 8.0), (14.0, 16.0), (24.0, 27.0)]
SCENE_CUT = 15.0
DURATION = 30.0


def make_test_video(path: str | Path, w: int = 640, h: int = 360, fps: int = 30) -> Path:
    path = Path(path)
    half = DURATION / 2
    gate = "(lt(t,5)+between(t,8,14)+between(t,16,24)+gte(t,27))"
    ffmpeg.run([
        "-f", "lavfi", "-i", f"testsrc2=s={w}x{h}:r={fps}:d={half}",
        "-f", "lavfi", "-i", f"color=c=0x2050C0:s={w}x{h}:r={fps}:d={half}",
        "-f", "lavfi", "-i", f"aevalsrc='0.4*sin(2*PI*440*t)*{gate}':s=48000:d={DURATION}",
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
        "-map", "[v]", "-map", "2:a", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(path),
    ])
    return path
