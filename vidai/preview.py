"""Cheap previews for Claude: single frames, contact sheets at anchor times, frame features for lab models."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw

from . import ffmpeg


def frame(video: str, t: float, out: str | Path | None = None, width: int = 640) -> Path:
    out = Path(out) if out else Path(video).with_name(f"{Path(video).stem}_t{t:.2f}.png")
    ffmpeg.extract_frame(video, t, str(out), width)
    return out


def contact_sheet(video: str, times: list[float], out: str | Path | None = None, cols: int = 4,
                  width: int = 320) -> Path:
    """One image with frames at the given times, each labelled with its timestamp."""
    out = Path(out) if out else Path(video).with_name(f"{Path(video).stem}_sheet.png")
    tiles = []
    with tempfile.TemporaryDirectory() as td:
        for i, t in enumerate(times):
            p = Path(td) / f"{i}.png"
            ffmpeg.extract_frame(video, t, str(p), width)
            im = PILImage.open(p).convert("RGB")
            d = ImageDraw.Draw(im)
            d.rectangle([0, 0, 90, 22], fill=(0, 0, 0))
            d.text((5, 4), f"{t:.2f}s", fill=(255, 255, 0))
            tiles.append(im)
    if not tiles:
        raise ValueError("no times given")
    w, h = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    sheet = PILImage.new("RGB", (w * min(cols, len(tiles)), h * rows), (20, 20, 20))
    for i, im in enumerate(tiles):
        sheet.paste(im, ((i % cols) * w, (i // cols) * h))
    sheet.save(out)
    return out


def frame_features(video: str, fps: float = 2.0, size: tuple[int, int] = (32, 18)) -> tuple[np.ndarray, np.ndarray]:
    """Small grayscale frames flattened as features, with their times. Input for frame classifiers."""
    fr = ffmpeg.read_frames(video, fps, size[0], size[1]).astype(np.float32) / 255.0
    return fr.reshape(len(fr), -1), np.arange(len(fr)) / fps
