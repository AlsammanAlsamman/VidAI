"""Render overlays (text, icons, arrows, highlights) to full-frame transparent PNGs with Pillow."""
from __future__ import annotations

import math
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont, features

from .edit import Image, Position, Shape, Text

MARGIN = 0.04
_ARABIC = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")


def parse_color(c: str) -> tuple[int, int, int, int]:
    c = c.lstrip("#")
    if len(c) == 6:
        c += "FF"
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4, 6))  # type: ignore[return-value]


@lru_cache(maxsize=8)
def find_font(arabic: bool = False, bold: bool = True) -> str | None:
    queries = (["Noto Sans Arabic:bold", "Noto Naskh Arabic", "DejaVu Sans:bold"] if arabic
               else ["DejaVu Sans:bold" if bold else "DejaVu Sans", "Liberation Sans:bold", "Noto Sans:bold"])
    if shutil.which("fc-match"):
        for q in queries:
            out = subprocess.run(["fc-match", "-f", "%{file}", q], capture_output=True, text=True).stdout.strip()
            if out and Path(out).exists():
                return out
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"):
        if Path(p).exists():
            return p
    return None


def shape_text(text: str) -> tuple[str, dict]:
    """Arabic needs shaping + RTL. Use libraqm if Pillow has it, else arabic_reshaper + python-bidi if installed."""
    if not _ARABIC.search(text):
        return text, {}
    if features.check("raqm") and features.check("fribidi"):
        return text, {"direction": "rtl"}
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display

        return get_display(arabic_reshaper.reshape(text)), {}
    except ImportError:
        return text, {}


def resolve_position(pos: Position, W: int, H: int, w: int, h: int) -> tuple[int, int]:
    if isinstance(pos, (tuple, list)):
        return int(pos[0] * W - w / 2), int(pos[1] * H - h / 2)
    mx, my = int(MARGIN * W), int(MARGIN * H)
    v, _, hz = pos.partition("-") if "-" in pos else (pos, "", pos)
    if pos == "center":
        v, hz = "center", "center"
    x = {"left": mx, "center": (W - w) // 2, "right": W - w - mx}.get(hz, (W - w) // 2)
    y = {"top": my, "center": (H - h) // 2, "middle": (H - h) // 2, "bottom": H - h - my}.get(v, H - h - my)
    return x, y


def _font(size_px: int, text: str, path: str | None) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = path or find_font(arabic=bool(_ARABIC.search(text)))
    if path:
        return ImageFont.truetype(path, size_px)
    return ImageFont.load_default(size=size_px)


def render_text(op: Text, W: int, H: int) -> PILImage.Image:
    img = PILImage.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    size = max(8, int(op.size * H))
    font = _font(size, op.text, op.font)
    lines = op.text.split("\n")
    shaped = [shape_text(line) for line in lines]
    boxes = [d.textbbox((0, 0), s, font=font, **kw) for s, kw in shaped]
    lh = int(size * 1.25)
    tw = max(b[2] - b[0] for b in boxes)
    th = lh * len(lines)
    pad = int(size * 0.35)
    x, y = resolve_position(op.position, W, H, tw + 2 * pad, th + 2 * pad)
    if op.box:
        d.rounded_rectangle([x, y, x + tw + 2 * pad, y + th + 2 * pad], radius=pad, fill=parse_color(op.box))
    for i, ((s, kw), b) in enumerate(zip(shaped, boxes)):
        lx = x + pad + (tw - (b[2] - b[0])) // 2 - b[0]
        d.text((lx, y + pad + i * lh - b[1] // 2), s, font=font, fill=parse_color(op.color), **kw)
    return img


def render_image(op: Image, W: int, H: int) -> PILImage.Image:
    img = PILImage.new("RGBA", (W, H), (0, 0, 0, 0))
    icon = PILImage.open(op.path).convert("RGBA")
    w = max(1, int(op.width * W))
    h = max(1, int(icon.height * w / icon.width))
    icon = icon.resize((w, h), PILImage.LANCZOS)
    img.alpha_composite(icon, resolve_position(op.position, W, H, w, h))
    return img


def render_shape(op: Shape, W: int, H: int) -> PILImage.Image:
    img = PILImage.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    col = parse_color(op.color)
    lw = max(2, int(op.thickness * H))
    cx, cy = op.x * W, op.y * H
    if op.shape == "circle":
        rx, ry = op.w * W / 2, op.h * H / 2
        d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], outline=col, width=lw)
    elif op.shape == "box":
        rx, ry = op.w * W / 2, op.h * H / 2
        d.rounded_rectangle([cx - rx, cy - ry, cx + rx, cy + ry], radius=lw * 2, outline=col, width=lw)
    else:  # arrow: tip at (x, y), tail `w*W` away in direction `angle`
        length = op.w * W
        a = math.radians(op.angle)
        tx, ty = cx + length * math.cos(a), cy - length * math.sin(a)
        d.line([tx, ty, cx, cy], fill=col, width=lw)
        head = max(lw * 4, length * 0.25)
        back = math.atan2(ty - cy, tx - cx)
        for s in (-0.45, 0.45):
            d.line([cx, cy, cx + head * math.cos(back + s), cy + head * math.sin(back + s)], fill=col, width=lw)
        d.ellipse([cx - lw / 2, cy - lw / 2, cx + lw / 2, cy + lw / 2], fill=col)
    return img


def render_overlay(op: Text | Image | Shape, W: int, H: int, out: str | Path) -> Path:
    fn = {"text": render_text, "image": render_image, "shape": render_shape}[op.op]
    out = Path(out)
    fn(op, W, H).save(out)
    return out
