"""Live processors: code that changes every frame while recording (the fast loop).

Write a new one (Claude does this during a recording):

    from vidai.live.processors import LiveProcessor, register

    @register
    class Spotlight(LiveProcessor):
        '''Darken everything except a circle.'''
        defaults = {"x": 0.5, "y": 0.5, "r": 0.2, "dark": 0.6}
        listens = {"voice_command"}                     # events delivered to on_event

        def on_event(self, ev, ctx):                    # fast loop, called from the bus
            if ev["data"].get("command") == "spotlight off":
                self.enabled = False

        def process(self, frame, t, ctx):               # frame: (H, W, 3) uint8 RGB, edit in place or return new
            ...
            return frame

Rules:
- `process` has a time budget (default 8 ms at 1080p). A processor that is too slow or raises is
  disabled automatically (the recording never stops) and an `error` event tells Claude why.
- Use NumPy/OpenCV (C) and `vidai.native` kernels; never loop over pixels in Python.
- `ctx.stats` = latest value of every live stat, `ctx.bus.publish(...)` to emit your own events,
  `ctx.overlay(...)` renders text/shapes/images to cached RGBA overlays.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from .. import native

REGISTRY: dict[str, type["LiveProcessor"]] = {}


def register(cls: type["LiveProcessor"]) -> type["LiveProcessor"]:
    REGISTRY[cls.__name__.lower()] = cls
    REGISTRY[getattr(cls, "type_name", cls.__name__.lower())] = cls
    return cls


class LiveProcessor:
    type_name = ""
    defaults: dict[str, Any] = {}
    listens: set[str] = set()
    budget_ms: float = 8.0
    stage: int = 1  # 0 = transforms the picture (zoom, blur, models) — runs first; 1 = overlays on top

    def __init__(self, name: str, params: dict[str, Any] | None = None, enabled: bool = True,
                 until: float | None = None) -> None:
        self.name = name
        self.params = {**self.defaults, **(params or {})}
        self.enabled = enabled
        self.until = until  # auto-disable at this recording time (s)
        self.slow_frames = 0
        self.total_ms = 0.0
        self.calls = 0

    def configure(self, params: dict[str, Any]) -> None:
        self.params.update(params)

    def on_event(self, ev: dict, ctx: "Context") -> None: ...

    def process(self, frame: np.ndarray, t: float, ctx: "Context") -> np.ndarray:
        return frame

    def describe(self) -> dict:
        return {"name": self.name, "type": type(self).__name__, "enabled": self.enabled, "params": self.params,
                "until": self.until, "avg_ms": round(self.total_ms / self.calls, 2) if self.calls else None}


class Context:
    """What processors see: frame size, live stats, the bus and an overlay cache."""

    def __init__(self, bus, width: int, height: int) -> None:
        self.bus, self.width, self.height = bus, width, height
        self._cache: dict[Any, np.ndarray] = {}

    @property
    def stats(self) -> dict[str, Any]:
        return self.bus.snapshot()

    def overlay(self, op) -> np.ndarray:
        """Render a vidai.edit Text/Shape/Image op to a full-frame RGBA array (cached)."""
        from ..overlays import render_image, render_shape, render_text

        key = op.model_dump_json()
        if key not in self._cache:
            fn = {"text": render_text, "shape": render_shape, "image": render_image}[op.op]
            self._cache[key] = np.asarray(fn(op, self.width, self.height).convert("RGBA"))
            if len(self._cache) > 64:
                self._cache.pop(next(iter(self._cache)))
        return self._cache[key]


def _fade(t: float, start: float, until: float | None, fade: float) -> float:
    a = min(1.0, (t - start) / fade) if fade > 0 else 1.0
    if until is not None and fade > 0:
        a = min(a, max(0.0, (until - t) / fade))
    return max(0.0, a)


# ---------------- built-in processors ----------------
@register
class Text(LiveProcessor):
    """Text card / lower-third. params: text, position, size, color, box, fade."""
    type_name = "text"
    defaults = {"text": "", "position": "bottom-center", "size": 0.05, "color": "#FFFFFF", "box": "#000000A0",
                "fade": 0.25}

    def __init__(self, *a, **k) -> None:
        super().__init__(*a, **k)
        self.start = None

    def process(self, frame, t, ctx):
        if not self.params["text"]:
            return frame
        from ..edit import Text as TextOp

        if self.start is None:
            self.start = t
        op = TextOp(start=0, end=1, text=self.params["text"], position=self.params["position"],
                    size=self.params["size"], color=self.params["color"], box=self.params["box"])
        native.alpha_blend(frame, ctx.overlay(op), 0, 0, _fade(t, self.start, self.until, self.params["fade"]))
        return frame

    def configure(self, params):
        super().configure(params)
        self.start = None


@register
class Shape(LiveProcessor):
    """Arrow / circle / box highlight. params like the edit Shape op."""
    type_name = "shape"
    defaults = {"shape": "circle", "x": 0.5, "y": 0.5, "w": 0.15, "h": 0.15, "angle": 225.0, "color": "#FF3B30",
                "thickness": 0.008}

    def process(self, frame, t, ctx):
        from ..edit import Shape as ShapeOp

        native.alpha_blend(frame, ctx.overlay(ShapeOp(start=0, end=1, **self.params)))
        return frame


@register
class Image(LiveProcessor):
    """Logo / sticker / icon. params: path, position, width, opacity."""
    type_name = "image"
    defaults = {"path": "", "position": "top-right", "width": 0.12, "opacity": 1.0}

    def process(self, frame, t, ctx):
        if not self.params["path"]:
            return frame
        from ..edit import Image as ImageOp

        op = ImageOp(start=0, end=1, path=self.params["path"], position=self.params["position"],
                     width=self.params["width"])
        native.alpha_blend(frame, ctx.overlay(op), 0, 0, self.params["opacity"])
        return frame


@register
class Zoom(LiveProcessor):
    """Smooth zoom into a region. params: x, y, w, h (fractions), ease (seconds)."""
    type_name = "zoom"
    stage = 0
    defaults = {"x": 0.25, "y": 0.25, "w": 0.5, "h": 0.5, "ease": 0.4}

    def __init__(self, *a, **k) -> None:
        super().__init__(*a, **k)
        self.cur = np.array([0.0, 0.0, 1.0, 1.0])
        self.last_t = None

    def process(self, frame, t, ctx):
        import cv2

        H, W = frame.shape[:2]
        target = np.array([self.params["x"], self.params["y"], self.params["w"], self.params["h"]], float)
        if self.until is not None and t > self.until - self.params["ease"]:
            target = np.array([0.0, 0.0, 1.0, 1.0])
        dt = 0.033 if self.last_t is None else max(0.0, t - self.last_t)
        self.last_t = t
        k = 1.0 if self.params["ease"] <= 0 else min(1.0, dt / self.params["ease"] * 3)
        self.cur += (target - self.cur) * k
        x, y, w, h = self.cur
        if w > 0.995 and h > 0.995:
            return frame
        x0, y0 = int(np.clip(x, 0, 1) * W), int(np.clip(y, 0, 1) * H)
        x1, y1 = min(W, x0 + max(16, int(w * W))), min(H, y0 + max(16, int(h * H)))
        return cv2.resize(frame[y0:y1, x0:x1], (W, H), interpolation=cv2.INTER_LINEAR)


@register
class Blur(LiveProcessor):
    """Hide a region (passwords, emails, faces). params: x, y, w, h, mode (pixelate|blur), strength."""
    type_name = "blur"
    stage = 0
    defaults = {"x": 0.0, "y": 0.0, "w": 0.3, "h": 0.1, "mode": "pixelate", "strength": 16}

    def process(self, frame, t, ctx):
        import cv2

        H, W = frame.shape[:2]
        p = self.params
        x0, y0 = int(p["x"] * W), int(p["y"] * H)
        x1, y1 = min(W, x0 + int(p["w"] * W)), min(H, y0 + int(p["h"] * H))
        if x1 <= x0 or y1 <= y0:
            return frame
        roi = frame[y0:y1, x0:x1]
        s = max(2, int(p["strength"]))
        if p["mode"] == "blur":
            roi[:] = cv2.GaussianBlur(roi, (0, 0), s)
        else:
            small = cv2.resize(roi, (max(1, roi.shape[1] // s), max(1, roi.shape[0] // s)), interpolation=cv2.INTER_AREA)
            roi[:] = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
        return frame


@register
class Captions(LiveProcessor):
    """Live subtitles from the speech-to-text stream. params: size, position, hold (s), max_chars."""
    type_name = "captions"
    defaults = {"size": 0.045, "position": "bottom-center", "hold": 3.5, "max_chars": 60, "color": "#FFFFFF",
                "box": "#000000B0"}
    listens = {"transcript"}

    def __init__(self, *a, **k) -> None:
        super().__init__(*a, **k)
        self.text, self.shown_at = "", -1e9

    def on_event(self, ev, ctx):
        txt = ev["data"].get("text", "").strip()
        if txt and not ev["data"].get("is_command"):
            n = self.params["max_chars"]
            words, lines, cur = txt.split(), [], ""
            for w in words:  # wrap to 2 lines max, keep the end
                if len(cur) + len(w) + 1 > n:
                    lines.append(cur)
                    cur = w
                else:
                    cur = f"{cur} {w}".strip()
            lines.append(cur)
            self.text, self.shown_at = "\n".join(lines[-2:]), ev["t"]

    def process(self, frame, t, ctx):
        if not self.text or t - self.shown_at > self.params["hold"]:
            return frame
        from ..edit import Text as TextOp

        op = TextOp(start=0, end=1, text=self.text, position=self.params["position"], size=self.params["size"],
                    color=self.params["color"], box=self.params["box"])
        native.alpha_blend(frame, ctx.overlay(op))
        return frame


@register
class ColorModel(LiveProcessor):
    """Apply a lab frame-transform model live (e.g. a trained ColorMatch). params: model, strength."""
    type_name = "model"
    stage = 0
    defaults = {"model": "", "strength": 1.0}
    budget_ms = 20.0

    def __init__(self, *a, **k) -> None:
        super().__init__(*a, **k)
        self._m = None

    def process(self, frame, t, ctx):
        if self._m is None:
            from ..lab import load_model

            self._m = load_model(self.params["model"])[0]
        return self._m.transform_frame(frame, t=t, strength=self.params["strength"])


class ProcessorChain:
    """Runs enabled processors in order with a time budget; disables the ones that misbehave."""

    def __init__(self, ctx: Context, scale: float = 1.0) -> None:
        self.ctx = ctx
        self.items: list[LiveProcessor] = []
        self.scale = scale  # budget scale (e.g. smaller frames -> same budget)

    def add(self, p: LiveProcessor) -> LiveProcessor:
        self.remove(p.name)
        self.items.append(p)
        self.items.sort(key=lambda q: q.stage)  # stable: transforms first, then overlays in insertion order
        if p.listens:
            self.ctx.bus.subscribe(lambda ev, p=p: p in self.items and p.on_event(ev, self.ctx), p.listens)
        return p

    def remove(self, name: str) -> bool:
        n = len(self.items)
        self.items = [p for p in self.items if p.name != name]
        return len(self.items) != n

    def get(self, name: str) -> LiveProcessor | None:
        return next((p for p in self.items if p.name == name), None)

    def run(self, frame: np.ndarray, t: float) -> np.ndarray:
        for p in list(self.items):
            if not p.enabled:
                continue
            if p.until is not None and t >= p.until:
                p.enabled = False
                self.ctx.bus.publish("action", {"what": "processor_ended", "name": p.name})
                continue
            t0 = time.perf_counter()
            try:
                out = p.process(frame, t, self.ctx)
                if out is not None:
                    frame = out
            except Exception as e:
                p.enabled = False
                self.ctx.bus.publish("error", {"processor": p.name, "error": repr(e)[:300], "disabled": True})
                continue
            ms = (time.perf_counter() - t0) * 1000
            p.total_ms += ms
            p.calls += 1
            if ms > p.budget_ms * self.scale:
                p.slow_frames += 1
                if p.slow_frames >= 30:
                    p.enabled = False
                    self.ctx.bus.publish("error", {"processor": p.name, "disabled": True,
                                                   "error": f"too slow: {ms:.1f} ms > {p.budget_ms} ms budget"})
            else:
                p.slow_frames = max(0, p.slow_frames - 1)
        return frame


@register
class Thinking(LiveProcessor):
    """Funny little VidAI icon that bounces and wobbles in a corner while Claude works on a request.
    params: position (top-right|top-left|bottom-right|bottom-left), size (fraction of frame width), label."""
    type_name = "thinking"
    defaults = {"position": "top-right", "size": 0.085, "label": "", "icon": ""}
    N = 36  # animation frames per cycle (1.2 s at 30 fps)

    def __init__(self, *a, **k) -> None:
        super().__init__(*a, **k)
        self._sprites: dict[int, list[np.ndarray]] = {}
        self.shown_at: float | None = None

    def _build(self, W: int) -> list[np.ndarray]:
        from pathlib import Path

        from PIL import Image as PILImage
        from PIL import ImageDraw

        path = self.params["icon"] or str(Path(__file__).resolve().parents[1] / "assets" / "icon.png")
        icon = PILImage.open(path).convert("RGBA")
        iw = max(24, int(self.params["size"] * W))
        ih = int(icon.height * iw / icon.width)
        icon = icon.resize((iw, ih), PILImage.LANCZOS)
        pad = int(iw * 0.35)
        cw, ch = iw + 2 * pad, ih + 2 * pad + int(iw * 0.22)
        frames = []
        for i in range(self.N):
            ph = 2 * np.pi * i / self.N
            canvas = PILImage.new("RGBA", (cw, ch), (0, 0, 0, 0))
            d = ImageDraw.Draw(canvas)
            cx, cy = cw // 2, pad + ih // 2
            # wobble + bounce + squash
            ang = 9 * np.sin(ph)
            sq = 1 + 0.06 * np.sin(2 * ph)
            spr = icon.resize((int(iw * sq), int(ih / sq)), PILImage.BILINEAR).rotate(ang, resample=PILImage.BICUBIC,
                                                                                    expand=True)
            jump = abs(np.sin(ph))
            bounce = int(-jump * iw * 0.12)
            # cartoon ground shadow: smaller and lighter while the icon is up in the air
            sw_ = int(iw * (0.8 - 0.25 * jump)) // 2
            sy = cy + ih // 2 + int(pad * 0.25)
            d.ellipse([cx - sw_, sy - max(2, iw // 22), cx + sw_, sy + max(2, iw // 22)],
                      fill=(0, 0, 0, int(90 - 45 * jump)))
            canvas.alpha_composite(spr, (cx - spr.width // 2, cy - spr.height // 2 + bounce))
            # three thinking dots
            r = max(2, iw // 18)
            by = ch - int(iw * 0.13)
            for k in range(3):
                on = 0.5 + 0.5 * np.sin(ph * 2 - k * 1.2)
                x = cx + (k - 1) * r * 4
                a = int(90 + 165 * on)
                yy = by - int(on * r * 1.5)
                d.ellipse([x - r, yy - r, x + r, yy + r], fill=(255, 255, 255, a), outline=(60, 80, 170, a))
            frames.append(np.asarray(canvas))
        return frames

    def process(self, frame, t, ctx):
        H, W = frame.shape[:2]
        if W not in self._sprites:
            self._sprites[W] = self._build(W)
        if self.shown_at is None:
            self.shown_at = t
        spr = self._sprites[W][int((t - self.shown_at) * 30) % self.N]
        sh, sw = spr.shape[:2]
        m = int(0.02 * W)
        pos = self.params["position"]
        x = W - sw - m if "right" in pos else m
        y = H - sh - m if "bottom" in pos else m
        native.alpha_blend(frame, np.ascontiguousarray(spr), x, y, min(1.0, (t - self.shown_at) / 0.25))
        return frame

    def configure(self, params):
        super().configure(params)
        self._sprites.clear()
