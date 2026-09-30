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

import math
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
    tracking: bool = False  # True = needs hands/face tracking (ctx.tracks)
    defaults: dict[str, Any] = {}
    listens: set[str] = set()
    budget_ms: float = 8.0
    stage: float = 1  # 0 = transforms the picture (zoom, blur, models) — runs first; 1 = overlays on top
    feeds: frozenset = frozenset()  # feed models this effect draws with (vidai.live.feed), added automatically

    def __init__(self, name: str, params: dict[str, Any] | None = None, enabled: bool = True,
                 until: float | None = None) -> None:
        self.name = name
        self.params = {**self.defaults, **(params or {})}
        self.enabled = enabled
        self.until = until  # auto-disable at this recording time (s)
        self.slow_frames = 0
        self.total_ms = 0.0
        self.calls = 0
        self.closed = False

    def close(self) -> None:
        """Removed from the chain: stop worker threads and free models (subclasses extend this)."""
        self.closed = True

    def configure(self, params: dict[str, Any]) -> None:
        self.params.update(params)

    def wants_feeds(self) -> set[str]:
        return set(self.feeds)

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
        self.tracks = None  # vidai.live.trackers.Tracks, started when an effect needs it
        self.quality = 0  # performance level set by the governor: 0 normal, 1 light, 2 minimal
        self.feed: dict[str, Any] = {}  # feed models trained on this video (vidai.live.feed), by key

    def composite(self, frame: np.ndarray, rgba: np.ndarray, x: int, y: int, alpha: float = 1.0,
                  occlude=frozenset(), strength: float = 0.8) -> np.ndarray:
        """Draw an overlay *into the scene*: matched to the room's light, and behind hair etc. (`occlude`)
        when those feed models are ready. For things attached to people/objects, not for screen graphics."""
        from .feed import composite

        return composite(self, frame, rgba, x, y, alpha, occlude, strength)

    def need_tracking(self) -> None:
        if self.tracks is None:
            from .trackers import Tracks

            self.tracks = Tracks(self.bus)

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
        if p.tracking:
            self.ctx.need_tracking()
        self.ensure_feeds(p)
        self.remove(p.name)
        # copy-on-write: the frame loop iterates self.items without a lock, so never mutate it in place
        self.items = sorted(self.items + [p], key=lambda q: q.stage)  # stable: transforms first, then overlays
        if p.listens:
            p._sub = self.ctx.bus.subscribe(lambda ev, p=p: p.on_event(ev, self.ctx), p.listens)
        return p

    def remove(self, name: str) -> bool:
        gone = [p for p in self.items if p.name == name]
        self.items = [p for p in self.items if p.name != name]
        for p in gone:
            self._close(p)
        if gone and not name.startswith("_feed_"):
            self.prune_feeds()
        return bool(gone)

    def ensure_feeds(self, p: LiveProcessor) -> None:
        """Start the feed models an effect draws with (e.g. {"lighting", "hair"}), shared by all effects."""
        from .feed import FEEDS

        for key in p.wants_feeds():
            if key in FEEDS and self.get(f"_feed_{key}") is None:
                self.add(FEEDS[key](f"_feed_{key}"))

    def prune_feeds(self) -> None:
        """Stop automatic feed models no remaining effect uses (they are saved to the lab when closed)."""
        wanted = set().union(*(p.wants_feeds() for p in self.items if not p.name.startswith("_feed_")))
        for p in [q for q in self.items if q.name.startswith("_feed_") and q.name[6:] not in wanted]:
            self.remove(p.name)

    def _close(self, p: LiveProcessor) -> None:
        if getattr(p, "_sub", None) is not None:
            self.ctx.bus.unsubscribe(p._sub)
            p._sub = None
        try:
            p.close()
        except Exception as e:
            self.ctx.bus.publish("error", {"processor": p.name, "where": "close", "error": repr(e)[:200]})

    def close_all(self) -> None:
        """The pipeline stopped: stop every processor's workers (the list stays for status/summary)."""
        for p in self.items:
            self._close(p)

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


TARGETS = ("hand", "right_hand", "left_hand", "other_hand", "finger", "head", "above_head", "face", "eyes",
           "nose", "mouth", "screen")


@register
class Attach(LiveProcessor):
    """Stick an emoji / word / image to a body part and follow it.
    params: what (emoji, word like "apple", or image path), to (hand | right_hand | left_hand | other_hand |
    finger | head | above_head | face | eyes | nose | mouth | screen), scale, dx, dy (offsets, fraction of
    the part's size), position (for to=screen), realistic (match the room's light, default on),
    behind_hair (hair covers it: for things worn on the head)."""
    type_name = "attach"
    tracking = True
    defaults = {"what": "🍎", "to": "hand", "scale": 1.0, "dx": 0.0, "dy": 0.0, "position": "bottom-left",
                "realistic": True, "behind_hair": False}

    def wants_feeds(self) -> set[str]:
        if self.params["to"] == "screen" or not self.params.get("realistic", True):
            return set()  # screen graphics stay crisp
        return {"lighting"} | ({"hair"} if self.params.get("behind_hair") else set())

    def __init__(self, *a, **k) -> None:
        super().__init__(*a, **k)
        self.alpha = 0.0

    def _place(self, W: int, H: int, tr) -> tuple[float, float, float] | None:
        """(center_x_px, center_y_px, width_px) or None if the part is not visible."""
        to, sc = self.params["to"], self.params["scale"]
        if to in ("hand", "right_hand", "left_hand", "other_hand", "finger"):
            side = {"hand": "any", "right_hand": "Right", "left_hand": "Left", "other_hand": "other",
                    "finger": "any"}[to]
            h = tr.hand(side)
            if h is None:
                return None
            w = h.size * W * 1.6 * sc
            if to == "finger":
                return h.tip[0] * W, h.tip[1] * H - w * 0.3, w * 0.6
            return h.palm[0] * W, h.palm[1] * H - w * 0.35, w
        f = tr.face_now()
        if f is None:
            return None
        fw = f.width * W
        if to == "head":
            return f.top[0] * W, f.top[1] * H - fw * 0.22 * sc, fw * 1.25 * sc
        if to == "above_head":
            return f.top[0] * W, f.top[1] * H - fw * 0.6 * sc, fw * 0.8 * sc
        if to == "face":
            return (f.box[0] + f.box[2] / 2) * W, (f.box[1] + f.box[3] / 2) * H, fw * 1.1 * sc
        if to == "eyes":
            (x1, y1), (x2, y2) = f.eyes
            dist = np.hypot((x2 - x1) * W, (y2 - y1) * H)
            return (x1 + x2) / 2 * W, (y1 + y2) / 2 * H, dist * 2.3 * sc
        if to == "nose":
            return f.nose[0] * W, f.nose[1] * H, fw * 0.35 * sc
        if to == "mouth":
            return f.mouth[0] * W, f.mouth[1] * H, fw * 0.45 * sc
        return None

    def process(self, frame, t, ctx):
        from .stickers import scaled

        H, W = frame.shape[:2]
        if self.params["to"] == "screen":
            spr = scaled(self.params["what"], W * 0.12 * self.params["scale"])
            from ..overlays import resolve_position

            x, y = resolve_position(self.params["position"], W, H, spr.shape[1], spr.shape[0])
            native.alpha_blend(frame, spr, x, y)
            return frame
        tr = ctx.tracks
        place = self._place(W, H, tr) if tr is not None else None
        self.alpha = min(1.0, self.alpha + 0.25) if place else max(0.0, self.alpha - 0.15)
        if place:
            self._last = place
        if self.alpha <= 0 or not getattr(self, "_last", None):
            return frame
        cx, cy, w = self._last
        spr = scaled(self.params["what"], w)
        sh, sw = spr.shape[:2]
        cx += self.params["dx"] * w
        cy += self.params["dy"] * w
        # keep it inside the picture (a big title above a head near the top edge must not vanish)
        cx = min(max(cx, sw / 2), W - sw / 2) if sw <= W else W / 2
        cy = min(max(cy, sh / 2), H - sh / 2) if sh <= H else H / 2
        if self.params.get("realistic", True):
            ctx.composite(frame, spr, int(cx - sw / 2), int(cy - sh / 2), self.alpha,
                          occlude={"hair"} if self.params.get("behind_hair") else frozenset())
        else:
            native.alpha_blend(frame, spr, int(cx - sw / 2), int(cy - sh / 2), self.alpha)
        return frame


@register
class BigEyes(LiveProcessor):
    """Cartoon pop-out eyes. params: zoom (magnification), size (eye patch size)."""
    type_name = "big_eyes"
    tracking = True
    stage = 0
    defaults = {"zoom": 1.8, "size": 1.0}

    def __init__(self, *a, **k) -> None:
        super().__init__(*a, **k)
        self._masks: dict[int, np.ndarray] = {}

    def _mask(self, r: int) -> np.ndarray:
        if r not in self._masks:
            yy, xx = np.mgrid[-r:r, -r:r]
            dist = np.sqrt(xx ** 2 + yy ** 2) / r
            self._masks[r] = np.clip((1.0 - dist) / 0.35, 0, 1)[..., None].astype(np.float32)
        return self._masks[r]

    def process(self, frame, t, ctx):
        import cv2

        f = ctx.tracks.face_now() if ctx.tracks else None
        if f is None:
            return frame
        H, W = frame.shape[:2]
        r = max(8, int(f.width * W * 0.17 * self.params["size"]))
        z = max(1.05, float(self.params["zoom"]))
        m = self._mask(r)
        for ex, ey in f.eyes:
            cx, cy = int(ex * W), int(ey * H)
            x0, y0, x1, y1 = cx - r, cy - r, cx + r, cy + r
            s = int(r / z)
            if x0 < 0 or y0 < 0 or x1 > W or y1 > H or s < 2:
                continue
            big = cv2.resize(frame[cy - s:cy + s, cx - s:cx + s], (2 * r, 2 * r)).astype(np.float32)
            roi = frame[y0:y1, x0:x1].astype(np.float32)
            frame[y0:y1, x0:x1] = (big * m + roi * (1 - m)).astype(np.uint8)
        return frame


_LUM = np.array([0.299, 0.587, 0.114], np.float32)


@register
class Adjust(LiveProcessor):
    """Picture adjustments, fast (one C colour pass + one gamma look-up table):
    brightness (-0.4..0.4, adds light), exposure (gain, 0.5..2), contrast (0.5..2), saturation (0 = black and white,
    1 = normal, 2 = vivid), warmth (-1 cool .. 1 warm), gamma (<1 lifts shadows), sharpen (0..1),
    auto (true = keep the picture well lit automatically, e.g. in a dark room)."""
    type_name = "adjust"
    stage = 0
    budget_ms = 12.0
    defaults = {"brightness": 0.0, "exposure": 1.0, "contrast": 1.0, "saturation": 1.0, "warmth": 0.0,
                "gamma": 1.0, "sharpen": 0.0, "auto": False, "target": 0.5}

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._auto_gamma = 1.0
        self._auto_gain = 1.0
        self._n = 0
        self._lut_key = None
        self._lut = None

    def matrix(self) -> np.ndarray:
        """4x3 matrix (rows r, g, b, 1 -> columns r, g, b) in 0..1 colour space."""
        p = self.params
        s, c = float(p["saturation"]), float(p["contrast"])
        gain = float(p["exposure"]) * (self._auto_gain if p["auto"] else 1.0)
        M = s * np.eye(3, dtype=np.float32) + (1 - s) * np.outer(_LUM, np.ones(3, np.float32))
        bias = np.zeros(3, np.float32)
        M, bias = M * c, bias * c + 0.5 * (1 - c)  # contrast around mid grey
        M, bias = M * gain, bias * gain
        w = float(p["warmth"])
        tint = np.array([1 + 0.12 * w, 1 + 0.02 * w, 1 - 0.12 * w], np.float32)
        M, bias = M * tint[None, :], bias * tint
        bias = bias + float(p["brightness"])
        return np.vstack([M, bias[None, :]]).astype(np.float32)

    def _measure(self, frame) -> None:
        """Auto light: look at the middle of the picture (where the person usually is) every 10 frames."""
        H, W = frame.shape[:2]
        mid = frame[H // 5:H * 4 // 5:8, W // 4:W * 3 // 4:8].astype(np.float32) / 255.0
        lum = float(np.mean(mid @ _LUM))
        lum = min(max(lum, 0.02), 0.98)
        target = float(self.params["target"])
        g = math.log(target) / math.log(lum)  # gamma that maps the current level to the target
        g = min(max(g, 0.45), 1.4)
        gain = min(max(target / max(lum ** g, 1e-3), 0.8), 1.3)
        self._auto_gamma += (g - self._auto_gamma) * 0.15  # smooth: no flicker
        self._auto_gain += (gain - self._auto_gain) * 0.1

    def tables(self) -> np.ndarray:
        """(3, 3, 256) tables: gamma on each input channel, then the colour matrix -> one C pass."""
        gamma = float(self.params["gamma"]) * (self._auto_gamma if self.params["auto"] else 1.0)
        v = (np.arange(256, dtype=np.float32) / 255.0) ** gamma  # gamma per input value
        M = self.matrix()  # rows: inputs r, g, b, 1 ; columns: outputs
        T = np.empty((3, 3, 256), np.float32)
        for c in range(3):
            for ch in range(3):
                T[c, ch] = v * M[ch, c] * 255.0
            T[c, 0] += M[3, c] * 255.0  # bias once per output channel
        return T

    def process(self, frame, t, ctx):
        import cv2

        p = self.params
        if p["auto"] and self._n % 10 == 0:
            self._measure(frame)
        self._n += 1
        key = (tuple(sorted((k, v) for k, v in p.items() if k != "sharpen")), round(self._auto_gamma, 3),
               round(self._auto_gain, 3))
        if key != self._lut_key:  # rebuild the tables only when something changed
            self._lut, self._lut_key = self.tables(), key
        frame = native.lut3x3(np.ascontiguousarray(frame), self._lut)  # one C pass, in place
        if float(p["sharpen"]) > 0.01:
            small = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2), interpolation=cv2.INTER_AREA)
            blur = cv2.resize(cv2.GaussianBlur(small, (0, 0), 0.9), (frame.shape[1], frame.shape[0]))
            frame = cv2.addWeighted(frame, 1 + float(p["sharpen"]), blur, -float(p["sharpen"]), 0)
        return frame


from . import background as _background  # noqa: E402,F401  (registers the built-in "background" effect)
from . import hub_effects as _hub_effects  # noqa: E402,F401  (emotion, gestures, style, grade)
from . import feed as _feed  # noqa: E402,F401  (feed models: lighting, hair — trained on the live video)
