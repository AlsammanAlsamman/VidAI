"""Feed models: small models trained locally on the live video feed, while you record.

The protocol (see docs/feed-models.md):

    teacher   slow but good labels for a few frames per second (a pretrained model, or plain statistics),
              run in a worker thread so the frame loop never waits for it
    student   a tiny model (NumPy, ~1 ms) trained on *your* frames against the teacher; runs every frame
    score     the student is checked against the teacher on frames it has not trained on yet; it is used
              only once `ready` (score >= target), like the lab's train-until-suitable loop
    memory    the trained student is saved to the lab (~/.vidai/models/feed_<key>) when the recording stops,
              so the next session starts trained and keeps fine-tuning

Feed models are processors (added, removed, closed like any effect; names start with "_feed_"). Effects ask
for them with `feeds = {"lighting", "hair"}` and draw through `ctx.composite(...)`, which applies whatever
is ready: the lighting model matches the overlay to the room, the hair mask lets hair cover its base.
New feed models: subclass FeedModel, set `key`, implement observe/teach/learn/infer, add it to FEEDS.
"""
from __future__ import annotations

import threading
import time
from typing import Any

import cv2
import numpy as np

from ..lab import LabModel
from .processors import LiveProcessor, register

FEEDS: dict[str, type["FeedModel"]] = {}


def feed(cls):
    FEEDS[cls.key] = cls
    return register(cls)


class FeedModel(LiveProcessor):
    """Base of the protocol. Subclasses implement:
      observe(frame, t, ctx)  frame thread, must be cheap: hand samples to the worker (self.offer)
      teach(sample) -> labels worker thread, may be slow (None = no teacher: learn from the sample itself)
      learn(sample, labels) -> score | None   worker thread: update the student, return its score if checked
      infer(...)              frame thread, fast: what effects use (via ctx.composite)
      student_state() / load_student(state)   for memory (optional: persist = False)"""
    key = ""
    stage = 0.5  # sees the picture after transforms (background, zoom) and before overlays
    rate = 3.0  # teacher calls per second (slower under load)
    target = 0.75  # score needed before the student is used
    min_checks = 1  # ... over at least this many checked frames
    persist = True
    budget_ms = 4.0

    def __init__(self, *a, **k) -> None:
        super().__init__(*a, **k)
        self.score: float | None = None
        self.checks = 0
        self.ready = False
        self.samples = 0
        self.error: str | None = None
        self._pending: Any = None
        self._lock = threading.Lock()
        self._ctx = None
        self._last_report = 0.0
        self._loaded = self._load()
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    # ---- frame thread
    def process(self, frame, t, ctx):
        self._ctx = ctx
        ctx.feed[self.key] = self
        self.observe(frame, t, ctx)
        if time.monotonic() - self._last_report > 5:
            self._last_report = time.monotonic()
            ctx.bus.publish("feed_model", {"model": self.key, "ready": self.ready, "samples": self.samples,
                                           "score": None if self.score is None else round(self.score, 3)})
        return frame

    def offer(self, sample: Any) -> None:
        with self._lock:
            self._pending = sample  # only the newest sample matters
        self._offered = time.monotonic()

    def want(self) -> bool:
        """Is it time for a new sample? (observe() should skip the crop/copy otherwise: it runs every frame)"""
        q = getattr(self._ctx, "quality", 0)
        return self._pending is None and time.monotonic() - getattr(self, "_offered", 0.0) >= 0.9 * (1 + q) / self.rate

    # ---- worker thread
    def _loop(self) -> None:
        last = 0.0
        while not self.closed:
            quality = getattr(self._ctx, "quality", 0)
            wait = (1 + quality) / self.rate - (time.monotonic() - last)
            if wait > 0:
                time.sleep(min(wait, 0.2))
                continue
            with self._lock:
                s, self._pending = self._pending, None
            if s is None:
                time.sleep(0.02)
                continue
            last = time.monotonic()
            try:
                labels = self.teach(s)
                score = self.learn(s, labels)
                self.samples += 1
                if score is not None:
                    self.score = score if self.score is None else 0.7 * self.score + 0.3 * score
                    self.checks += 1
                    self.ready = self.checks >= self.min_checks and self.score >= self.target
            except Exception as e:  # a failing teacher must never touch the recording
                self.error = repr(e)[:200]
                if self._ctx is not None:
                    self._ctx.bus.publish("error", {"processor": self.name, "where": "feed", "error": self.error})
                time.sleep(1.0)

    # ---- protocol
    def observe(self, frame: np.ndarray, t: float, ctx) -> None: ...
    def teach(self, sample: Any) -> Any:
        return None

    def learn(self, sample: Any, labels: Any) -> float | None:
        return None

    def student_state(self) -> dict[str, np.ndarray] | None:
        return None

    def load_student(self, state: dict[str, np.ndarray]) -> None: ...

    # ---- memory
    def _load(self) -> bool:
        if not self.persist:
            return False
        from .. import lab

        try:
            model, spec = lab.load_model(f"feed_{self.key}")
        except (KeyError, ValueError, OSError):
            return False
        self.load_student(model.state())
        self.score = float(spec.metrics.get("score", 0.0)) * 0.9  # re-checked on the first new frames
        self.checks = self.min_checks - 1  # one good check on this video and it is used
        return True

    def save(self) -> str | None:
        st = self.student_state()
        if not self.persist or st is None or not self.ready:
            return None
        from .. import lab

        spec = lab.ModelSpec(name=f"feed_{self.key}", class_path="vidai.live.feed:Weights", task="feed_model",
                             description=f"feed model '{self.key}' trained on the user's own video",
                             metrics={"score": float(self.score or 0.0), "samples": float(self.samples)})
        lab.save_model(Weights(state=st), spec)
        return spec.name

    def close(self) -> None:
        super().close()
        if self._ctx is not None and self._ctx.feed.get(self.key) is self:
            self._ctx.feed.pop(self.key, None)
        try:
            saved = self.save()
            if saved and self._ctx is not None:
                self._ctx.bus.publish("action", {"what": "feed_model_saved", "model": self.key, "as": saved,
                                                 "score": round(self.score or 0.0, 3)})
        except Exception as e:
            if self._ctx is not None:
                self._ctx.bus.publish("error", {"processor": self.name, "where": "feed_save", "error": repr(e)[:200]})

    def describe(self) -> dict:
        return {**super().describe(), "ready": self.ready, "samples": self.samples,
                "score": None if self.score is None else round(self.score, 3), "from_memory": self._loaded,
                **({"error": self.error} if self.error else {})}


class Weights(LabModel):
    """How feed students are stored in the lab registry (just their arrays)."""
    task = "feed_model"

    def __init__(self, state: dict[str, np.ndarray] | None = None, **hp) -> None:
        super().__init__(**hp)
        self._state = dict(state or {})

    def state(self) -> dict[str, np.ndarray]:
        return self._state

    def load_state(self, state: dict[str, np.ndarray]) -> None:
        self._state = dict(state)


# ------------------------------------------------------------------ lighting (no teacher: statistics)
@feed
class Lighting(FeedModel):
    """Learns the room from the feed: light level and colour per region, camera softness and sensor grain.
    Overlays drawn through ctx.composite get the same light, tint, softness and grain, so they look filmed
    instead of pasted. The scene changes all the time, so it is not stored (persist = False)."""
    key = "lighting"
    type_name = "feed_lighting"
    rate = 6.0
    target = 0.0
    persist = False
    GRID = (6, 8)  # rows, cols
    defaults = {"strength": 0.85}

    def __init__(self, *a, **k) -> None:
        self.cells: np.ndarray | None = None  # (rows, cols, 3) mean RGB
        self.softness = 0.0  # gaussian sigma, as a fraction of the frame width
        self.grain = 0.0  # noise std (0-255)
        super().__init__(*a, **k)

    def observe(self, frame, t, ctx):
        if self.want():
            self.offer(cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA))

    def learn(self, small, labels):
        rows, cols = self.GRID
        cells = cv2.resize(small.astype(np.float32), (cols, rows), interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
        smooth = cv2.GaussianBlur(g, (0, 0), 1.2)
        resid = g - smooth
        lap = cv2.Laplacian(g, cv2.CV_32F)
        flat = np.abs(lap) < np.percentile(np.abs(lap), 40)  # grain is measured where there are no edges
        grain = float(resid[flat].std()) if flat.any() else 0.0
        sharp = float(np.percentile(np.abs(lap), 95))  # strong edges: a soft webcam gives low values
        softness = float(np.clip((40.0 - sharp) / 40.0, 0, 1)) * 0.0025
        k = 0.2 if self.cells is not None else 1.0
        self.cells = cells if self.cells is None else (1 - k) * self.cells + k * cells
        self.grain = (1 - k) * self.grain + k * grain
        self.softness = (1 - k) * self.softness + k * softness
        return 1.0

    def adapt(self, rgba: np.ndarray, x: int, y: int, W: int, H: int) -> np.ndarray:
        """The overlay as if it were filmed here: region light and tint, camera softness, sensor grain."""
        if self.cells is None:
            return rgba
        s = float(self.params["strength"])
        rows, cols = self.GRID
        h, w = rgba.shape[:2]
        cy = int(np.clip((y + h / 2) / H * rows, 0, rows - 1))
        cx = int(np.clip((x + w / 2) / W * cols, 0, cols - 1))
        local = self.cells[max(0, cy - 1):cy + 2, max(0, cx - 1):cx + 2].reshape(-1, 3).mean(0)
        lum = float(local @ np.array([0.299, 0.587, 0.114]))
        gain = np.clip(lum / 150.0, 0.45, 1.15)  # 150 = a well-lit reference
        tint = local / max(1.0, local.mean())
        tint = np.clip(tint / tint.max(), 0.6, 1.0) * 1.0
        mul = (1 - s) + s * gain * tint
        out = cv2.multiply(np.ascontiguousarray(rgba), (float(mul[0]), float(mul[1]), float(mul[2]), 1.0))
        sigma = self.softness * W * s
        if sigma > 0.3:
            out = cv2.GaussianBlur(out, (0, 0), sigma)
        amp = int(round(self.grain * s))
        if amp >= 1:
            out = cv2.add(out, self._noise(out.shape[:2], amp), dtype=cv2.CV_8U)
        return out

    def _noise(self, shape: tuple[int, int], amp: int) -> np.ndarray:
        """Sensor-like grain (RGB, alpha untouched), cut from a pre-made tile: fresh noise per frame is slow."""
        tile = getattr(self, "_tile", None)
        if tile is None or self._tile_amp != amp or tile.shape[0] < shape[0] or tile.shape[1] < shape[1]:
            h, w = max(2 * shape[0], 512), max(2 * shape[1], 512)
            n = np.random.default_rng(3).standard_normal((h, w, 3), dtype=np.float32) * amp
            tile = self._tile = np.dstack([np.round(n), np.zeros((h, w), np.float32)]).astype(np.int16)
            self._tile_amp = amp
        oy = np.random.randint(0, tile.shape[0] - shape[0] + 1)
        ox = np.random.randint(0, tile.shape[1] - shape[1] + 1)
        return np.ascontiguousarray(tile[oy:oy + shape[0], ox:ox + shape[1]])


# ------------------------------------------------------------------ hair (teacher: MediaPipe multiclass)
class PixelMLP:
    """Tiny per-pixel classifier (features -> 16 tanh -> 1), trained with momentum SGD. ~0.5 ms per crop."""

    def __init__(self, n_in: int, hidden: int = 16, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.W1 = (rng.normal(0, 1, (n_in, hidden)) / np.sqrt(n_in)).astype(np.float32)
        self.b1 = np.zeros(hidden, np.float32)
        self.W2 = (rng.normal(0, 1, (hidden, 1)) / np.sqrt(hidden)).astype(np.float32)
        self.b2 = np.zeros(1, np.float32)
        self.mu = np.zeros(n_in, np.float32)
        self.sd = np.ones(n_in, np.float32)
        self.n = 0
        self._v = [np.zeros_like(p) for p in self.params()]

    def params(self) -> list[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2]

    def predict(self, X: np.ndarray) -> np.ndarray:
        h = np.tanh(((X - self.mu) / self.sd) @ self.W1 + self.b1)
        return 1 / (1 + np.exp(-(h @ self.W2 + self.b2)[:, 0]))

    def fit(self, X: np.ndarray, y: np.ndarray, steps: int = 40, lr: float = 0.08, batch: int = 256) -> None:
        # running feature normalisation (the user's colours, not ImageNet's)
        m, s = X.mean(0), X.std(0) + 1e-3
        k = 1.0 if self.n == 0 else 0.1
        self.mu, self.sd = (1 - k) * self.mu + k * m, (1 - k) * self.sd + k * s
        self.n += 1
        rng = np.random.default_rng(self.n)
        Z = (X - self.mu) / self.sd
        for _ in range(steps):
            i = rng.integers(0, len(Z), min(batch, len(Z)))
            z, t = Z[i], y[i]
            h = np.tanh(z @ self.W1 + self.b1)
            p = 1 / (1 + np.exp(-(h @ self.W2 + self.b2)[:, 0]))
            d = ((p - t) / len(t))[:, None].astype(np.float32)  # cross-entropy gradient
            gW2, gb2 = h.T @ d, d.sum(0)
            dh = (d @ self.W2.T) * (1 - h ** 2)
            gW1, gb1 = z.T @ dh, dh.sum(0)
            for j, (p_, g) in enumerate(zip(self.params(), (gW1, gb1, gW2, gb2))):
                self._v[j] = 0.9 * self._v[j] - lr * g
                p_ += self._v[j]

    def state(self) -> dict[str, np.ndarray]:
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2, "mu": self.mu, "sd": self.sd}

    def load(self, st: dict[str, np.ndarray]) -> None:
        if st["W1"].shape == self.W1.shape:
            self.W1, self.b1, self.W2, self.b2 = (st[k].astype(np.float32) for k in ("W1", "b1", "W2", "b2"))
            self.mu, self.sd = st["mu"].astype(np.float32), st["sd"].astype(np.float32)
            self.n = 1
            self._v = [np.zeros_like(p) for p in self.params()]


def pixel_features(crop: np.ndarray, face_box_px: tuple[float, float, float, float]) -> np.ndarray:
    """Per-pixel features of an RGB crop: colour (Lab), local colour, edge strength, position vs the face."""
    h, w = crop.shape[:2]
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).astype(np.float32)
    loc = cv2.blur(lab, (5, 5))
    L = lab[..., 0]
    grad = np.sqrt(cv2.Sobel(L, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(L, cv2.CV_32F, 0, 1) ** 2)
    fx, fy, fw, fh = face_box_px
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = (xx - (fx + fw / 2)) / max(fw, 1)
    v = (yy - fy) / max(fh, 1)
    return np.dstack([lab, loc, grad, u, v, u * u, v * v, u * v]).reshape(-1, 12).astype(np.float32)


@feed
class Hair(FeedModel):
    """Where your hair is, around your head. Teacher: MediaPipe's multiclass selfie segmenter (~90 ms, a few
    times per second); student: a PixelMLP on your own colours (~1 ms, every frame it is needed).
    Effects on the head use it so your hair covers their base (ctx.composite(..., occlude={"hair"}))."""
    key = "hair"
    type_name = "feed_hair"
    tracking = True
    rate = 2.5
    target = 0.7
    min_checks = 4
    CROP_W = 96
    defaults = {"threshold": 0.5}

    def __init__(self, *a, **k) -> None:
        self.student = PixelMLP(12)
        self._seg = None
        self._mask_cache: tuple[float, Any] | None = None
        super().__init__(*a, **k)

    # the region around the head that matters (expanded face box), in frame pixels
    @staticmethod
    def region(face, W: int, H: int) -> tuple[int, int, int, int]:
        x, y, w, h = face.box
        x0, x1 = int((x - 0.7 * w) * W), int((x + 1.7 * w) * W)
        y0, y1 = int((y - 1.1 * h) * H), int((y + 0.7 * h) * H)
        return max(0, x0), max(0, y0), min(W, x1), min(H, y1)

    def _crop(self, frame, face):
        H, W = frame.shape[:2]
        x0, y0, x1, y1 = self.region(face, W, H)
        if x1 - x0 < 16 or y1 - y0 < 16:
            return None
        s = self.CROP_W / (x1 - x0)
        ch = max(8, int((y1 - y0) * s))
        crop = cv2.resize(frame[y0:y1, x0:x1], (self.CROP_W, ch), interpolation=cv2.INTER_AREA)
        fx, fy, fw, fh = face.box
        box = ((fx * W - x0) * s, (fy * H - y0) * s, fw * W * s, fh * H * s)
        return crop, box, (x0, y0, x1, y1)

    def observe(self, frame, t, ctx):
        if ctx.tracks is None or not self.want():
            return
        face = ctx.tracks.face_now()
        if face is None:
            return
        c = self._crop(frame, face)
        if c is not None:
            self.offer((c[0].copy(), c[1]))

    def teach(self, sample):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        from ..models_dl import ensure

        if self._seg is None:
            self._seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
                base_options=BaseOptions(model_asset_path=str(ensure("selfie_multiclass_256x256.tflite"))),
                running_mode=vision.RunningMode.IMAGE, output_confidence_masks=True))
        crop, _ = sample
        r = self._seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop)))
        return np.squeeze(r.confidence_masks[1].numpy_view()).astype(np.float32)  # 1 = hair

    def learn(self, sample, labels):
        crop, box = sample
        X = pixel_features(crop, box)
        y = (labels.reshape(-1) > 0.5).astype(np.float32)
        score = None
        if self.student.n > 0:  # checked on this frame before training on it
            p = self.student.predict(X) > 0.5
            inter, union = float((p & (y > 0)).sum()), float((p | (y > 0)).sum())
            score = inter / union if union > 20 else None
        pos, neg = np.flatnonzero(y > 0), np.flatnonzero(y == 0)
        if len(pos) < 10:
            return score
        rng = np.random.default_rng(self.samples)
        idx = np.concatenate([rng.choice(pos, min(len(pos), 1200)), rng.choice(neg, min(len(neg), 1200))])
        self.student.fit(X[idx], y[idx])
        return score

    def infer(self, frame, ctx) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
        """Hair probability over the head region (frame pixels) for this frame, or None."""
        if not self.ready or ctx.tracks is None:
            return None
        face = ctx.tracks.face_now()
        if face is None:
            return None
        key = (id(frame), ctx.tracks.face_at if hasattr(ctx.tracks, "face_at") else 0)
        if self._mask_cache and self._mask_cache[0] == key:
            return self._mask_cache[1]
        c = self._crop(frame, face)
        if c is None:
            return None
        crop, box, (x0, y0, x1, y1) = c
        p = self.student.predict(pixel_features(crop, box)).reshape(crop.shape[:2])
        p = cv2.GaussianBlur(p, (0, 0), 1.0)
        out = (cv2.resize(p, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR), (x0, y0, x1, y1))
        self._mask_cache = (key, out)
        return out

    def student_state(self):
        return self.student.state() if self.student.n else None

    def load_student(self, state):
        self.student.load(state)

    def close(self) -> None:
        super().close()
        if self._seg is not None:
            try:
                self._seg.close()
            except Exception:
                pass
            self._seg = None


# ------------------------------------------------------------------ the compositor effects draw through
def composite(ctx, frame: np.ndarray, rgba: np.ndarray, x: int, y: int, alpha: float = 1.0,
              occlude: set[str] | frozenset = frozenset(), strength: float = 0.8) -> np.ndarray:
    """Draw an RGBA overlay into the scene using every feed model that is ready (lighting always, masks such
    as hair for `occlude`). Falls back to a plain blend when none is ready."""
    from .. import native

    H, W = frame.shape[:2]
    feeds = getattr(ctx, "feed", {}) or {}
    light = feeds.get("lighting")
    copied = False
    if light is not None and not light.closed:
        rgba, copied = light.adapt(rgba, x, y, W, H), True
    for key in occlude:
        m = feeds.get(key)
        got = m.infer(frame, ctx) if m is not None and not m.closed and hasattr(m, "infer") else None
        if got is None:
            continue
        mask, (x0, y0, x1, y1) = got
        h, w = rgba.shape[:2]
        # overlap of the overlay and the mask region, in overlay coordinates
        ox0, oy0, ox1, oy1 = max(x, x0), max(y, y0), min(x + w, x1), min(y + h, y1)
        if ox1 <= ox0 or oy1 <= oy0:
            continue
        if not copied:  # never change a cached sprite
            rgba, copied = rgba.copy(), True
        sub = mask[oy0 - y0:oy1 - y0, ox0 - x0:ox1 - x0]
        a = rgba[oy0 - y:oy1 - y, ox0 - x:ox1 - x, 3].astype(np.float32)
        rgba[oy0 - y:oy1 - y, ox0 - x:ox1 - x, 3] = (a * (1 - strength * np.clip(sub, 0, 1))).astype(np.uint8)
    native.alpha_blend(frame, np.ascontiguousarray(rgba), int(x), int(y), alpha)
    return frame
