"""Adapters for hub models (vidai.hub): each knows its model's input/output and runs it in a worker thread,
so the frame loop only draws the latest result (the video stays at 30 fps)."""
from __future__ import annotations

import threading
import time

import numpy as np

from .. import native
from .processors import LiveProcessor, register

EMOTIONS = ["neutral", "happy", "surprise", "sad", "angry", "disgust", "fear", "contempt"]
EMOJI = {"neutral": "😐", "happy": "😄", "surprise": "😮", "sad": "😢", "angry": "😠", "disgust": "🤢",
         "fear": "😨", "contempt": "😒"}
GESTURE_EMOJI = {"Thumb_Up": "👍", "Thumb_Down": "👎", "Victory": "✌", "Open_Palm": "✋", "Pointing_Up": "☝",
                 "Closed_Fist": "✊", "ILoveYou": "🤟"}


def _session(path: str, threads: int = 2):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
    so.log_severity_level = 3  # quiet
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


class _Worker(LiveProcessor):
    """Base: the frame loop hands over the newest frame; `work(frame)` runs in a thread at up to `rate` Hz."""
    rate = 5.0

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._frame = None
        self._lock = threading.Lock()
        self.error = None
        self._started = False

    def _start(self):
        if not self._started and not self.closed:
            self._started = True
            threading.Thread(target=self._loop, daemon=True).start()

    def offer(self, frame, ctx):
        self._ctx = ctx
        with self._lock:
            if self._frame is None:
                self._frame = frame.copy()
        self._start()

    def _loop(self):
        if not getattr(self, "_setup_done", False):
            try:
                self.setup()
                self._setup_done = True
            except Exception as e:
                self.error = repr(e)[:200]
                if getattr(self, "_ctx", None):
                    self._ctx.bus.publish("error", {"processor": self.name, "error": self.error})
                return
        try:
            self._work_loop()
        finally:
            self._started = False  # disabled or removed: offer() starts it again when re-enabled

    def _work_loop(self):
        last = 0.0
        while not self.closed and (self.enabled or self._frame is not None):
            wait = 1.0 / self.params.get("rate", self.rate) - (time.monotonic() - last)
            if wait > 0:
                time.sleep(wait)
            with self._lock:
                f, self._frame = self._frame, None
            if f is None:
                time.sleep(0.01)
                continue
            last = time.monotonic()
            try:
                self.work(f)
            except Exception as e:
                self.error = repr(e)[:200]

    def setup(self): ...
    def work(self, frame): ...


@register
class Emotion(_Worker):
    """Your facial expression as a live stat ('emotion' events) + an emoji above your head (show=True)."""
    type_name = "emotion"
    tracking = True
    defaults = {"show": True, "rate": 5.0, "min_confidence": 0.35, "neutral_weight": 0.5, "margin": -0.05}

    def setup(self):
        from .. import hub

        self.sess = _session(str(hub.path("emotion")))
        self.inp = self.sess.get_inputs()[0].name
        self.label, self.prob = None, 0.0
        self._smooth = np.zeros(len(EMOTIONS), np.float32)

    def work(self, frame):
        import cv2

        f = self._ctx.tracks.face_now() if self._ctx.tracks else None
        if f is None:
            return
        H, W = frame.shape[:2]
        x, y, w, h = f.box
        m = float(self.params["margin"])  # a tight crop (like the model's training faces) reads expressions best
        x0, y0 = max(0, int((x - m * w) * W)), max(0, int((y - m * h) * H))
        x1, y1 = min(W, int((x + (1 + m) * w) * W)), min(H, int((y + (1 + m) * h) * H))
        if x1 - x0 < 16 or y1 - y0 < 16:
            return
        g = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)[None, None]
        logits = self.sess.run(None, {self.inp: g})[0][0]
        p = np.exp(logits - logits.max())
        p[0] *= float(self.params["neutral_weight"])  # webcams make FER+ say "neutral" too often
        p /= p.sum()
        self._smooth = 0.5 * self._smooth + 0.5 * p
        i = int(np.argmax(self._smooth))
        label, prob = EMOTIONS[i], float(self._smooth[i])
        if label != self.label and prob >= self.params["min_confidence"]:
            self.label, self.prob = label, prob
            self._ctx.bus.publish("emotion", {"label": label, "confidence": round(prob, 2)})

    def process(self, frame, t, ctx):
        self.offer(frame, ctx)
        if self.params["show"] and getattr(self, "label", None) and ctx.tracks is not None:
            f = ctx.tracks.face_now()
            if f is not None:
                from .stickers import scaled

                H, W = frame.shape[:2]
                spr = scaled(EMOJI[self.label], int(f.width * W * 0.45))
                sh, sw = spr.shape[:2]
                native.alpha_blend(frame, spr, int(f.top[0] * W - sw / 2), int(f.top[1] * H - sh * 1.15))
        return frame


@register
class Gestures(_Worker):
    """Hand gestures as live events ('gesture': 👍 👎 ✌ ✋ ☝ ✊ 🤟) — use them in rules; show=True draws the emoji."""
    type_name = "gestures"
    defaults = {"show": True, "rate": 10.0, "min_score": 0.6, "show_seconds": 1.2}

    def setup(self):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        from .. import hub

        self.mp = mp
        self.rec = vision.GestureRecognizer.create_from_options(vision.GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=str(hub.path("gestures"))),
            running_mode=vision.RunningMode.VIDEO, num_hands=2))
        self.ts, self.current, self.pending, self.shown_until = 0, None, (None, 0), 0.0
        self.where = (0.5, 0.5)

    def work(self, frame):
        import cv2

        small = cv2.resize(frame[::2, ::2], (480, int(480 * frame.shape[0] / frame.shape[1])))
        self.ts += 100
        r = self.rec.recognize_for_video(self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                                                       data=np.ascontiguousarray(small)), self.ts)
        name, score = None, 0.0
        for gs, lms in zip(r.gestures, r.hand_landmarks):
            if gs and gs[0].category_name != "None" and gs[0].score > score:
                name, score = gs[0].category_name, gs[0].score
                self.where = (float(np.mean([p.x for p in lms])), float(min(p.y for p in lms)))
        if score < self.params["min_score"]:
            name = None
        cand, n = self.pending
        self.pending = (name, n + 1) if name == cand else (name, 1)
        if self.pending[1] >= 2 and name != self.current:  # stable for 2 checks
            self.current = name
            if name:
                self._ctx.bus.publish("gesture", {"name": name, "emoji": GESTURE_EMOJI.get(name, ""),
                                                  "score": round(score, 2)})
                self.shown_until = time.monotonic() + self.params["show_seconds"]

    def process(self, frame, t, ctx):
        self.offer(frame, ctx)
        if self.params["show"] and getattr(self, "current", None) and time.monotonic() < self.shown_until:
            from .stickers import scaled

            H, W = frame.shape[:2]
            spr = scaled(GESTURE_EMOJI.get(self.current, "✨"), int(W * 0.08))
            sh, sw = spr.shape[:2]
            native.alpha_blend(frame, spr, int(self.where[0] * W - sw / 2), int(self.where[1] * H - sh * 1.2))
        return frame


@register
class Style(_Worker):
    """Painting / anime looks from a hub model. The model repaints the picture a few times a second
    (as fast as it can); every frame shows the latest painting blended with the live picture."""
    type_name = "style"
    stage = 0
    defaults = {"model": "style_mosaic", "strength": 0.85, "rate": 6.0, "threads": 3}

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.out = None  # latest painting (made by the worker)

    def setup(self):
        from .. import hub

        m = hub.CATALOG[self.params["model"]]
        self.layout, self.size = m["params"]["layout"], int(m["params"]["size"])
        self.sess = _session(str(hub.path(self.params["model"])), int(self.params["threads"]))
        self.inp = self.sess.get_inputs()[0].name
        self.out = None

    def work(self, frame):
        import cv2

        H, W = frame.shape[:2]
        s = self.size
        if self.layout == "nhwc_tanh":  # AnimeGAN: keep the aspect ratio (dimensions multiple of 8)
            w = s * 16 // 9 // 8 * 8
            x = cv2.resize(frame, (w, s), interpolation=cv2.INTER_AREA).astype(np.float32) / 127.5 - 1.0
            y = self.sess.run(None, {self.inp: x[None]})[0][0]
            img = np.clip((y + 1.0) * 127.5, 0, 255).astype(np.uint8)
        else:  # fast neural style: NCHW, 0-255
            x = cv2.resize(frame, (s, s), interpolation=cv2.INTER_AREA).astype(np.float32).transpose(2, 0, 1)[None]
            y = self.sess.run(None, {self.inp: x})[0][0]
            img = np.clip(y.transpose(1, 2, 0), 0, 255).astype(np.uint8)
        self.out = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)

    def process(self, frame, t, ctx):
        import cv2

        self.offer(frame, ctx)
        out = getattr(self, "out", None)
        if out is None or out.shape != frame.shape:
            return frame
        a = float(self.params["strength"])
        return cv2.addWeighted(out, a, frame, 1 - a, 0)


@register
class Grade(LiveProcessor):
    """Colours like a reference photo: per-channel histogram matching -> 3 look-up tables (trained in ms,
    refreshed every few seconds so it adapts to the light). params: image (path), strength (0..1)."""
    type_name = "grade"
    stage = 0
    defaults = {"image": "", "strength": 0.8, "refresh": 3.0}

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._ref = None
        self._lut = None
        self._at = -1e9

    @staticmethod
    def _cdf(ch: np.ndarray) -> np.ndarray:
        h = np.bincount(ch.ravel(), minlength=256).astype(np.float64)
        c = np.cumsum(h)
        return c / c[-1]

    def _train(self, frame):
        import cv2

        if self._ref is None:
            ref = cv2.cvtColor(cv2.imread(self.params["image"]), cv2.COLOR_BGR2RGB)
            ref = cv2.resize(ref, (320, int(320 * ref.shape[0] / ref.shape[1])))
            self._ref = [self._cdf(ref[..., c]) for c in range(3)]
        small = frame[::6, ::6]
        s = float(self.params["strength"])
        lut = np.empty((256, 1, 3), np.uint8)
        for c in range(3):
            src = self._cdf(small[..., c])
            m = np.searchsorted(self._ref[c], src).clip(0, 255)  # value with the same rank in the photo
            lut[:, 0, c] = np.clip(np.arange(256) * (1 - s) + m * s + 0.5, 0, 255).astype(np.uint8)
        self._lut = lut

    def process(self, frame, t, ctx):
        import cv2

        if not self.params["image"]:
            return frame
        if self._lut is None or t - self._at > self.params["refresh"]:
            self._train(frame)
            self._at = t
        return cv2.LUT(frame, self._lut)
