"""Built-in background replacement (RVM matting; MediaPipe fallback). Models download themselves."""
import threading
import time

import cv2
import numpy as np

from vidai import native
from vidai.live.processors import LiveProcessor, register


@register
class Background(LiveProcessor):
    """Replace the background behind the person.
    engine: rvm (Robust Video Matting: soft, stable edges, keeps hands and hair) | mediapipe (lighter, rougher).
    mode: color | blur | image | animated. The model runs in a worker thread (~15 Hz); the frame loop only
    upsizes the latest matte and blends it in C."""
    type_name = "background"
    stage = 0
    budget_ms = 14.0
    defaults = {"engine": "rvm", "mode": "color", "color": [40, 30, 90], "image": "", "drift": 40, "rate": 15,
                "width": 640, "ratio": 0.375, "threads": 2}

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.latest, self.alpha = None, None
        self._bg = {}
        self._t = 0.0
        self._lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    # ---------------- matting worker ----------------
    def _run(self):
        try:
            step = self._rvm() if self.params["engine"] == "rvm" else self._mediapipe()
        except Exception:
            step = self._mediapipe()  # fallback
        last = 0.0
        while True:
            rate = self.params["rate"] / (1 + getattr(self, "_quality", 0))  # slower under load (governor)
            wait = 1.0 / rate - (time.monotonic() - last)
            if wait > 0:
                time.sleep(wait)
            with self._lock:
                img, self.latest = self.latest, None
            if img is None:
                time.sleep(0.005)
                continue
            last = time.monotonic()
            a = step(img)
            self.alpha = np.clip(a * 255, 0, 255).astype(np.uint8)

    def _rvm(self):
        import onnxruntime as ort
        from vidai.models_dl import ensure

        so = ort.SessionOptions()
        so.intra_op_num_threads, so.inter_op_num_threads = int(self.params["threads"]), 1
        sess = ort.InferenceSession(str(ensure("rvm_mobilenetv3_fp32.onnx")), so, providers=["CPUExecutionProvider"])
        rec = [np.zeros((1, 1, 1, 1), np.float32)] * 4
        ratio = np.array([self.params["ratio"]], np.float32)

        def step(img):
            nonlocal rec
            x = img.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
            out = sess.run(None, {"src": x, "r1i": rec[0], "r2i": rec[1], "r3i": rec[2], "r4i": rec[3],
                                  "downsample_ratio": ratio})
            rec = out[2:]  # memory between frames = stable, flicker-free matte
            return out[1][0, 0]
        return step

    def _mediapipe(self):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        from vidai.models_dl import ensure

        seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=str(ensure("selfie_segmenter.tflite"))),
            running_mode=vision.RunningMode.VIDEO, output_confidence_masks=True))
        ts = [0]
        prev = [None]

        def step(img):
            ts[0] += 66
            small = cv2.resize(img, (256, 144))
            m = np.squeeze(seg.segment_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=small), ts[0])
                           .confidence_masks[0].numpy_view()).astype(np.float32)
            m = cv2.resize(m, (img.shape[1], img.shape[0]))
            prev[0] = m if prev[0] is None else 0.5 * prev[0] + 0.5 * m
            return np.clip((prev[0] - 0.3) * 2.5, 0, 1)
        return step

    # ---------------- backgrounds ----------------
    def _animated(self, W, H, t):
        key = ("anim", W, H)
        if key not in self._bg:
            rng = np.random.default_rng(7)
            cw = W * 2
            canvas = np.zeros((H // 4, cw // 4, 3), np.float32)
            base = np.array([22, 16, 48], np.float32)
            canvas[:] = base
            cols = [(139, 108, 255), (91, 140, 255), (255, 59, 92), (52, 211, 153), (255, 191, 0)]
            yy, xx = np.mgrid[0:H // 4, 0:cw // 4]
            for i in range(14):
                cx, cy, r = rng.uniform(0, cw // 4), rng.uniform(0, H // 4), rng.uniform(H / 16, H / 7)
                for dx in (-cw // 4, 0, cw // 4):
                    d = np.sqrt((xx - cx - dx) ** 2 + (yy - cy) ** 2) / r
                    canvas += np.exp(-d ** 2)[..., None] * (np.array(cols[i % 5], np.float32) - base) * 0.55
            self._bg[key] = np.ascontiguousarray(cv2.resize(np.clip(canvas, 0, 255).astype(np.uint8), (cw, H),
                                                            interpolation=cv2.INTER_CUBIC))
        off = int((t * self.params["drift"]) % W)
        return np.ascontiguousarray(self._bg[key][:, off:off + W])

    def _background(self, W, H, frame):
        mode = self.params["mode"]
        if mode == "animated":
            return self._animated(W, H, self._t)
        if mode == "blur":
            small = cv2.resize(frame[::8, ::8], (W // 16, H // 16), interpolation=cv2.INTER_AREA)
            return cv2.resize(cv2.GaussianBlur(small, (0, 0), 3), (W, H), interpolation=cv2.INTER_LINEAR)
        key = (W, H, mode, str(self.params["color"]), self.params["image"])
        if key not in self._bg:
            if mode == "image" and self.params["image"]:
                img = cv2.cvtColor(cv2.imread(self.params["image"]), cv2.COLOR_BGR2RGB)
                ih, iw = img.shape[:2]  # cover the frame without stretching
                s = max(W / iw, H / ih)
                img = cv2.resize(img, (int(iw * s + 0.5), int(ih * s + 0.5)), interpolation=cv2.INTER_AREA)
                y0, x0 = (img.shape[0] - H) // 2, (img.shape[1] - W) // 2
                bg = np.ascontiguousarray(img[y0:y0 + H, x0:x0 + W])
            else:
                c = np.array(self.params["color"], np.float32)
                ramp = np.linspace(1.35, 0.65, H, dtype=np.float32)[:, None, None]
                bg = np.ascontiguousarray(np.broadcast_to(np.clip(c * ramp, 0, 255).astype(np.uint8), (H, W, 3)))
            self._bg = {k: v for k, v in self._bg.items() if k[0] == "anim"}
            self._bg[key] = bg
        return self._bg[key]

    def process(self, frame, t, ctx):
        self._t = t
        self._quality = getattr(ctx, "quality", 0)
        H, W = frame.shape[:2]
        with self._lock:
            if self.latest is None:
                tw = int(self.params["width"])
                k = max(1, W // (tw * 2))
                self.latest = cv2.resize(frame[::k, ::k], (tw, int(H * tw / W)), interpolation=cv2.INTER_AREA)
        if self.alpha is None:
            return frame
        alpha = cv2.resize(self.alpha, (W, H), interpolation=cv2.INTER_LINEAR)
        return native.mask_blend(frame, self._background(W, H, frame), alpha)
