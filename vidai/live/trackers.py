"""Shared body tracking for live effects: hands (21 points each, left/right) and face (box, eyes, nose,
mouth). One worker thread runs MediaPipe on small frames (~15-25 Hz); every effect reads the latest,
smoothed result from `Tracks` — no effect runs its own model. Models download themselves (models_dl)."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

PALM = [0, 5, 9, 13, 17]


@dataclass
class Hand:
    side: str  # "Left" | "Right" (as MediaPipe reports it for the picture)
    palm: tuple[float, float]  # normalized 0..1
    size: float  # wrist -> middle-finger base, fraction of frame width
    tip: tuple[float, float]  # index finger tip
    points: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class Face:
    box: tuple[float, float, float, float]  # x, y, w, h (normalized)
    eyes: tuple[tuple[float, float], tuple[float, float]]
    nose: tuple[float, float]
    mouth: tuple[float, float]

    @property
    def top(self) -> tuple[float, float]:
        return self.box[0] + self.box[2] / 2, self.box[1]

    @property
    def width(self) -> float:
        return self.box[2]


class Tracks:
    """Latest tracking results. Fed with small RGB frames from the frame loop via `feed`."""

    def __init__(self, bus=None, width: int = 640, want: set[str] | None = None) -> None:
        self.bus, self.width = bus, width
        self.want = set(want or {"hands", "face"})
        self.hands: list[Hand] = []
        self.face: Face | None = None
        self.hands_at = self.face_at = -1.0
        self.error: str | None = None
        self.ready = False
        self.max_hz = 30.0  # lowered by the performance governor
        self._frame = None
        self._lock = threading.Lock()
        self._stop = False
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def feed(self, frame: np.ndarray) -> None:
        """Offer the newest full frame (only resized when the worker is free)."""
        with self._lock:
            if self._frame is not None:
                return
        import cv2

        H, W = frame.shape[:2]
        k = max(1, W // (self.width * 2))  # 1920 -> stride 1 view of every k-th pixel, then linear resize
        small = cv2.resize(frame[::k, ::k], (self.width, int(H * self.width / W)), interpolation=cv2.INTER_LINEAR)
        with self._lock:
            self._frame = small

    def hand(self, which: str = "any", max_age: float = 0.4) -> Hand | None:
        if time.monotonic() - self.hands_at > max_age or not self.hands:
            return None
        if which in ("any", "hand", ""):
            return max(self.hands, key=lambda h: h.size)
        w = which.lower()
        for h in self.hands:
            if h.side.lower() == w:
                return h
        if w == "other" and len(self.hands) > 1:
            return sorted(self.hands, key=lambda h: h.size)[0]
        return None

    def face_now(self, max_age: float = 0.5) -> Face | None:
        return self.face if self.face and time.monotonic() - self.face_at <= max_age else None

    def _run(self) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision

            from ..models_dl import ensure

            hl = fd = None
            if "hands" in self.want:
                hl = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(ensure("hand_landmarker.task"))),
                    running_mode=vision.RunningMode.VIDEO, num_hands=2,
                    min_hand_detection_confidence=0.5, min_tracking_confidence=0.5))
            if "face" in self.want:
                fd = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
                    base_options=BaseOptions(model_asset_path=str(ensure("blaze_face_short_range.tflite"))),
                    running_mode=vision.RunningMode.VIDEO, min_detection_confidence=0.5))
        except Exception as e:
            self.error = f"tracking unavailable: {e!r}"[:300]
            if self.bus:
                self.bus.publish("error", {"where": "tracking", "error": self.error})
            return
        self.ready = True
        if self.bus:
            self.bus.publish("action", {"what": "tracking_ready", "tracks": sorted(self.want)})
        try:
            self._loop(mp, hl, fd)
        finally:
            for m in (hl, fd):
                if m is not None:
                    try:
                        m.close()
                    except Exception:
                        pass

    def _loop(self, mp, hl, fd) -> None:
        ts = 0
        last = 0.0
        while not self._stop:
            wait = 1.0 / self.max_hz - (time.monotonic() - last)
            if wait > 0:
                time.sleep(wait)
            last = time.monotonic()
            with self._lock:
                img, self._frame = self._frame, None
            if img is None:
                time.sleep(0.004)
                continue
            ts += 33
            mimg = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
            h, w = img.shape[:2]
            if hl is not None:
                r = hl.detect_for_video(mimg, ts)
                hands = []
                for pts, hd in zip(r.hand_landmarks, r.handedness):
                    palm = (float(np.mean([pts[i].x for i in PALM])), float(np.mean([pts[i].y for i in PALM])))
                    size = float(np.hypot(pts[0].x - pts[9].x, (pts[0].y - pts[9].y) * h / w))
                    hands.append(Hand(hd[0].category_name, palm, size, (pts[8].x, pts[8].y),
                                      [(p.x, p.y) for p in pts]))
                if hands:
                    self.hands = [self._smooth_hand(n) for n in hands]
                    self.hands_at = time.monotonic()
            if fd is not None:
                r = fd.detect_for_video(mimg, ts)
                if r.detections:
                    d = max(r.detections, key=lambda d: d.bounding_box.width)
                    b, kp = d.bounding_box, d.keypoints
                    f = Face((b.origin_x / w, b.origin_y / h, b.width / w, b.height / h),
                             ((kp[0].x, kp[0].y), (kp[1].x, kp[1].y)), (kp[2].x, kp[2].y), (kp[3].x, kp[3].y))
                    self.face = f if self.face is None else _lerp_face(self.face, f, 0.5)
                    self.face_at = time.monotonic()

    def _smooth_hand(self, new: Hand) -> Hand:
        old = next((h for h in self.hands if h.side == new.side), None)
        if old is None:
            return new
        k = 0.5
        return Hand(new.side, _lerp(old.palm, new.palm, k), old.size + 0.3 * (new.size - old.size),
                    _lerp(old.tip, new.tip, k), new.points)

    def close(self) -> None:
        self._stop = True
        self._th.join(2)  # the worker closes the MediaPipe models itself (no noisy errors at exit)


def _lerp(a, b, k):
    return tuple(x + k * (y - x) for x, y in zip(a, b))


def _lerp_face(a: Face, b: Face, k: float) -> Face:
    return Face(_lerp(a.box, b.box, k), (_lerp(a.eyes[0], b.eyes[0], k), _lerp(a.eyes[1], b.eyes[1], k)),
                _lerp(a.nose, b.nose, k), _lerp(a.mouth, b.mouth, k))
