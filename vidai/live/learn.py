"""Instant models: learned and corrected *during* the recording.

    "vidai learn slide"        -> new learner "slide" (labels yes/no)      (or Claude: {"cmd": "learn", ...})
    "vidai label yes"          -> the last second of frames are examples of "yes"
    "vidai wrong"              -> the current prediction was wrong: relabel the recent frames (binary)

A learner is a k-nearest-neighbour model on tiny frame features (optionally only a region of the frame),
so adding or correcting examples takes effect on the very next prediction — no training step. It predicts
at 2 Hz and publishes `learner` events when its answer changes. At the end it is saved to the lab registry
(`InstantKNN`), so it can label other videos (`classify_video`) or be improved offline.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from ..lab import LabModel

FEAT_W, FEAT_H = 24, 14


def features(small_rgb: np.ndarray, region: tuple[float, float, float, float] | None = None) -> np.ndarray:
    import cv2

    img = small_rgb
    if region:
        h, w = img.shape[:2]
        x, y, rw, rh = region
        img = img[int(y * h): max(int(y * h) + 2, int((y + rh) * h)), int(x * w): max(int(x * w) + 2, int((x + rw) * w))]
    f = cv2.resize(img, (FEAT_W, FEAT_H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    g = f.mean(axis=2)
    edges = np.abs(np.diff(g, axis=0)).ravel()
    v = np.concatenate([f.ravel(), edges * 2.0])
    return (v - v.mean()) / (v.std() + 1e-6)


class InstantKNN(LabModel):
    """k-NN frame classifier (cosine similarity on standardized tiny features)."""

    task = "frame_classifier"

    def __init__(self, k: int = 5, max_per_label: int = 400, labels: list[str] | None = None,
                 region: list[float] | None = None) -> None:
        super().__init__(k=k, max_per_label=max_per_label, labels=labels or ["yes", "no"], region=region)
        self.k, self.max_per_label = k, max_per_label
        self.labels = list(labels or ["yes", "no"])
        self.region = tuple(region) if region else None
        self.X: dict[str, list[np.ndarray]] = {l: [] for l in self.labels}

    def add(self, feats: list[np.ndarray], label: str) -> None:
        if label not in self.X:
            self.labels.append(label)
            self.X[label] = []
        self.X[label].extend(feats)
        del self.X[label][: max(0, len(self.X[label]) - self.max_per_label)]

    def remove_near(self, feats: list[np.ndarray], label: str, thr: float = 0.97) -> int:
        """Drop examples of `label` that look like `feats` (used by corrections)."""
        if not self.X.get(label) or not feats:
            return 0
        A = np.stack(self.X[label])
        B = np.stack(feats)
        sim = (A @ B.T) / A.shape[1]
        keep = sim.max(axis=1) < thr
        n = int((~keep).sum())
        self.X[label] = [a for a, k in zip(self.X[label], keep) if k]
        return n

    @property
    def ready(self) -> bool:
        return any(self.X[l] for l in self.labels)

    @property
    def n_labels_with_examples(self) -> int:
        return sum(1 for l in self.labels if self.X[l])

    def counts(self) -> dict[str, int]:
        return {l: len(v) for l, v in self.X.items()}

    def predict_one(self, f: np.ndarray) -> tuple[str | None, float]:
        if not self.ready:
            return None, 0.0
        X = np.concatenate([np.stack(v) for v in self.X.values() if v])
        y = [l for l, v in self.X.items() for _ in v]
        sim = X @ f / f.size
        k = min(self.k, len(y))
        top = np.argpartition(-sim, k - 1)[:k]
        votes: dict[str, float] = {}
        for i in top:
            votes[y[i]] = votes.get(y[i], 0.0) + max(0.0, float(sim[i])) + 1e-3
        best = max(votes, key=votes.get)
        return best, votes[best] / sum(votes.values())

    # LabModel API (offline use)
    def fit(self, X, Y) -> None:
        for x, y in zip(X, Y):
            self.add([np.asarray(x, np.float32)], str(y))

    def predict(self, X):
        return np.array([self.predict_one(np.asarray(x, np.float32))[0] for x in X])

    def evaluate(self, X, Y) -> dict[str, float]:
        return {"accuracy": float(np.mean(self.predict(X) == np.asarray(Y).astype(str)))}

    def state(self) -> dict[str, np.ndarray]:
        out = {}
        for i, l in enumerate(self.labels):
            if self.X[l]:
                out[f"X{i}"] = np.stack(self.X[l])
        out["labels"] = np.array(self.labels)
        return out

    def load_state(self, state) -> None:
        self.labels = [str(l) for l in state["labels"]]
        self.X = {l: list(state[f"X{i}"]) if f"X{i}" in state else [] for i, l in enumerate(self.labels)}


class LiveLearner:
    def __init__(self, bus, name: str, labels: list[str] | None = None, region: list[float] | None = None,
                 k: int = 5, examples_window: float = 1.0, rate_hz: float = 2.0) -> None:
        self.bus, self.name = bus, name
        self.model = InstantKNN(k=k, labels=labels, region=region)
        self.recent: deque[tuple[float, np.ndarray]] = deque(maxlen=60)
        self.window = examples_window
        self.rate = rate_hz
        self.last_pred_t = -1e9
        self.current: str | None = None
        self.pending: tuple[str | None, int] = (None, 0)
        self.segments: list[tuple[float, float, str]] = []
        self._seg_start = 0.0
        self.started = False  # predictions begin once two different labels were taught

    def feed(self, small_rgb: np.ndarray, t: float) -> None:
        f = features(small_rgb, self.model.region)
        self.recent.append((t, f))
        self.started = self.started or self.model.n_labels_with_examples >= 2
        if t - self.last_pred_t < 1.0 / self.rate or not self.started or not self.model.ready:
            return
        self.last_pred_t = t
        label, conf = self.model.predict_one(f)
        cand, n = self.pending
        self.pending = (label, n + 1) if label == cand else (label, 1)
        if label != self.current and self.pending[1] >= 2:  # 2 agreeing predictions = stable change
            if self.current is not None:
                self.segments.append((self._seg_start, t, self.current))
            self.current, self._seg_start = label, t
            self.bus.publish("learner", {"name": self.name, "label": label, "confidence": round(conf, 2)}, t)

    def _recent(self, t: float | None = None) -> list[np.ndarray]:
        if not self.recent:
            return []
        t = self.recent[-1][0] if t is None else t
        return [f for ft, f in self.recent if t - self.window <= ft <= t]

    def label(self, value: str, correcting: bool = False) -> dict:
        feats = self._recent()
        if correcting:  # a correction also removes the conflicting examples of the wrong label
            for other in self.model.labels:
                if other != value:
                    self.model.remove_near(feats, other)
        self.model.add(feats, value)
        self.last_pred_t = -1e9
        return {"name": self.name, "label": value, "added": len(feats), "counts": self.model.counts()}

    def wrong(self) -> dict:
        """The current prediction is wrong. Binary: relabel recent frames with the other label."""
        if self.current is None:
            return {"name": self.name, "error": "no prediction yet"}
        others = [l for l in self.model.labels if l != self.current]
        if len(others) != 1:
            return {"name": self.name, "error": "more than 2 labels: say which label it is ('vidai label X')"}
        wrong_label, self.current = self.current, None
        out = self.label(others[0], correcting=True)
        out["corrected_from"] = wrong_label
        return out

    def finish(self, t: float) -> None:
        if self.current is not None:
            self.segments.append((self._seg_start, t, self.current))

    def describe(self) -> dict:
        return {"name": self.name, "labels": self.model.labels, "counts": self.model.counts(),
                "current": self.current, "region": self.model.region}
