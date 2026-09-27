"""Small reference models. Claude copies these patterns when it writes new ones."""
from __future__ import annotations

import numpy as np

from .. import native
from . import LabModel


class ColorMatch(LabModel):
    """Frame transform: learn an affine color mapping (3x4) from pairs of (source, target) pixels.

    Use: match a webcam's colors to a reference look, fix a color cast, or unify two cameras.
    hparams: ridge (float) regularization; degree (1 or 2) adds squared terms for curves.
    """

    task = "frame_transform"

    def __init__(self, ridge: float = 1e-3, degree: int = 1) -> None:
        super().__init__(ridge=ridge, degree=degree)
        self.ridge, self.degree = ridge, degree
        self.W: np.ndarray | None = None

    def _feat(self, px: np.ndarray) -> np.ndarray:
        x = px.reshape(-1, 3).astype(np.float32) / 255.0
        cols = [x, np.ones((x.shape[0], 1), np.float32)]
        if self.degree >= 2:
            cols.insert(1, x ** 2)
        return np.concatenate(cols, axis=1)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        F = self._feat(X)
        T = Y.reshape(-1, 3).astype(np.float32) / 255.0
        A = F.T @ F + self.ridge * np.eye(F.shape[1], dtype=np.float32)
        self.W = np.linalg.solve(A, F.T @ T)

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.W is not None, "model not trained"
        out = self._feat(X) @ self.W
        return np.clip(np.rint(out * 255.0), 0, 255).reshape(X.shape).astype(np.uint8)

    def evaluate(self, X: np.ndarray, Y: np.ndarray) -> dict[str, float]:
        err = self.predict(X).astype(np.float32) - Y.astype(np.float32)
        mse = float(np.mean(err ** 2))
        return {"mae": float(np.mean(np.abs(err))), "psnr": float(10 * np.log10(255 ** 2 / max(mse, 1e-9)))}

    def transform_frame(self, frame: np.ndarray, t: float = 0.0, strength: float = 1.0, **_) -> np.ndarray:
        assert self.W is not None, "model not trained"
        fast = native.affine_color(frame, self.W, self.degree, strength)  # C, multi-threaded
        if fast is not None:
            return fast
        out = self.predict(frame)
        if strength >= 1.0:
            return out
        return (frame * (1 - strength) + out * strength).astype(np.uint8)

    def state(self) -> dict[str, np.ndarray]:
        return {"W": self.W} if self.W is not None else {}

    def load_state(self, state: dict[str, np.ndarray]) -> None:
        self.W = state.get("W")


class LogisticFrameClassifier(LabModel):
    """Frame classifier on small feature vectors (e.g. downscaled gray frames or face crops).

    Use: detect user-specific moments (a gesture, a slide type, closed eyes) to create new anchors.
    hparams: lr, epochs, l2
    """

    task = "frame_classifier"

    def __init__(self, lr: float = 0.5, epochs: int = 200, l2: float = 1e-4) -> None:
        super().__init__(lr=lr, epochs=epochs, l2=l2)
        self.lr, self.epochs, self.l2 = lr, int(epochs), l2
        self.w: np.ndarray | None = None
        self.mu: np.ndarray | None = None
        self.sd: np.ndarray | None = None

    def _norm(self, X: np.ndarray) -> np.ndarray:
        X = X.reshape(len(X), -1).astype(np.float32)
        return np.hstack([(X - self.mu) / self.sd, np.ones((len(X), 1), np.float32)])

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        Xf = X.reshape(len(X), -1).astype(np.float32)
        self.mu, self.sd = Xf.mean(0), Xf.std(0) + 1e-6
        Z = self._norm(X)
        y = y.astype(np.float32)
        self.w = np.zeros(Z.shape[1], np.float32)
        for _ in range(self.epochs):
            p = 1 / (1 + np.exp(-(Z @ self.w)))
            self.w -= self.lr * (Z.T @ (p - y) / len(y) + self.l2 * self.w)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-(self._norm(X) @ self.w)))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(np.int32)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        return {"accuracy": float(np.mean(self.predict(X) == y))}

    def state(self) -> dict[str, np.ndarray]:
        return {"w": self.w, "mu": self.mu, "sd": self.sd} if self.w is not None else {}

    def load_state(self, state: dict[str, np.ndarray]) -> None:
        self.w, self.mu, self.sd = state.get("w"), state.get("mu"), state.get("sd")
