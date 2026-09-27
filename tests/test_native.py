"""The C kernels must match the NumPy fallback exactly (or within rounding)."""
import numpy as np
import pytest

from vidai import native

pytestmark = pytest.mark.skipif(not native.AVAILABLE, reason=f"native build unavailable: {native.BUILD_ERROR}")


@pytest.fixture
def numpy_only(monkeypatch):
    def run(fn, *a, **k):
        monkeypatch.setattr(native, "AVAILABLE", False)
        try:
            return fn(*a, **k)
        finally:
            monkeypatch.setattr(native, "AVAILABLE", True)
    return run


def test_rms_db(numpy_only):
    x = np.random.default_rng(0).normal(0, 0.1, 160_003).astype(np.float32)
    x[:32000] = 0
    np.testing.assert_allclose(native.rms_db(x, 1600), numpy_only(native.rms_db, x, 1600), atol=1e-3)


def test_find_runs(numpy_only):
    x = np.random.default_rng(1).uniform(-90, 0, 5000).astype(np.float32)
    for below in (True, False):
        assert native.find_runs(x, -45, below, 3) == numpy_only(native.find_runs, x, -45, below, 3)
    assert native.find_runs(np.array([-80, -80], np.float32), -45) == [(0, 2)]


def test_frame_mad(numpy_only):
    f = np.random.default_rng(2).integers(0, 256, (20, 90, 160)).astype(np.uint8)
    np.testing.assert_allclose(native.frame_mad(f), numpy_only(native.frame_mad, f), atol=1e-6)
    assert native.frame_mad(f[:0]).size == 0


def test_affine_color_matches_numpy_model():
    from vidai.lab.examples import ColorMatch

    rng = np.random.default_rng(3)
    X = rng.integers(0, 256, (3000, 3)).astype(np.uint8)
    for degree in (1, 2):
        m = ColorMatch(degree=degree)
        m.fit(X, (255 - X).astype(np.uint8))
        frame = rng.integers(0, 256, (36, 64, 3)).astype(np.uint8)
        c = m.transform_frame(frame)
        assert np.abs(c.astype(int) - m.predict(frame).astype(int)).max() <= 1
