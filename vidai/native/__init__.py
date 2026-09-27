"""C acceleration for VidAI hot loops.

fastops.c is compiled on first use with the system C compiler (gcc/cc, -O3, OpenMP if available)
into $VIDAI_HOME/native/ and loaded with ctypes. If no compiler is available every function falls
back to NumPy, so VidAI still works; `native.AVAILABLE` tells which path is used.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

_SRC = Path(__file__).with_name("fastops.c")
_lib: ctypes.CDLL | None = None
AVAILABLE = False
BUILD_ERROR: str | None = None


def _build() -> ctypes.CDLL | None:
    global BUILD_ERROR
    if os.environ.get("VIDAI_NO_NATIVE"):
        BUILD_ERROR = "disabled by VIDAI_NO_NATIVE"
        return None
    cc = os.environ.get("CC") or shutil.which("gcc") or shutil.which("cc") or shutil.which("clang")
    if not cc:
        BUILD_ERROR = "no C compiler found"
        return None
    src = _SRC.read_bytes()
    tag = hashlib.sha1(src + cc.encode()).hexdigest()[:12]
    home = Path(os.environ.get("VIDAI_HOME", Path.home() / ".vidai"))
    out = home / "native" / f"fastops_{tag}.so"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(f".{os.getpid()}.tmp")
        base = [cc, "-O3", "-shared", "-fPIC", "-std=c99", str(_SRC), "-o", str(tmp), "-lm"]
        for extra in (["-march=native", "-fopenmp"], ["-fopenmp"], []):
            r = subprocess.run(base[:2] + extra + base[2:], capture_output=True, text=True)
            if r.returncode == 0:
                break
        else:
            BUILD_ERROR = r.stderr[-500:]
            return None
        os.replace(tmp, out)
    try:
        return ctypes.CDLL(str(out))
    except OSError as e:
        BUILD_ERROR = str(e)
        return None


def _load() -> None:
    global _lib, AVAILABLE
    _lib = _build()
    if _lib is None:
        return
    P = np.ctypeslib.ndpointer
    i64, i32, f32 = ctypes.c_int64, ctypes.c_int32, ctypes.c_float
    _lib.rms_db.argtypes = [P(np.float32, flags="C"), i64, i32, P(np.float32, flags="C")]
    _lib.find_runs.argtypes = [P(np.float32, flags="C"), i64, f32, i32, i64,
                               P(np.int64, flags="C"), P(np.int64, flags="C"), i64]
    _lib.find_runs.restype = i64
    _lib.frame_mad.argtypes = [P(np.uint8, flags="C"), i64, i64, P(np.float32, flags="C")]
    _lib.affine_color.argtypes = [P(np.uint8, flags="C"), P(np.uint8, flags="C"), i64,
                                  P(np.float32, flags="C"), i32, f32]
    _lib.alpha_blend.argtypes = [P(np.uint8, flags="C"), i32, i32, P(np.uint8, flags="C"), i32, i32, i32, i32, f32]
    AVAILABLE = True


_load()


def rms_db(pcm: np.ndarray, hop: int) -> np.ndarray:
    pcm = np.ascontiguousarray(pcm, np.float32)
    m = pcm.size // hop
    if AVAILABLE:
        out = np.empty(m, np.float32)
        if m:
            _lib.rms_db(pcm, pcm.size, hop, out)
        return out
    rms = np.sqrt(np.mean(pcm[: m * hop].reshape(m, hop).astype(np.float64) ** 2, axis=1) + 1e-12)
    return np.clip(20 * np.log10(rms + 1e-9), -90, 0).astype(np.float32)


def find_runs(x: np.ndarray, thr: float, below: bool = True, min_len: int = 1) -> list[tuple[int, int]]:
    x = np.ascontiguousarray(x, np.float32)
    if AVAILABLE:
        cap = x.size // 2 + 1
        s, e = np.empty(cap, np.int64), np.empty(cap, np.int64)
        k = _lib.find_runs(x, x.size, thr, int(below), max(1, min_len), s, e, cap)
        return list(zip(s[:k].tolist(), e[:k].tolist()))
    mask = (x < thr) if below else (x >= thr)
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [(int(a), int(b)) for a, b in zip(starts, ends) if b - a >= max(1, min_len)]


def frame_mad(frames: np.ndarray) -> np.ndarray:
    frames = np.ascontiguousarray(frames, np.uint8)
    n = frames.shape[0]
    size = int(np.prod(frames.shape[1:])) if n else 0
    if AVAILABLE:
        out = np.empty(n, np.float32)
        if n:
            _lib.frame_mad(frames.reshape(-1), n, size, out)
        return out
    if n < 2:
        return np.zeros(n, np.float32)
    d = np.abs(np.diff(frames.astype(np.int16), axis=0)).reshape(n - 1, -1).mean(axis=1) / 255.0
    return np.concatenate([[0.0], d]).astype(np.float32)


def affine_color(frame: np.ndarray, W: np.ndarray, degree: int = 1, strength: float = 1.0) -> np.ndarray | None:
    """C path for ColorMatch; returns None when native code is unavailable (caller uses NumPy)."""
    if not AVAILABLE:
        return None
    src = np.ascontiguousarray(frame, np.uint8)
    out = np.empty_like(src)
    _lib.affine_color(src.reshape(-1), out.reshape(-1), src.size // 3,
                      np.ascontiguousarray(W, np.float32).reshape(-1), int(degree), float(strength))
    return out


def alpha_blend(frame: np.ndarray, overlay: np.ndarray, x: int = 0, y: int = 0, opacity: float = 1.0) -> np.ndarray:
    """Blend an RGBA overlay onto an RGB uint8 frame in place (clipped at the edges). Returns the frame."""
    fh, fw = frame.shape[:2]
    oh, ow = overlay.shape[:2]
    if AVAILABLE and frame.flags.c_contiguous:
        ov = np.ascontiguousarray(overlay, np.uint8)
        _lib.alpha_blend(frame.reshape(-1), fw, fh, ov.reshape(-1), ow, oh, int(x), int(y), float(opacity))
        return frame
    x0, y0, x1, y1 = max(0, -x), max(0, -y), min(ow, fw - x), min(oh, fh - y)
    if x0 >= x1 or y0 >= y1:
        return frame
    o = overlay[y0:y1, x0:x1].astype(np.float32)
    a = o[..., 3:4] / 255.0 * opacity
    d = frame[y + y0:y + y1, x + x0:x + x1]
    d[:] = np.clip(o[..., :3] * a + d * (1 - a) + 0.5, 0, 255).astype(np.uint8)
    return frame
