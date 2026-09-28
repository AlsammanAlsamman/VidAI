"""VidAI downloads its own model files (so Claude never has to download anything during a recording).

    vidai setup            # fetch everything once (speech + hand + face)
Files go to $VIDAI_HOME/assets (default ~/.vidai/assets). Whisper models go to the Hugging Face cache.
"""
from __future__ import annotations

import os
import threading
import urllib.request
from pathlib import Path

MODEL_URLS = {
    "hand_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "blaze_face_short_range.tflite":
        "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
    "en_US-lessac-medium.onnx":
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
    "en_US-lessac-medium.onnx.json":
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
    "selfie_segmenter.tflite":
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
    "selfie_segmenter_landscape.tflite":
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite",
    "selfie_multiclass_256x256.tflite":
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
    "rvm_mobilenetv3_fp32.onnx":
        "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx",
    "face_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
}
_lock = threading.Lock()


def assets_dir() -> Path:
    """Downloaded model files are shared (not user data): $VIDAI_ASSETS or ~/.vidai/assets, whatever VIDAI_HOME is."""
    d = Path(os.environ.get("VIDAI_ASSETS", Path.home() / ".vidai" / "assets"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure(name: str) -> Path:
    """Path to a model file, downloading it first if needed (atomic, safe from several threads)."""
    path = assets_dir() / name
    if path.exists() and path.stat().st_size > 1000:
        return path
    with _lock:
        if not path.exists():
            tmp = path.with_suffix(path.suffix + ".part")
            with urllib.request.urlopen(MODEL_URLS[name], timeout=120) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 16):
                    f.write(chunk)
            os.replace(tmp, path)
    return path


def ensure_whisper(model: str = "base") -> None:
    from faster_whisper import WhisperModel

    WhisperModel(model, device="cpu", compute_type="int8")


def setup(whisper: tuple[str, ...] = ("base",), log=print) -> dict:
    out = {}
    for name in MODEL_URLS:
        try:
            out[name] = str(ensure(name))
            log(f"ok  {name}")
        except Exception as e:
            out[name] = f"error: {e}"
            log(f"ERR {name}: {e}")
    for m in whisper:
        try:
            ensure_whisper(m)
            out[f"whisper-{m}"] = "ok"
            log(f"ok  whisper {m}")
        except Exception as e:
            out[f"whisper-{m}"] = f"error: {e}"
            log(f"ERR whisper {m}: {e}")
    return out


def prefetch_background() -> None:
    """Start downloading missing tracker models in the background (called when a session starts)."""
    def run() -> None:
        for name in MODEL_URLS:
            try:
                ensure(name)
            except Exception:
                pass

    threading.Thread(target=run, daemon=True).start()
