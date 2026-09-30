"""VidAI downloads its own model files (so Claude never has to download anything during a recording).

    vidai setup            # fetch everything once (speech + hand + face)
Files go to $VIDAI_HOME/assets (default ~/.vidai/assets). Whisper models go to the Hugging Face cache.
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import socket
import threading
import urllib.parse
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
# pinned SHA-256 of every file VidAI downloads by itself; files without a pin are pinned on first download
SHA256 = {
    "hand_landmarker.task": "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
    "blaze_face_short_range.tflite": "b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f",
    "en_US-lessac-medium.onnx": "5efe09e69902187827af646e1a6e9d269dee769f9877d17b16b1b46eeaaf019f",
    "en_US-lessac-medium.onnx.json": "efe19c417bed055f2d69908248c6ba650fa135bc868b0e6abb3da181dab690a0",
    "selfie_segmenter.tflite": "191ac9529ae506ee0beefa6b2c945a172dab9d07d1e802a290a4e4038226658b",
    "selfie_segmenter_landscape.tflite": "490e9ea734313e0de10fa0cd9e3c6133e36ea4db2b7a49bde9ef019f72796b8e",
    "selfie_multiclass_256x256.tflite": "c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0",
    "rvm_mobilenetv3_fp32.onnx": "88d4531297118f595bf2fd60f6f566aec2e559393802d1f436c380f0cbbd2828",
    "face_landmarker.task": "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
}
MAX_BYTES = 2 * 1024 ** 3
_lock = threading.Lock()


def check_url(url: str) -> None:
    """Only https, and never a local / private address."""
    u = urllib.parse.urlparse(url)
    if u.scheme != "https" or not u.hostname:
        raise ValueError(f"only https downloads: {url}")
    try:
        addrs = {ai[4][0] for ai in socket.getaddrinfo(u.hostname, None)}
    except OSError as e:
        raise ValueError(f"cannot resolve {u.hostname}: {e}") from e
    for a in addrs:
        ip = ipaddress.ip_address(a.split("%")[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise ValueError(f"{u.hostname} is a local/private address")


def _pins_file() -> Path:
    return assets_dir() / "checksums.json"


def pinned(key: str) -> str | None:
    if key in SHA256:
        return SHA256[key]
    try:
        return json.loads(_pins_file().read_text()).get(key)
    except (OSError, ValueError):
        return None


def _pin(key: str, digest: str) -> None:
    f = _pins_file()
    try:
        pins = json.loads(f.read_text())
    except (OSError, ValueError):
        pins = {}
    pins[key] = digest
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(pins, indent=1))
    os.replace(tmp, f)


def fetch(url: str, target: Path, sha256: str | None = None, pin_key: str | None = None,
          max_bytes: int = MAX_BYTES, timeout: float = 300) -> tuple[Path, int, str]:
    """Download url -> target atomically. Checks the size limit and the SHA-256 (the given one, else the pinned
    one for pin_key; an unpinned pin_key is pinned now). The .part file never survives a failure."""
    check_url(url)
    sha256 = sha256 or (pinned(pin_key) if pin_key else None)
    tmp = target.with_suffix(target.suffix + ".part")
    h, n = hashlib.sha256(), 0
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "vidai"})
        with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 16):
                n += len(chunk)
                if n > max_bytes:
                    raise ValueError("download too large")
                h.update(chunk)
                f.write(chunk)
        digest = h.hexdigest()
        if sha256 and digest != sha256:
            raise ValueError(f"checksum mismatch for {target.name}: got {digest}, expected {sha256}")
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    if pin_key and not sha256:
        _pin(pin_key, digest)
    return target, n, digest


def assets_dir() -> Path:
    """Downloaded model files are shared (not user data): $VIDAI_ASSETS or ~/.vidai/assets, whatever VIDAI_HOME is."""
    d = Path(os.environ.get("VIDAI_ASSETS", Path.home() / ".vidai" / "assets"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure(name: str) -> Path:
    """Path to a model file, downloading it first if needed (atomic, safe from several threads)."""
    path = assets_dir() / name
    min_size = 100 if name.endswith(".json") else 1000
    if path.exists() and path.stat().st_size > min_size:
        return path
    with _lock:
        if not (path.exists() and path.stat().st_size > min_size):  # missing or truncated: (re)download
            fetch(MODEL_URLS[name], path, pin_key=name, timeout=120)
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
