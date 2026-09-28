"""Model hub: small, open, public AI models VidAI installs and applies instead of Claude writing code.

Every catalog entry is verified (source, file, license, size, speed on a laptop CPU) and has an ADAPTER — a
built-in live effect that knows the model's input/output. Anything else can be found on Hugging Face with
`search()`; Claude then wraps it with a small adapter processor (saved to the library for next time).
Files live in ~/.vidai/assets/hub. Downloads go through VidAI's permission flow (ask / full access).
"""
from __future__ import annotations

import re
from pathlib import Path

from .models_dl import assets_dir

MP = "https://storage.googleapis.com/mediapipe-models"
_STYLES = {"mosaic": "colourful mosaic tiles", "candy": "bright candy colours", "udnie": "cubist painting",
           "rain-princess": "impressionist rain painting", "pointilism": "dotted pointillism painting"}

CATALOG: dict[str, dict] = {
    "emotion": {"title": "Emotion from your face (FER+)", "adapter": "emotion", "live": True,
                "url": "https://huggingface.co/onnxmodelzoo/emotion-ferplus-8/resolve/main/emotion-ferplus-8.onnx",
                "file": "emotion-ferplus-8.onnx", "mb": 35, "license": "Apache-2.0", "ms": 40,
                "words": ["emotion", "emotions", "mood", "feeling", "feelings", "expression", "smile", "happy", "sad"],
                "does": "reads your facial expression (neutral, happy, surprise, sad, angry…) as a live stat"},
    "gestures": {"title": "Hand gestures (MediaPipe)", "adapter": "gestures", "live": True,
                 "url": f"{MP}/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task",
                 "file": "gesture_recognizer.task", "mb": 8, "license": "Apache-2.0", "ms": 12,
                 "words": ["gesture", "gestures", "thumbs", "thumb", "victory", "peace", "palm", "fist", "pointing"],
                 "does": "recognises 👍 👎 ✌️ ✋ ☝️ ✊ 🤟 as live events (for rules: 'when I give a thumbs up…')"},
    "anime": {"title": "Anime look (AnimeGANv2 Hayao)", "adapter": "style", "live": False,
              "url": "https://huggingface.co/vumichien/AnimeGANv2_Hayao/resolve/main/AnimeGANv2_Hayao.onnx",
              "file": "AnimeGANv2_Hayao.onnx", "mb": 8.6, "license": "Apache-2.0", "ms": 1200,
              "params": {"layout": "nhwc_tanh", "size": 256},
              "words": ["anime", "cartoon", "ghibli", "hayao", "manga", "animated"],
              "does": "turns the picture into anime style (slow live, full quality when editing)"},
}
for _k, _d in _STYLES.items():
    CATALOG[f"style_{_k.replace('-', '_')}"] = {
        "title": f"{_k.replace('-', ' ').title()} painting style", "adapter": "style", "live": False,
        "url": f"https://huggingface.co/onnxmodelzoo/{_k}-9/resolve/main/{_k}-9.onnx", "file": f"{_k}-9.onnx",
        "mb": 6.7, "license": "Apache-2.0", "ms": 300, "params": {"layout": "nchw_255", "size": 224},
        "words": [_k.replace("-", " "), _k.split("-")[0], "painting", "painted", "artistic", "art", "style"],
        "does": f"paints the picture in {_d} style (slow live, full quality when editing)"}


def hub_dir() -> Path:
    d = assets_dir() / "hub"
    d.mkdir(parents=True, exist_ok=True)
    return d


def path(model_id: str) -> Path:
    return hub_dir() / CATALOG[model_id]["file"]


def installed(model_id: str) -> bool:
    p = path(model_id)
    return p.exists() and p.stat().st_size > 1000


def search(query: str, online: bool = False, limit: int = 5) -> dict:
    """Catalog entries that fit the request (+ optional Hugging Face ONNX candidates)."""
    words = set(re.findall(r"[a-z]+", query.lower()))
    scored = []
    for mid, m in CATALOG.items():
        score = sum(1 for w in m["words"] if w in words or (" " in w and w in query.lower()))
        if score:
            scored.append((score, mid))
    out = {"catalog": [{"id": mid, **{k: CATALOG[mid][k] for k in ("title", "does", "mb", "license", "live")},
                        "installed": installed(mid)} for _, mid in sorted(scored, reverse=True)]}
    if online:
        out["huggingface"] = search_hf(query, limit)
    return out


def search_hf(query: str, limit: int = 5) -> list[dict]:
    """ONNX models on Hugging Face (most downloaded first): repo, downloads, license, .onnx files with sizes."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return [{"error": "huggingface_hub is not installed"}]
    api = HfApi()
    out = []
    for m in api.list_models(search=query, filter="onnx", sort="downloads", limit=limit):
        try:
            info = api.model_info(m.id, files_metadata=True)
            files = [(s.rfilename, round((s.size or 0) / 1e6, 1)) for s in info.siblings if s.rfilename.endswith(".onnx")]
            lic = (info.card_data or {}).get("license") if info.card_data else None
        except Exception:
            files, lic = [], None
        out.append({"repo": m.id, "downloads": m.downloads, "license": lic, "onnx_files": files[:6],
                    "url_template": f"https://huggingface.co/{m.id}/resolve/main/<file>"})
    return out
