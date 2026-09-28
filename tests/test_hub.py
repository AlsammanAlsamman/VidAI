"""Model hub: catalog search, real model adapters, permission-gated install, colour from a photo."""
import os
import time
from pathlib import Path

import numpy as np
import pytest

from vidai import hub
from vidai.capture import CaptureConfig
from vidai.live.bus import LiveBus
from vidai.live.pipeline import LiveConfig, LivePipeline
from vidai.live.processors import REGISTRY, Context


def test_catalog_search():
    ids = lambda q: [m["id"] for m in hub.search(q)["catalog"]]
    assert ids("show my emotion") == ["emotion"]
    assert ids("react to my thumbs up gesture") == ["gestures"]
    assert ids("make me anime")[0] == "anime"
    assert "style_candy" in ids("candy painting style")
    assert ids("make a sandwich") == []
    for m in hub.CATALOG.values():
        assert m["license"] and m["url"].startswith("https://") and m["adapter"] in REGISTRY


def _wait(pred, timeout=20):
    t0 = time.time()
    while time.time() - t0 < timeout and not pred():
        time.sleep(0.05)
    return pred()


@pytest.mark.skipif(not hub.installed("style_mosaic"), reason="mosaic model not downloaded")
def test_style_adapter_repaints_the_picture():
    ctx = Context(LiveBus(), 320, 180)
    p = REGISTRY["style"]("fx_style", {"model": "style_mosaic", "strength": 1.0})
    f = np.zeros((180, 320, 3), np.uint8)
    f[:, :160] = (200, 60, 60)
    for _ in range(3):
        p.process(f.copy(), 0, ctx)
    assert _wait(lambda: p.out is not None)
    out = p.process(f.copy(), 0, ctx)
    p.enabled = False
    assert out.shape == f.shape and np.abs(out.astype(int) - f).mean() > 10  # it really repainted


@pytest.mark.skipif(not hub.installed("emotion"), reason="emotion model not downloaded")
def test_emotion_adapter_publishes_a_stat():
    from vidai.live.trackers import Face

    class Tracks:
        def face_now(self, max_age=0.5):
            return Face((0.3, 0.2, 0.4, 0.6), ((0.4, 0.4), (0.6, 0.4)), (0.5, 0.55), (0.5, 0.7))

    bus = LiveBus()
    ctx = Context(bus, 320, 240)
    ctx.tracks = Tracks()
    p = REGISTRY["emotion"]("fx_emotion", {"min_confidence": 0.0})
    f = np.random.default_rng(0).integers(0, 256, (240, 320, 3)).astype(np.uint8)
    for _ in range(4):
        p.process(f.copy(), 0, ctx)
        time.sleep(0.3)
    p.enabled = False
    assert _wait(lambda: any(e["kind"] == "emotion" for e in bus.history), 10)
    ev = next(e["data"] for e in bus.history if e["kind"] == "emotion")
    assert ev["label"] in ("neutral", "happy", "surprise", "sad", "angry", "disgust", "fear", "contempt")


def test_grade_copies_the_colours_of_a_photo(tmp_path):
    import cv2

    ref = np.zeros((100, 160, 3), np.uint8)
    ref[..., 2] = 200  # a very blue photo (RGB)
    cv2.imwrite(str(tmp_path / "blue.png"), cv2.cvtColor(ref, cv2.COLOR_RGB2BGR))
    p = REGISTRY["grade"]("g", {"image": str(tmp_path / "blue.png"), "strength": 1.0})
    f = np.random.default_rng(0).integers(60, 200, (90, 160, 3)).astype(np.uint8)
    out = p.process(f.copy(), 0, Context(LiveBus(), 160, 90))
    assert out[..., 2].mean() > out[..., 0].mean() + 60  # blue now dominates


def test_model_command_asks_permission_downloads_and_applies(tmp_path, monkeypatch):
    fake = tmp_path / "fake_model.bin"
    fake.write_bytes(b"x" * 5000)
    monkeypatch.setitem(hub.CATALOG, "fake", {"title": "Fake model", "adapter": "adjust", "live": True,
                                              "url": fake.as_uri(), "file": "fake_model.bin", "mb": 0.005,
                                              "license": "MIT", "words": ["fake"], "does": "test"})
    monkeypatch.setattr(hub, "hub_dir", lambda: tmp_path / "hub")
    (tmp_path / "hub").mkdir()
    pl = LivePipeline(CaptureConfig(mode="test", out_height=360, mic=False), LiveConfig(stt=False, speak=False),
                      None, session_dir=tmp_path)
    pl.start()
    assert pl.command({"cmd": "model", "model": "fake"})["state"] == "installing"
    assert _wait(lambda: bool(pl.asks), 5)  # VidAI asks "Master, I need to download the Fake model …"
    assert "Fake model" in list(pl.asks.values())[0]
    pl.command({"cmd": "confirm"}, source="voice")
    assert _wait(lambda: pl.chain.get("fx_fake") is not None, 10)
    pl.stop()
    assert (tmp_path / "hub" / "fake_model.bin").exists()
    assert "Fake model" in (tmp_path / "CREDITS.txt").read_text()


def test_gesture_rule_shows_a_sticker(tmp_path):
    pl = LivePipeline(CaptureConfig(mode="test", out_height=360, mic=False), LiveConfig(stt=False, speak=False),
                      None, session_dir=tmp_path)
    pl.ctx.need_tracking = lambda: None
    pl.start()
    pl.command({"cmd": "rule", "rule": {"id": "thumbs", "when": {"kind": "gesture", "where": {"name": "Thumb_Up"}},
                                        "do": [{"sticker": "👍", "for": 2}]}})
    pl.bus.publish("gesture", {"name": "Thumb_Up", "emoji": "👍"})
    stickers = [p for p in pl.chain.items if p.name.startswith("sticker")]
    pl.stop()
    assert stickers and stickers[0].params["what"] == "👍"
