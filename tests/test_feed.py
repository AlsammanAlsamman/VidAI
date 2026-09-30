"""Feed models: trained locally on the live video (teacher -> student -> score -> memory) + the compositor."""
import time

import cv2
import numpy as np

from vidai import lab
from vidai.live import feed as F
from vidai.live.bus import LiveBus
from vidai.live.processors import REGISTRY, Context, LiveProcessor, ProcessorChain


class _Face:
    box = (0.4, 0.4, 0.2, 0.3)
    eyes = ((0.45, 0.5), (0.55, 0.5))
    top = (0.5, 0.4)
    width = 0.2
    nose = (0.5, 0.55)
    mouth = (0.5, 0.62)


class _Tracks:
    face_at = 0.0

    def face_now(self, max_age=0.5):
        return _Face()

    def hand(self, which="any", max_age=0.4):
        return None

    def feed(self, frame):
        pass

    def close(self):
        pass


HAIR_RGB = np.array([70, 45, 30])


def _scene(light=(40, 45, 60)):
    f = np.full((360, 640, 3), light, np.uint8)
    cv2.ellipse(f, (320, 150), (90, 60), 0, 0, 360, tuple(int(v) for v in HAIR_RGB), -1)
    cv2.ellipse(f, (320, 190), (60, 55), 0, 0, 360, (190, 150, 120), -1)
    return f


def _teacher(sample):  # stands in for the MediaPipe segmenter: hair = the hair colour
    crop, _ = sample
    return (np.abs(crop.astype(int) - HAIR_RGB).sum(-1) < 30).astype(np.float32)


def _chain():
    ctx = Context(LiveBus(), 640, 360)
    ctx.tracks = _Tracks()
    return ctx, ProcessorChain(ctx)


def _run_until(chain, cond, frame, seconds=6.0):
    end = time.monotonic() + seconds
    i = 0
    while time.monotonic() < end and not cond():
        chain.run(frame.copy(), i / 30)
        i += 1
        time.sleep(0.03)
    return cond()


def test_effects_start_the_feed_models_they_draw_with_and_stop_them(monkeypatch):
    monkeypatch.setattr(F.Hair, "teach", lambda self, s: _teacher(s))
    ctx, chain = _chain()
    chain.add(REGISTRY["attach"]("fx_crown", {"what": "👑", "to": "head", "behind_hair": True}))
    assert {p.name for p in chain.items} == {"_feed_lighting", "_feed_hair", "fx_crown"}
    screen = chain.add(REGISTRY["attach"]("fx_logo", {"what": "⭐", "to": "screen"}))
    assert screen.wants_feeds() == set()  # screen graphics stay crisp
    hair = chain.get("_feed_hair")
    chain.remove("fx_crown")
    assert chain.get("_feed_hair") is None and hair.closed  # nobody needs them any more
    assert chain.get("_feed_lighting") is None


def test_hair_student_learns_from_the_teacher_then_is_used(monkeypatch):
    monkeypatch.setattr(F.Hair, "teach", lambda self, s: _teacher(s))
    ctx, chain = _chain()
    hair = chain.add(F.Hair("_feed_hair"))
    assert hair.infer(_scene(), ctx) is None  # not ready: never used before it is good enough
    assert _run_until(chain, lambda: hair.ready, _scene())
    assert hair.score >= hair.target and hair.checks >= hair.min_checks
    mask, (x0, y0, x1, y1) = hair.infer(_scene(), ctx)
    assert mask[100 - y0, 320 - x0] > 0.8 and mask[210 - y0, 320 - x0] < 0.2  # hair yes, face no
    t = time.perf_counter()
    for _ in range(20):
        hair._mask_cache = None
        hair.infer(_scene(), ctx)
    assert (time.perf_counter() - t) / 20 < 0.01  # the student is fast (the teacher is ~90 ms)
    chain.close_all()


def test_student_is_remembered_for_the_next_session(monkeypatch):
    monkeypatch.setattr(F.Hair, "teach", lambda self, s: _teacher(s))
    ctx, chain = _chain()
    hair = chain.add(F.Hair("_feed_hair"))
    assert _run_until(chain, lambda: hair.ready, _scene())
    chain.close_all()  # recording stops -> saved to the lab
    assert any(m["name"] == "feed_hair" for m in lab.list_models())
    again = F.Hair("_feed_hair")
    assert again._loaded and not again.ready and again.student.n == 1  # trained, re-checked on new frames
    np.testing.assert_allclose(again.student.W1, hair.student.W1)
    again.close()


def test_lighting_matches_overlays_to_the_room():
    ctx, chain = _chain()
    light = chain.add(F.Lighting("_feed_lighting"))
    assert _run_until(chain, lambda: light.cells is not None, _scene((30, 32, 40)))
    white = np.full((40, 40, 4), 255, np.uint8)
    dark = light.adapt(white, 20, 300, 640, 360)[..., :3].mean()
    light.cells = None
    assert _run_until(chain, lambda: light.cells is not None, _scene((235, 235, 235)))
    bright = light.adapt(white, 20, 300, 640, 360)[..., :3].mean()
    assert dark < 200 < bright  # dimmer in a dark room, (almost) unchanged in a bright one
    assert white.min() == 255  # the cached sprite itself is never changed
    chain.close_all()


def test_composite_hides_the_overlay_behind_hair_only(monkeypatch):
    monkeypatch.setattr(F.Hair, "teach", lambda self, s: _teacher(s))
    ctx, chain = _chain()
    hair = chain.add(F.Hair("_feed_hair"))
    assert _run_until(chain, lambda: hair.ready, _scene())
    red = np.zeros((40, 200, 4), np.uint8)
    red[..., 0], red[..., 3] = 255, 255
    frame = _scene()
    ctx.composite(frame, red, 180, 80, occlude={"hair"}, strength=1.0)  # across the hair and the background
    assert frame[100, 320, 0] < 150  # over the hair: hidden, hair shows
    assert frame[100, 185, 0] > 200 and frame[100, 185, 1] < 80  # outside the hair: drawn
    chain.close_all()


def test_a_failing_teacher_never_breaks_the_recording(monkeypatch):
    def boom(self, s):
        raise RuntimeError("no model")

    monkeypatch.setattr(F.Hair, "teach", boom)
    ctx, chain = _chain()
    hair = chain.add(F.Hair("_feed_hair"))
    errs = []
    ctx.bus.subscribe(lambda ev: errs.append(ev), {"error"})
    _run_until(chain, lambda: bool(errs), _scene(), 3)
    assert errs and not hair.ready and hair.enabled
    out = chain.run(_scene(), 1.0)
    assert out.shape == (360, 640, 3)
    chain.close_all()


def test_custom_feed_model_follows_the_protocol():
    """How other VidAI features add their own: subclass, set key, implement observe/teach/learn."""

    @F.feed
    class Brightness(F.FeedModel):
        key = "brightness_test"
        type_name = "feed_brightness_test"
        persist = False
        target = 0.5

        def observe(self, frame, t, ctx):
            self.offer(float(frame.mean()))

        def teach(self, sample):
            return sample / 255.0

        def learn(self, sample, labels):
            self.level = labels
            return 1.0

    class Uses(LiveProcessor):
        feeds = frozenset({"brightness_test"})

    ctx, chain = _chain()
    chain.add(Uses("fx_uses"))
    m = chain.get("_feed_brightness_test")
    assert _run_until(chain, lambda: m.ready, _scene())
    assert ctx.feed["brightness_test"] is m and 0 < m.level < 1
    chain.close_all()
    F.FEEDS.pop("brightness_test")
    REGISTRY.pop("brightness", None)
