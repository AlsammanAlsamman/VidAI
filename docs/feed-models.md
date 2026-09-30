# Feed models: training small models on your own video, locally

VidAI can train small models **on the live camera feed, on your computer, while you record**. No GPU, no
upload, no dataset. They make effects look filmed instead of pasted: the rabbit ears take the room's light,
and your hair falls in front of their base.

This page is the protocol every feed model follows, so new ones (hands in front of stickers, a faster
background cut-out, your face for glasses) can be added the same way.

## The protocol

```
            a few times per second (worker thread)                 every frame (~1 ms)
 frame ──► observe() ──► offer(sample) ──► teach(sample) ──► learn(sample, labels) ──► score
                                            slow, good          updates the student
                                            (pretrained model,  (tiny NumPy model on
                                             or statistics)      YOUR colours/light)
                                                                        │
 effects ──► ctx.composite(frame, overlay, x, y, occlude={"hair"}) ◄── infer()  (only once ready)
                                                                        │
 recording stops ──► close() ──► saved to ~/.vidai/models/feed_<key> ──► next session starts trained
```

| Part | Where it runs | Rule |
|---|---|---|
| `observe(frame, t, ctx)` | frame loop | must be cheap: crop/downscale and `self.offer(sample)` (only the newest sample is kept) |
| `teach(sample)` | worker thread | may be slow (a pretrained model: ~90 ms is fine). Return labels, or `None` if the model learns from the frames themselves (statistics) |
| `learn(sample, labels)` | worker thread | first **score** the student on this sample (it has not trained on it yet), then train on it. Return the score (or `None` if not checked) |
| `infer(...)` | frame loop | fast (~1 ms); return nothing until `self.ready` |
| `student_state()` / `load_student()` | stop / start | the student's arrays; `persist = False` for things that change every session (light) |

**Ready rule** (same idea as the lab's train-until-suitable loop): the score is an exponential average of the
checks; the student is used only when `score >= target` over at least `min_checks` checks. A model loaded
from memory needs one good check on the new video before it is used.

**Safety rules**
- The frame loop never waits for a teacher; a teacher that fails publishes an `error` and the recording goes on.
- Under load (performance governor) the worker slows down (`rate / (1 + quality)`).
- Feed models are processors named `_feed_<key>`: hidden from the user's effect list, never carried or
  undone, closed (thread stopped, student saved) when no effect needs them any more or when recording stops.
- Students are plain arrays in the lab registry (`vidai models` lists them as `feed_<key>`).

## Using feed models in an effect

```python
from vidai.live.processors import LiveProcessor, register

@register
class Crown(LiveProcessor):
    tracking = True
    feeds = frozenset({"lighting", "hair"})      # started automatically, shared by every effect

    def process(self, frame, t, ctx):
        ...
        ctx.composite(frame, sprite_rgba, x, y, occlude={"hair"})   # instead of native.alpha_blend
        return frame
```

`ctx.composite` applies whatever is ready: lighting always, masks listed in `occlude`. Before the models are
ready it is a plain blend, so effects never wait. Use it for things **in the scene** (worn, held, attached);
keep screen graphics (titles in a corner, captions) on `native.alpha_blend` so they stay crisp.

Built-in: `attach` stickers use it automatically (`realistic` on by default, `behind_hair` for things worn on
the head). Say **"VidAI, make it realistic"** to put head effects behind the hair.

## Adding a new feed model

```python
from vidai.live.feed import FeedModel, feed

@feed
class Hands(FeedModel):
    key = "hands"               # effects ask for it with feeds = {"hands"}
    type_name = "feed_hands"
    tracking = True
    rate, target, min_checks = 3.0, 0.7, 4

    def observe(self, frame, t, ctx): ...       # crop around the hand, self.offer(...)
    def teach(self, sample): ...                # e.g. MediaPipe hand landmarks -> a hand mask
    def learn(self, sample, labels): ...        # score, then train a PixelMLP on the user's skin
    def infer(self, frame, ctx): ...            # (mask, region) like Hair.infer
    def student_state(self): ...
    def load_student(self, state): ...
```

Checklist:
1. The teacher exists and runs on a laptop CPU (a pretrained ONNX/TFLite model already in VidAI, or statistics).
2. The student is small enough for ~1 ms per frame (`PixelMLP` on a ~96 px crop is a good default).
3. `learn` scores before it trains (honest score), and returns `None` when the sample says nothing.
4. A test with a fake teacher: the student becomes ready, `infer` is fast, it is saved and loaded, a failing
   teacher never breaks the chain (see `tests/test_feed.py`).
5. Document the key here and in `live_guide("feed")`.

## Built-in feed models

| Key | Teacher | Student | Used for |
|---|---|---|---|
| `lighting` | none: statistics of the feed (light and tint per region, camera softness, sensor grain) | running averages | every in-scene overlay takes the room's light, softness and grain |
| `hair` | MediaPipe multiclass selfie segmenter (hair class, ~90 ms) at 2.5 Hz | `PixelMLP` (12 features: colour, local colour, edges, position vs the face) | hair in front of things worn on the head |

## What this cannot do

Feed models make overlays *fit* the video. They do not generate new realistic pixels (a real pair of rabbit
ears, different glasses): that needs large generative models, thousands of examples and a strong GPU, and
would not run live on a laptop. For realism beyond drawn effects, use a real photo (PNG with transparency)
through `attach` — it gets the same light matching and hair occlusion.
