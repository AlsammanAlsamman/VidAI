"""Live processing: bus, processors, rules, voice commands, instant models, Claude's control channel."""
import json
import time
from pathlib import Path

import numpy as np
import pytest

from vidai import ffmpeg
from vidai.capture import CaptureConfig
from vidai.live.bus import LiveBus, read_events
from vidai.live.learn import InstantKNN, LiveLearner
from vidai.live.pipeline import LiveConfig, LivePipeline
from vidai.live.processors import REGISTRY, Context, LiveProcessor, ProcessorChain
from vidai.live.rules import RuleEngine, fill, matches
from vidai.live.sensors import AudioSensor, parse_command

DATA = Path(__file__).parent / "data"


def test_parse_voice_commands():
    assert parse_command("VidAI, zoom out please") == {"command": "zoom_out", "args": "please"}
    assert parse_command("vid ai stop recording") == {"command": "stop", "args": ""}
    assert parse_command("Video AI new section Results")["command"] == "section"
    assert parse_command("فيداي قسم جديد النتائج") == {"command": "section", "args": "النتائج"}
    assert parse_command("VidAI make the title bigger") == {"command": "claude", "args": "make the title bigger"}
    assert parse_command("videos are great") is None


def test_rules_match_and_fill():
    r = {"when": {"kind": "silence_end", "where": {"duration": {">": 2}}}}
    assert matches(r, {"kind": "silence_end", "data": {"duration": 3}})
    assert not matches(r, {"kind": "silence_end", "data": {"duration": 1}})
    assert not matches(r, {"kind": "speech_end", "data": {"duration": 3}})
    assert matches({"when": {"kind": "screen_text", "where": {"text": {"contains": "error"}}}},
                   {"kind": "screen_text", "data": {"text": "Fatal ERROR: x"}})
    assert fill({"show_text": "{section} ({duration}s)"}, {"section": "Intro", "duration": 3}) == {
        "show_text": "Intro (3s)"}


def test_rule_engine_cooldown():
    bus = LiveBus()
    fired = []
    eng = RuleEngine(bus, lambda a, ev, r: fired.append(a))
    eng.add({"id": "r", "when": {"kind": "loud"}, "do": [{"mark": "important"}], "cooldown": 5})
    for t in (1.0, 2.0, 7.0):
        bus.publish("loud", {"db": -1}, t)
    assert len(fired) == 2


def test_audio_sensor_events():
    bus = LiveBus()
    utter = []
    a = AudioSensor(bus, min_silence=0.5, on_utterance=lambda s, t0, t1: utter.append((t0, t1)))
    tone = (0.3 * np.sin(np.linspace(0, 2 * np.pi * 440, 1600))).astype(np.float32)
    quiet = np.zeros(1600, np.float32)
    for block in [quiet] * 15 + [tone] * 20 + [quiet] * 10:
        a.feed(block)
    kinds = [e["kind"] for e in bus.history]
    assert "speech_start" in kinds and "speech_end" in kinds and "silence_end" in kinds
    assert len(utter) == 1 and utter[0][0] == pytest.approx(1.3, abs=0.3) and utter[0][1] == pytest.approx(3.5, abs=0.3)


def test_processor_chain_budget_and_errors():
    bus = LiveBus()
    ch = ProcessorChain(Context(bus, 320, 180))

    class Boom(LiveProcessor):
        def process(self, frame, t, ctx):
            raise RuntimeError("bad code")

    class Slow(LiveProcessor):
        budget_ms = 0.001

        def process(self, frame, t, ctx):
            time.sleep(0.002)
            return frame

    ch.add(Boom("boom"))
    ch.add(Slow("slow"))
    ch.add(REGISTRY["text"]("title", {"text": "Hi"}))
    f = np.zeros((180, 320, 3), np.uint8)
    for i in range(40):
        f = ch.run(f, i / 30)
    assert not ch.get("boom").enabled and not ch.get("slow").enabled and ch.get("title").enabled
    errs = [e["data"] for e in bus.history if e["kind"] == "error"]
    assert any("bad code" in e["error"] for e in errs) and any("too slow" in e["error"] for e in errs)
    assert f.max() > 0  # the text still rendered


def test_transforms_run_before_overlays():
    ch = ProcessorChain(Context(LiveBus(), 320, 180))
    ch.add(REGISTRY["text"]("logo", {"text": "L"}))
    ch.add(REGISTRY["zoom"]("zoom", {}))
    assert [p.name for p in ch.items] == ["zoom", "logo"]


def test_instant_model_learns_and_corrects():
    bus = LiveBus()
    L = LiveLearner(bus, "slide", ["yes", "no"])
    red = np.zeros((180, 320, 3), np.uint8)
    red[..., 0] = 220
    blue = np.zeros((180, 320, 3), np.uint8)
    blue[..., 2] = 220
    blue[40:140, 60:260] = 255
    t = 0.0
    for _ in range(6):
        L.feed(red, t); t += 0.2
    L.label("yes")
    for _ in range(6):
        L.feed(blue, t); t += 0.2
    L.label("no")
    for img in [red] * 8 + [blue] * 8:
        L.feed(img, t); t += 0.6
    preds = [e["data"]["label"] for e in bus.history if e["kind"] == "learner"]
    assert preds == ["yes", "no"]
    # correction: the model now says "no" on blue; the user says it is wrong -> blue becomes "yes"
    out = L.wrong()
    assert out["corrected_from"] == "no"
    for _ in range(4):
        L.feed(blue, t); t += 0.6
    assert L.current == "yes"
    m = InstantKNN(labels=["yes", "no"])
    m.load_state(L.model.state())
    assert m.counts()["yes"] > 0


def _pipe(tmp_path, live, **cap):
    cfg = CaptureConfig(mode="test", out_height=360, **cap)
    return LivePipeline(cfg, live, output=tmp_path / "video.mkv", session_dir=tmp_path)


def test_pipeline_claude_control_rules_and_file(tmp_path):
    live = LiveConfig(stt=False, rules=[{"id": "pause", "when": {"kind": "silence_end", "where": {"duration": {">": 1}}},
                                        "do": [{"show_text": "pause {duration}", "for": 1}]}])
    pl = _pipe(tmp_path, live)
    pl.start()

    def send(c):
        with open(tmp_path / "control.jsonl", "a") as f:
            f.write(json.dumps(c) + "\n")

    time.sleep(1.2)
    from vidai import actions

    actions.allow_code(tmp_path)  # the user allowed Claude's effect code in this session
    code = tmp_path / "invert.py"
    code.write_text("from vidai.live.processors import LiveProcessor, register\n"
                    "@register\nclass Invert(LiveProcessor):\n"
                    "    def process(self, frame, t, ctx):\n        return 255 - frame\n")
    send({"cmd": "add", "name": "inv", "file": str(code), "for": 1.0})
    send({"cmd": "text", "text": "hello", "for": 2})
    send({"cmd": "nonsense"})
    time.sleep(3.5)
    pl.stop()
    ev = read_events(tmp_path / "live.jsonl", limit=10 ** 6)
    acks = [e["data"]["command"] for e in ev if e["kind"] == "ack"]
    assert "add" in acks and "text" in acks and "rule" in acks
    assert any(e["kind"] == "error" and e["data"].get("command") == "nonsense" for e in ev)
    assert any(e["kind"] == "ack" and e["data"].get("source") == "rule:pause" for e in ev)
    info = ffmpeg.probe(str(tmp_path / "video.mkv"))
    assert info.has_audio and info.has_video and 4 < info.duration < 7
    assert not list(tmp_path.glob("*.part-*"))  # parts were joined and removed
    added = next(e["t"] for e in ev if e["kind"] == "action" and e["data"].get("name") == "inv")
    f_in = ffmpeg.read_frames(str(tmp_path / "video.mkv"), 30, 64, 36, start=added + 0.3, duration=0.04)[0]
    f_after = ffmpeg.read_frames(str(tmp_path / "video.mkv"), 30, 64, 36, start=added + 2.5, duration=0.04)[0]
    # the left colour bar is static: inverted while the processor ran, normal after -> they add up to ~255
    bar_in, bar_after = f_in[28:34, 0:4].mean(), f_after[28:34, 0:4].mean()
    assert abs(bar_in + bar_after - 255) < 25 and abs(bar_in - bar_after) > 60


@pytest.mark.slow
def test_pipeline_voice_commands_from_speech(tmp_path):
    """English speech file as the microphone: transcripts, a section marker, a live zoom, a request to Claude."""
    live = LiveConfig(stt=True, stt_language="en", stt_model="base")
    pl = _pipe(tmp_path, live, mic_source=str(DATA / "speech_en.wav"))
    pl.start()
    time.sleep(18.5)
    pl.stop()
    ev = read_events(tmp_path / "live.jsonl", limit=10 ** 6)
    texts = [e["data"]["text"].lower() for e in ev if e["kind"] == "transcript"]
    assert any("blast" in t for t in texts)
    cmds = [e["data"]["command"] for e in ev if e["kind"] == "voice_command"]
    assert "section" in cmds and "zoom_in" in cmds and "claude" in cmds
    assert any(e["kind"] == "marker" and e["data"]["note"] == "results" for e in ev)
    assert any(e["kind"] == "claude" and "title" in e["data"]["message"] for e in ev)
    assert any(e["kind"] == "action" and e["data"].get("what") == "zoom" for e in ev)


def test_claude_slow_loop_through_service_tools(tmp_path):
    """What Claude does while the user records: read stats, send commands, write/fix a processor."""
    from vidai import AnchorFile, service
    from vidai.session import SessionRecorder, create_session

    s = create_session({"title": "live test", "style": "screencast"}, capture=CaptureConfig(mode="test", out_height=360),
                       root=tmp_path, live={"stt": False, "ocr": False})
    r = SessionRecorder(s)
    r.start()
    time.sleep(1.5)
    st = service.live_stats(s.dir)
    assert st["state"] == "recording" and st["next"] > 0
    res = service.live_control(s.dir, [{"cmd": "text", "text": "Claude was here", "for": 2},
                                       {"cmd": "mark", "type": "section", "note": "demo"}])
    assert res["missing"] == 0 and all(x["kind"] == "ack" for x in res["results"])
    from vidai import actions

    actions.allow_code(s.dir)  # the user said yes once ("run effect code written by Claude")
    bad = ("from vidai.live.processors import LiveProcessor, register\n"
           "@register\nclass Tint(LiveProcessor):\n"
           "    def process(self, frame, t, ctx):\n        return frame * undefined_name\n")
    out = service.live_processor(s.dir, "tint", bad)
    assert "undefined_name" in out["runtime_error"]["error"]  # Claude sees the bug...
    good = bad.replace("frame * undefined_name", "np.clip(frame.astype(np.int16) + 40, 0, 255).astype(np.uint8)")
    good = "import numpy as np\n" + good
    out = service.live_processor(s.dir, "tint", good, duration=1.0)
    assert "runtime_error" not in out  # ...and fixes it instantly
    status = service.live_status(s.dir)
    assert any(p["name"] == "tint" for p in status["processors"])
    time.sleep(1.5)
    r.stop()
    a = AnchorFile.load(s.video_path)
    whats = [e.data.get("what") for e in a.events_of("live_action")]
    assert "text" in whats and "processor_added" in whats
    assert any(e.data.get("note") == "demo" for e in a.events_of("markers"))
    assert service.live_control(s.dir, [{"cmd": "status"}])["error"].startswith("not recording")


def test_thinking_icon_while_claude_works(tmp_path):
    """A voice request to Claude shows the animated icon in the video until Claude answers (done)."""
    pl = _pipe(tmp_path, LiveConfig(stt=False))
    pl.start()
    time.sleep(1.0)
    pl.bus.publish("claude", {"message": "make the title bigger", "source": "voice"})
    time.sleep(1.0)
    assert pl.pending and pl._thinking.enabled
    t_on = pl.clock()
    pl.command({"cmd": "text", "text": "BIG TITLE", "size": 0.1})
    pl.command({"cmd": "done"})
    assert not pl.pending and not pl._thinking.enabled
    pl.bus.publish("claude", {"message": "rule note", "source": "rule:x"})  # rules don't show the icon
    assert not pl.pending
    time.sleep(1.0)
    pl.stop()
    whats = [e["data"].get("what") for e in read_events(tmp_path / "live.jsonl", limit=10 ** 6) if e["kind"] == "action"]
    assert whats.count("thinking_on") == 1 and whats.count("thinking_off") == 1
    # the icon is in the top-right corner of the recorded video while Claude was thinking, not after
    during = ffmpeg.read_frames(str(tmp_path / "video.mkv"), 30, 640, 360, start=t_on - 0.3, duration=0.04)[0]
    after = ffmpeg.read_frames(str(tmp_path / "video.mkv"), 30, 640, 360, start=t_on + 0.6, duration=0.04)[0]
    corner_d, corner_a = during[10:80, 540:630].astype(int), after[10:80, 540:630].astype(int)
    assert np.abs(corner_d - corner_a).mean() > 10


def test_voice_record_command_in_preview(tmp_path):
    """In preview (not recording) voice still works: 'VidAI record' asks the GUI to start recording."""
    started = []
    cfg = CaptureConfig(mode="test", out_height=360)
    pl = LivePipeline(cfg, LiveConfig(stt=False), None, session_dir=tmp_path, on_start_request=lambda: started.append(1))
    pl.start()
    time.sleep(0.5)
    pl.bus.publish("voice_command", {"command": "record", "args": "", "text": "VidAI record"})
    pl.bus.publish("voice_command", {"command": "zoom_in", "args": "", "text": "VidAI zoom in"})
    time.sleep(0.3)
    assert started == [1] and pl.chain.get("zoom") is not None  # effects work in preview too
    pl.stop()
    assert parse_command("VidAI, start recording") == {"command": "record", "args": ""}
    assert parse_command("فيداي ابدأ")["command"] == "record"
    # the recording pipeline continues the same event log without reusing sequence numbers
    pl2 = LivePipeline(cfg, LiveConfig(stt=False), tmp_path / "v.mkv", session_dir=tmp_path)
    assert pl2.bus._seq >= max(e["seq"] for e in read_events(tmp_path / "live.jsonl", limit=10 ** 6))
    pl2.bus.close()


def test_wake_word_then_pause_then_command():
    """'VidAI' <pause> 'zoom in' arrives as two utterances but is one command."""
    from vidai.live.sensors import SpeechToText

    bus = LiveBus()
    stt = SpeechToText.__new__(SpeechToText)  # no Whisper needed for the text logic
    stt.bus, stt.wake_words, stt.transcripts = bus, None, []
    stt.armed_until, stt.arm_seconds = -1.0, 5.0
    stt.handle_text("VidAI", 10.0, 10.4, "en")
    stt.handle_text("Zoom in.", 11.2, 12.0, "en")
    stt.handle_text("zoom in", 30.0, 30.6, "en")  # much later: plain speech, not a command
    cmds = [e["data"]["command"] for e in bus.history if e["kind"] == "voice_command"]
    assert cmds == ["zoom_in"]
    assert any(e["kind"] == "action" and e["data"]["what"] == "listening" for e in bus.history)


def test_whisper_hallucinations_are_dropped():
    from vidai.live.sensors import is_hallucination

    for bad in ["I'm sorry.", "Thank you for watching.", "www.mesmerism.info", "3. 3. 3. 4. 3. 4.", "You",
                "dhidhadidhaddhaddhadaadbhaddhadaad",
                "I'm not sure. I'm not sure. I'm not sure. I'm not sure. I'm not sure."]:
        assert is_hallucination(bad), bad
    for good in ["VidAI zoom in", "Today we will learn how to use BLAST.", "Thank you all for joining this session"]:
        assert not is_hallucination(good), good


def test_request_to_claude_waits_for_the_rest_of_the_sentence(tmp_path):
    """'VidAI add a text above my head saying...' <pause> 'an award' -> one request."""
    pl = LivePipeline(CaptureConfig(mode="test", out_height=360, mic=False), LiveConfig(stt=False), None,
                      session_dir=tmp_path)
    pl.request_gap = 0.5
    pl.start()
    pl.bus.publish("voice_command", {"command": "claude", "args": "add a text above my head saying", "text": "..."})
    pl.bus.publish("speech_start", {})
    time.sleep(0.8)
    assert not any(e["kind"] == "claude" for e in pl.bus.history)  # still waiting: the user is talking
    pl.bus.publish("transcript", {"text": "an award.", "is_command": False})
    time.sleep(1.0)
    reqs = [e["data"]["message"] for e in pl.bus.history
            if e["kind"] == "claude" or (e["kind"] == "action" and e["data"].get("what") == "fast_request")]
    pl.stop()
    assert reqs == ["add a text above my head saying an award"]  # one request, not two
    assert pl.chain.get("fx_text_above_head").params["what"] == "text:An Award"  # text, not a trophy


class _FakeTracks:
    def __init__(self, hands=(), face=None):
        self.hands, self.face = list(hands), face

    def hand(self, which="any", max_age=0.4):
        for h in self.hands:
            if which in ("any", "hand") or h.side.lower() == which.lower():
                return h
        return None

    def face_now(self, max_age=0.5):
        return self.face


def test_fast_path_understands_common_requests():
    from vidai.live.intents import match

    ch = ProcessorChain(Context(LiveBus(), 640, 360))
    ch.ctx.need_tracking = lambda: None

    def apply(text):
        cmds = match(text, ch) or []
        for c in cmds:
            c = dict(c)
            k = c.pop("cmd")
            if k == "add":
                ch.add(REGISTRY[c["type"]](c["name"], c.get("params", {})))
            elif k == "remove":
                ch.remove(c["name"])
            elif k == "set":
                ch.get(c["name"]).configure(c["params"])
        return cmds

    assert apply("add an apple in my hand")[0]["params"] == {"what": "🍎", "to": "hand"}
    cmds = apply("put an orange on my other hand")
    assert [c["params"]["to"] for c in cmds] == ["right_hand", "left_hand"]
    assert apply("at horns on my head")[0]["params"]["to"] == "head"  # misheard "add"
    assert apply("make my eyes pop up")[0]["type"] == "big_eyes"
    assert apply("bigger apple")[0]["params"]["scale"] == pytest.approx(1.4)
    assert apply("remove the apple") == [{"cmd": "remove", "name": "fx_apple_hand"}]
    assert match("draw a spaceship flying around me", ch) is None  # new idea -> Claude
    assert match("make the title bigger please", ch) is None
    assert {c["name"] for c in apply("remove everything")} == {"fx_orange_left_hand", "fx_horns_head",
                                                                "fx_big_eyes"}


def test_attach_follows_the_tracked_part():
    from vidai.live.trackers import Face, Hand

    ctx = Context(LiveBus(), 640, 360)
    ctx.tracks = _FakeTracks([Hand("Right", (0.25, 0.6), 0.12, (0.25, 0.4))],
                             Face((0.55, 0.2, 0.2, 0.3), ((0.6, 0.3), (0.7, 0.3)), (0.65, 0.38), (0.65, 0.45)))
    for to, (x, y) in {"hand": (160, 216), "head": (416, 72), "eyes": (416, 108)}.items():
        p = REGISTRY["attach"]("a", {"what": "🍎", "to": to})
        f = np.zeros((360, 640, 3), np.uint8)
        for i in range(5):
            f = p.process(f, i / 30, ctx)
        ys, xs = np.nonzero(f.max(axis=2))
        assert abs(xs.mean() - x) < 40 and abs(ys.mean() - y) < 50, (to, xs.mean(), ys.mean())
    p = REGISTRY["attach"]("b", {"what": "🍎", "to": "left_hand"})  # not visible -> nothing drawn
    assert p.process(np.zeros((360, 640, 3), np.uint8), 0, ctx).max() == 0


def test_request_uses_fast_path_before_claude(tmp_path):
    pl = LivePipeline(CaptureConfig(mode="test", out_height=360, mic=False), LiveConfig(stt=False), None,
                      session_dir=tmp_path)
    pl.ctx.need_tracking = lambda: None
    pl.start()
    assert pl.request("put a crown on my head")["handled"] == "fast"
    assert pl.chain.get("fx_crown_head") is not None
    assert pl.request("write a poem on the screen")["handled"] == "claude"
    time.sleep(0.3)
    pl.stop()
    kinds = [e["kind"] for e in pl.bus.history]
    assert kinds.count("claude") == 1


def test_wait_request_and_effect_library(tmp_path):
    import threading

    from vidai import service
    from vidai.session import SessionRecorder, create_session

    s = create_session({"title": "w"}, capture=CaptureConfig(mode="test", out_height=360, mic=False), root=tmp_path,
                       live={"stt": False})
    r = SessionRecorder(s)
    r.start()
    time.sleep(0.8)
    threading.Timer(0.5, lambda: r.pipe.request("add a spinning planet above me")).start()
    got = service.live_wait_request(s.dir, 0, timeout=10)
    assert got["requests"][0]["message"] == "add a spinning planet above me"
    code = ("from vidai.live.processors import LiveProcessor, register\n"
            "@register\nclass Planet(LiveProcessor):\n    def process(self, frame, t, ctx):\n        return frame\n")
    from vidai import actions

    actions.allow_code(s.dir)
    out = service.live_processor(s.dir, "planet", code, save_as="planet", description="spinning planet")
    assert out["saved_to_library"] == "planet"
    assert "planet" in service.live_effects()["library"]
    res = service.live_effect(s.dir, "planet", instance="planet2")
    assert res["results"][0]["kind"] == "ack"
    r.stop()
    assert service.live_wait_request(s.dir, got["next"], timeout=5)["state"] == "done"


def test_obvious_request_acts_without_waiting(tmp_path):
    pl = LivePipeline(CaptureConfig(mode="test", out_height=360, mic=False), LiveConfig(stt=False), None,
                      session_dir=tmp_path)
    pl.ctx.need_tracking = lambda: None
    pl.request_gap = 5.0  # would be slow if it waited
    pl.start()
    t0 = time.monotonic()
    pl.bus.publish("voice_command", {"command": "claude", "args": "add an apple in my hand", "text": "..."})
    while pl.chain.get("fx_apple_hand") is None and time.monotonic() - t0 < 3:
        time.sleep(0.01)
    took = time.monotonic() - t0
    pl.bus.publish("voice_command", {"command": "claude", "args": "add a text above my head saying", "text": "..."})
    time.sleep(0.3)
    assert pl._req is not None  # unfinished sentence: still listening
    pl.stop()
    assert took < 0.5


def test_preview_effects_carry_into_recording(tmp_path):
    """Effects set up in preview are still there when the recording starts (new pipeline)."""
    from vidai.session import SessionRecorder, create_session

    s = create_session({"title": "c"}, capture=CaptureConfig(mode="test", out_height=360, mic=False), root=tmp_path,
                       live={"stt": False, "speak": False})
    prev = LivePipeline(s.capture, s.live, None, session_dir=s.dir)
    prev.ctx.need_tracking = lambda: None
    prev.start()
    from vidai import actions

    actions.allow_code(s.dir)
    code = Path(s.dir) / "processors" / "tint.py"
    code.parent.mkdir(exist_ok=True)
    code.write_text("from vidai.live.processors import LiveProcessor, register\n"
                    "@register\nclass Tint(LiveProcessor):\n    def process(self, frame, t, ctx):\n        return frame\n")
    prev.command({"cmd": "add", "name": "fx_tint", "file": str(code), "params": {"k": 2}})
    prev.command({"cmd": "add", "name": "fx_crown_head", "type": "attach", "params": {"what": "👑", "to": "head"}})
    prev.command({"cmd": "text", "text": "temporary", "for": 2})  # temporary: not carried
    carry = prev.carry_specs()
    prev.stop()
    assert {c["name"] for c in carry} == {"fx_tint", "fx_crown_head"}
    r = SessionRecorder(s, carry=carry)
    r.pipe = None
    r.start()
    time.sleep(0.5)
    names = [p.name for p in r.pipe.chain.items]
    r.stop()
    assert "fx_tint" in names and "fx_crown_head" in names
    assert r.pipe.chain.get("fx_tint").params["k"] == 2
