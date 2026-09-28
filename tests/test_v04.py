"""v0.4: undo/redo, help, notify, Claude questions, performance governor, tiny.en default, typed requests."""
import threading
import time
from pathlib import Path

import pytest

from vidai.capture import CaptureConfig
from vidai.live.bus import read_events
from vidai.live.pipeline import LiveConfig, LivePipeline, _match_option
from vidai.live.sensors import parse_command

DATA = Path(__file__).parent / "data"


def _pipe(tmp_path, **live):
    pl = LivePipeline(CaptureConfig(mode="test", out_height=360, mic=False),
                      LiveConfig(stt=False, speak=False, **live), None, session_dir=tmp_path)
    pl.ctx.need_tracking = lambda: None
    return pl


def _names(pl):
    return [p.name for p in pl.chain.items if not p.name.startswith("_")]


def test_new_voice_phrases():
    assert parse_command("VidAI undo")["command"] == "undo"
    assert parse_command("VidAI, go back")["command"] == "undo"
    assert parse_command("VidAI redo")["command"] == "redo"  # not the older "mistake" meaning
    assert parse_command("VidAI help")["command"] == "help"
    assert parse_command("VidAI, what can I say")["command"] == "help"
    assert parse_command("VidAI lighter")["command"] == "lighter"
    assert parse_command("VidAI again")["command"] == "mistake"


def test_undo_redo_whole_requests(tmp_path):
    pl = _pipe(tmp_path)
    pl.start()
    pl.request("add an apple in my hand")
    pl.request("put a crown on my head")
    assert _names(pl) == ["fx_apple_hand", "fx_crown_head"]
    pl.command({"cmd": "undo"}, source="voice")
    assert _names(pl) == ["fx_apple_hand"]
    pl.command({"cmd": "redo"}, source="voice")
    assert _names(pl) == ["fx_apple_hand", "fx_crown_head"]
    # one request that changes two effects is undone as one step
    pl.request("put an orange on my other hand")
    assert pl.chain.get("fx_apple_hand").params["to"] == "right_hand"
    pl.command({"cmd": "undo"}, source="voice")
    assert pl.chain.get("fx_orange_left_hand") is None and pl.chain.get("fx_apple_hand").params["to"] == "hand"
    # undoing a removal brings the effect back with its settings
    pl.request("bigger apple")
    pl.request("remove the apple")
    assert pl.chain.get("fx_apple_hand") is None
    pl.command({"cmd": "undo"}, source="voice")
    assert pl.chain.get("fx_apple_hand").params["scale"] == pytest.approx(1.4)
    for _ in range(6):
        pl.command({"cmd": "undo"}, source="voice")
    assert _names(pl) == []
    assert pl.undo() == {"undone": 0}
    pl.stop()
    notes = [e["data"]["text"] for e in pl.bus.history if e["kind"] == "notify"]
    assert "↶ Undone" in notes and "Nothing to undo" in notes


def test_new_request_clears_redo(tmp_path):
    pl = _pipe(tmp_path)
    pl.start()
    pl.request("add an apple in my hand")
    pl.undo()
    pl.request("put a crown on my head")
    assert pl.redo() == {"redone": 0}
    pl.stop()


def test_help_notify_and_summary(tmp_path):
    pl = _pipe(tmp_path)
    pl.start()
    pl.command({"cmd": "help"}, source="voice")
    pl.command({"cmd": "notify", "text": "Hair made bigger — say VidAI undo"}, source="claude")
    pl.request("add an apple in my hand")
    pl.request("write a poem about DNA")  # goes to Claude
    s = pl.summary()
    pl.stop()
    help_ev = [e["data"] for e in pl.bus.history if e["kind"] == "help"]
    assert help_ev and any("undo" in line for line in help_ev[0]["lines"])
    assert any(e["kind"] == "notify" and "Hair" in e["data"]["text"] for e in pl.bus.history)
    assert s["requests"] == 2 and s["instant"] == 1 and s["by_claude"] == 1 and "fx_apple_hand" in s["effects"]


def test_match_option():
    opts = ["blur", "purple", "beach"]
    assert _match_option("the second one", opts) == "purple"
    assert _match_option("option 3", opts) == "beach"
    assert _match_option("Blur please", opts) == "blur"
    assert _match_option("purpel", opts) == "purple"  # typo / mishearing
    assert _match_option("something else entirely", opts) == "something else entirely"
    assert _match_option("anything", []) == "anything"


def test_question_answered_by_voice_or_typing(tmp_path):
    pl = _pipe(tmp_path)
    pl.start()
    pl.command({"cmd": "question", "id_q": "q1", "text": "Which background?", "options": ["blur", "purple"],
                "speak": False}, source="claude")
    assert "q1" in pl.questions
    # "VidAI, the first one" while a question is open is an answer, not a new request to Claude
    pl.bus.publish("voice_command", {"command": "claude", "args": "the first one", "text": "VidAI the first one"})
    assert not pl.questions
    ans = [e["data"] for e in pl.bus.history if e["kind"] == "answer"]
    assert ans and ans[0]["answer"] == "blur"
    assert not any(e["kind"] == "claude" for e in pl.bus.history)
    pl.command({"cmd": "question", "id_q": "q2", "text": "Keep it?", "options": ["yes", "no"], "speak": False})
    pl.command({"cmd": "answer", "question": "q2", "text": "no", "by": "typed"}, source="gui")
    pl.stop()
    assert [e["data"]["answer"] for e in pl.bus.history if e["kind"] == "answer"] == ["blur", "no"]


def test_live_ask_user_and_notify_tools(tmp_path, monkeypatch):
    from vidai import service
    from vidai.session import SessionRecorder, create_session

    s = create_session({"title": "q"}, capture=CaptureConfig(mode="test", out_height=360, mic=False), root=tmp_path,
                       live={"stt": False, "speak": False})
    r = SessionRecorder(s)
    r.start()
    time.sleep(0.5)

    def answer_soon():
        for _ in range(100):
            if r.pipe.questions:
                r.pipe.command({"cmd": "answer", "text": "beach", "by": "button"}, source="gui")
                return
            time.sleep(0.05)

    threading.Thread(target=answer_soon, daemon=True).start()
    got = service.live_ask_user(s.dir, "Which background?", ["blur", "purple", "beach"], timeout=10, speak=False)
    assert got == {"answer": "beach", "said": "beach", "by": "button"}
    out = service.live_notify(s.dir, "Done — say VidAI undo if you don't like it")
    assert out["results"][0]["kind"] == "ack"
    r.stop()
    assert (Path(s.dir) / "summary.json").exists()


class _Tracks:
    max_hz = 30.0

    def feed(self, frame):
        pass

    def close(self):
        pass


def test_performance_governor_levels(tmp_path):
    pl = _pipe(tmp_path)
    pl.ctx.tracks = _Tracks()
    pl.start()
    pl._t0_wall = time.monotonic() - 10
    pl.lag_frames, pl.frame_ms_ema = 20, 30.0  # falling behind
    pl._level_since -= 5
    pl._govern()
    assert pl.level == 1 and pl.ctx.tracks.max_hz == 15
    pl._level_since -= 5
    pl._govern()
    assert pl.level == 2 and pl.ctx.tracks.max_hz == 8
    pl.lag_frames, pl.frame_ms_ema = 0, 2.0  # relaxed again
    pl._level_since -= 10
    pl._govern()
    assert pl.level == 1
    pl.command({"cmd": "lighter"}, source="voice")
    assert pl.level == 2
    pl.stop()
    warn = [e["data"] for e in pl.bus.history if e["kind"] == "warning"]
    assert warn and all("Performance" in w["text"] for w in warn)


def test_governor_turns_off_worst_effect_only_when_needed(tmp_path):
    from vidai.live.processors import LiveProcessor

    class Heavy(LiveProcessor):
        def process(self, frame, t, ctx):
            return frame

    pl = _pipe(tmp_path)
    pl.start()
    light, heavy = Heavy("fx_light"), Heavy("fx_heavy")
    for p, ms in ((light, 1.0), (heavy, 25.0)):
        p.calls, p.total_ms = 10, ms * 10
        pl.chain.add(p)
    pl._t0_wall = time.monotonic() - 30
    pl._set_level(2)
    pl._level_since -= 10
    pl.lag_frames, pl.frame_ms_ema = 40, 40.0
    pl._pressure_since = time.monotonic() - 9
    pl._govern()
    pl.stop()
    assert heavy.enabled is False and light.enabled is True
    assert any("fx_heavy" in e["data"].get("text", "") for e in pl.bus.history if e["kind"] == "warning")


def test_typed_request_goes_the_same_way_as_voice(tmp_path):
    pl = _pipe(tmp_path)
    pl.start()
    assert pl.request("add an apple in my hand", source="typed")["handled"] == "fast"
    assert pl.request("draw my name in gold letters", source="typed")["handled"] == "claude"
    pl.stop()
    claude = [e["data"] for e in pl.bus.history if e["kind"] == "claude"]
    assert claude[0]["source"] == "typed"


@pytest.mark.slow
def test_tiny_en_is_the_default_and_hears_commands(tmp_path):
    cfg = CaptureConfig(mode="test", out_height=360, mic_source=str(DATA / "speech_en.wav"))
    live = LiveConfig(stt=True, stt_language="en", speak=False)
    assert live.stt_model == "" and live.utterance_gap == pytest.approx(0.35)
    pl = LivePipeline(cfg, live, tmp_path / "v.mkv", session_dir=tmp_path)
    pl.ctx.need_tracking = lambda: None
    pl.start()
    time.sleep(18.5)
    pl.stop()
    ev = read_events(tmp_path / "live.jsonl", limit=10 ** 6)
    ready = [e["data"] for e in ev if e["kind"] == "action" and e["data"].get("what") == "stt_ready"]
    assert ready and ready[0]["model"] == "tiny.en"
    cmds = [e["data"]["command"] for e in ev if e["kind"] == "voice_command"]
    assert "section" in cmds and "zoom_in" in cmds


# ---------------- fixes from the 2026-09-28 live test ----------------
def test_vidai_does_not_answer_itself(tmp_path):
    pl = _pipe(tmp_path)
    pl.start()
    got = []
    pl.stt = type("S", (), {"submit": lambda self, *a: got.append(a), "enabled": True, "armed_until": -1})()
    pl.speaking_until = pl.clock() + 5  # VidAI is talking
    pl._on_utterance(None, pl.clock(), pl.clock() + 1)
    assert got == [] and any(e["kind"] == "action" and e["data"]["what"] == "ignored_own_voice"
                             for e in pl.bus.history)
    pl.speaking_until = -1
    pl._on_utterance(None, pl.clock(), pl.clock() + 1)
    pl.stt = None
    pl.stop()
    assert len(got) == 1


def test_question_expires_and_can_be_cancelled(tmp_path):
    pl = _pipe(tmp_path)
    pl.start()
    pl.command({"cmd": "question", "id_q": "old", "text": "?", "options": ["a", "b"], "speak": False})
    pl.questions["old"]["t"] -= 100  # nobody answered for 100 s
    time.sleep(0.4)  # the control loop closes it
    assert "old" not in pl.questions
    pl.command({"cmd": "question", "id_q": "q2", "text": "?", "options": ["a", "b"], "speak": False})
    assert pl.command({"cmd": "cancel_question", "question": "q2"})["cancelled"] is True
    pl.stop()
    cancelled = [e["data"] for e in pl.bus.history if e["kind"] == "answer" and e["data"].get("cancelled")]
    assert {c["question"] for c in cancelled} == {"old", "q2"}


def test_a_new_request_is_not_taken_as_an_answer(tmp_path):
    pl = _pipe(tmp_path)
    pl.start()
    pl.command({"cmd": "question", "id_q": "q", "text": "What instead?", "options": ["Cover with sunglasses",
                                                                                     "Cartoon eyes", "Leave them"],
                "speak": False})
    pl.bus.publish("voice_command", {"command": "claude", "args": "add an orange in my hand", "text": "..."})
    time.sleep(1.5)
    assert "q" in pl.questions  # still open
    assert pl.chain.get("fx_orange_hand") is not None  # and the request was done
    pl.bus.publish("voice_command", {"command": "claude", "args": "cartoon eyes", "text": "..."})
    pl.stop()
    assert [e["data"]["answer"] for e in pl.bus.history if e["kind"] == "answer"] == ["Cartoon eyes"]


def test_ask_user_timeout_closes_the_question(tmp_path):
    from vidai import service
    from vidai.session import SessionRecorder, create_session

    s = create_session({"title": "q"}, capture=CaptureConfig(mode="test", out_height=360, mic=False), root=tmp_path,
                       live={"stt": False, "speak": False})
    r = SessionRecorder(s)
    r.start()
    time.sleep(0.5)
    assert service.live_ask_user(s.dir, "Anyone?", ["yes", "no"], timeout=1, speak=False) == {"answer": None,
                                                                                             "timeout": True}
    time.sleep(0.5)
    open_questions = dict(r.pipe.questions)
    r.stop()
    assert open_questions == {}


def test_more_fast_path_words(tmp_path):
    from vidai.live.intents import match
    from vidai.live.processors import REGISTRY, Context, LiveProcessor, ProcessorChain
    from vidai.live.bus import LiveBus

    ch = ProcessorChain(Context(LiveBus(), 640, 360))
    ch.ctx.need_tracking = lambda: None
    ch.add(REGISTRY["attach"]("fx_orange_hand", {"what": "🍊", "to": "hand"}))
    ch.add(REGISTRY["attach"]("fx_text_above_head", {"what": "text:Introduction", "to": "above_head"}))

    class Numbers(LiveProcessor):
        pass

    ch.add(Numbers("fx_finger_numbers"))
    assert match("removed that orange", ch) == [{"cmd": "remove", "name": "fx_orange_hand"}]
    assert match("remove these numbers", ch) == [{"cmd": "remove", "name": "fx_finger_numbers"}]
    assert match("remove introduction", ch) == [{"cmd": "remove", "name": "fx_text_above_head"}]
    assert match("remove the spaceship", ch) is None
    assert match("increase", ch)[0]["cmd"] == "set" and match("reduce it", ch)[0]["params"]["scale"] < 1
    assert parse_command("VidAI starts recording")["command"] == "record"
    assert parse_command("VidAI begin recording")["command"] == "record"


def test_claude_can_start_the_recording(tmp_path):
    started = []
    pl = LivePipeline(CaptureConfig(mode="test", out_height=360, mic=False), LiveConfig(stt=False, speak=False),
                      None, session_dir=tmp_path, on_start_request=lambda: started.append(1))
    pl.start()
    assert pl.command({"cmd": "record"}, source="claude") == {"recording": "starting"}
    pl.stop()
    assert started == [1]


def test_request_waits_for_a_long_correction(tmp_path):
    """'add 1 2 3 4 above each of my fingers' then a 3.4 s correction 'add 1, 2, 3, 4, 5' -> one request."""
    pl = _pipe(tmp_path)
    pl.request_gap = 0.3
    pl.start()
    pl.bus.publish("voice_command", {"command": "claude", "args": "add 1 2 3 4 above each of my fingers", "text": ""})
    pl.bus.publish("speech_start", {})
    time.sleep(4.0)  # still talking (longer than the old 3 s limit)
    assert not any(e["kind"] == "claude" for e in pl.bus.history)
    pl.bus.publish("speech_end", {})
    time.sleep(0.4)
    pl.bus.publish("transcript", {"text": "Add 1, 2, 3, 4, 5", "is_command": False})
    time.sleep(0.8)
    msgs = [e["data"]["message"] for e in pl.bus.history if e["kind"] == "claude"]
    pl.stop()
    assert len(msgs) == 1 and msgs[0].endswith("Add 1, 2, 3, 4, 5")  # the correction arrives complete


def test_remove_only_answers_do_not_become_shortcuts(tmp_path):
    from vidai.profile import Profile

    pl = _pipe(tmp_path)
    pl.start()
    pl.command({"cmd": "add", "name": "fx_x", "type": "text", "params": {"text": "x"}}, source="config")
    pl.request("thank you please take the x away now")  # not understood -> Claude
    pl.command({"cmd": "remove", "name": "fx_x"}, source="claude")
    pl.command({"cmd": "done"}, source="claude")
    for p in pl._probation:
        p["t_done"] -= 30
    pl._check_probation()
    pl.stop()
    assert Profile().find_macro("thank you please take the x away now") is None


def test_only_self_contained_answers_become_shortcuts(tmp_path):
    """'increase' / 'order these' change what is on screen now; replayed later they would be wrong."""
    from vidai.profile import Profile

    pl = _pipe(tmp_path)
    pl.start()
    pl.command({"cmd": "add", "name": "fx_t", "type": "text", "params": {"text": "x"}}, source="config")
    pl.request("increase it a lot please")
    pl.command({"cmd": "set", "name": "fx_t", "params": {"size": 0.1}}, source="claude")
    pl.command({"cmd": "done"}, source="claude")
    pl.request("write my name in gold letters")
    pl.command({"cmd": "text", "name": "rocket", "text": "rocket"}, source="claude")
    pl.command({"cmd": "done"}, source="claude")
    for p in pl._probation:
        p["t_done"] -= 30
    pl._check_probation()
    pl.stop()
    assert Profile().find_macro("increase it a lot please") is None
    assert Profile().find_macro("write my name in gold letters") is not None


# ---------------- "VidAI talk": VidAI answers out loud, in the video ----------------
def test_talk_mode_sends_the_question_to_claude_for_a_voice_answer(tmp_path):
    pl = _pipe(tmp_path)
    pl.request_gap = 0.3
    pl.start()
    assert parse_command("VidAI talk")["command"] == "talk"
    assert parse_command("VidAI, I have a question")["command"] == "talk"
    pl.bus.publish("voice_command", {"command": "talk", "args": "", "text": "VidAI talk"})
    said = [e["data"]["text"] for e in pl.bus.history if e["kind"] == "action" and e["data"]["what"] == "vidai_said"]
    assert said == ["How can I help you?"]
    # the question contains a sticker phrase, but it is a question: it must go to Claude, not the fast path
    pl.bus.publish("voice_command", {"command": "claude", "args": "why do we put an apple in my hand", "text": ""})
    time.sleep(1.0)
    pl.stop()
    reqs = [e["data"] for e in pl.bus.history if e["kind"] == "claude"]
    assert reqs == [{"message": "why do we put an apple in my hand", "source": "talk", "reply": "voice"}]
    assert pl.chain.get("fx_apple_hand") is None


def test_mux_mixes_vidai_voice_into_the_audio(tmp_path):
    import numpy as np

    from vidai import ffmpeg
    from vidai.live.pipeline import mux_parts

    v, a, clip = tmp_path / "v.mkv", tmp_path / "a.mka", tmp_path / "voice.wav"
    ffmpeg.run(["-f", "lavfi", "-i", "color=c=black:s=320x240:r=30:d=6", "-c:v", "libx264", "-preset", "ultrafast",
                str(v)])
    ffmpeg.run(["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", "6", "-c:a", "aac", str(a)])  # quiet mic
    ffmpeg.run(["-f", "lavfi", "-i", "sine=f=500:d=1.5:r=22050", str(clip)])  # stands in for VidAI's voice
    out = mux_parts(v, a, tmp_path / "final.mkv", [(3.0, str(clip), 1.5)])
    pcm = ffmpeg.read_audio(str(out), 16000)
    rms = lambda t0, t1: float(np.sqrt(np.mean(pcm[int(t0 * 16000):int(t1 * 16000)] ** 2)))
    assert rms(3.2, 4.3) > 0.05 and rms(0.5, 2.5) < 0.01  # the voice is there, at its time
    assert not v.exists() and ffmpeg.probe(str(out)).has_video


def test_say_while_recording_puts_the_voice_in_the_video(tmp_path, monkeypatch):
    from vidai import service
    from vidai.live import voice as voice_mod
    from vidai.session import SessionRecorder, create_session

    if voice_mod.Voice().engine != "piper":
        pytest.skip("piper not installed")
    monkeypatch.setattr(voice_mod.Voice, "play", staticmethod(lambda path: time.sleep(0.2)))  # no speakers in tests
    s = create_session({"title": "talk"}, capture=CaptureConfig(mode="test", out_height=360, mic=False),
                       root=tmp_path, live={"stt": False, "speak": True})
    r = SessionRecorder(s)
    r.start()
    time.sleep(1.0)
    out = service.live_say(s.dir, "DNA carries the genetic instructions of living things.", subtitle=True)
    assert out["results"][0]["said"] is True and out["results"][0]["engine"] == "piper"
    time.sleep(2.0)
    clips = list(r.pipe.voice_clips)
    r.stop()
    assert len(clips) == 1 and clips[0][2] > 1.0
    from vidai import ffmpeg

    info = ffmpeg.probe(str(s.video_path))
    assert info.has_audio  # mic was off: the audio track is VidAI's voice
    import numpy as np

    pcm = ffmpeg.read_audio(str(s.video_path), 16000)
    t0, dur = clips[0][0], clips[0][2]
    seg = pcm[int((t0 + 0.2) * 16000):int((t0 + dur - 0.2) * 16000)]
    assert float(np.sqrt(np.mean(seg ** 2))) > 0.01


def test_self_hearing_guard_uses_the_audio_clock(tmp_path):
    """Speech times come from the audio clock; if the video clock lags behind, VidAI must still ignore itself."""
    pl = _pipe(tmp_path)
    pl.start()
    got = []
    pl.stt = type("S", (), {"submit": lambda self, *a: got.append(a), "enabled": True, "armed_until": -1})()
    pl.audio.written = 16000 * 50  # audio clock at 50 s while the video clock is ~0 s
    assert pl._audio_now() == pytest.approx(50.0)
    pl.speaking_until = pl._audio_now() + 3  # VidAI talking now (audio time)
    pl._on_utterance(None, 50.5, 51.5)  # its own voice, heard at audio time 50.5
    pl.command({"cmd": "talk"}, source="voice")
    armed = pl.stt.armed_until
    pl.stt = None
    pl.stop()
    assert got == [] and armed > 50  # listening window is on the audio clock too
