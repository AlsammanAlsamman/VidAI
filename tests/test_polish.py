"""Fixes from the live test: chatter is not a request, garbled speech is not a shortcut, emoji in messages."""
import time

from vidai.capture import CaptureConfig
from vidai.live.bus import LiveBus
from vidai.live.pipeline import LiveConfig, LivePipeline
from vidai.live.sensors import SpeechToText
from vidai.profile import Profile, plausible_shortcut


def _stt(bus):
    stt = SpeechToText.__new__(SpeechToText)  # no Whisper: only the text handling
    stt.bus, stt.wake_words, stt.transcripts, stt.profile = bus, None, [], None
    stt.armed_until, stt.armed_by, stt.arm_seconds = -1.0, "wake", 5.0
    return stt


def test_speech_says_why_vidai_was_listening():
    bus = LiveBus()
    got = []
    bus.subscribe(lambda ev: got.append(ev["data"]), {"voice_command"})
    stt = _stt(bus)
    stt.arm(10, "question")
    stt.handle_text("And thank you guys", 1, 2)
    stt.handle_text("VidAI", 3, 3.5)  # a bare wake word re-arms for anything
    stt.handle_text("add a crown on my head", 4, 5)
    assert [(g["args"], g["wake"], g["armed"]) for g in got] == [
        ("and thank you guys", False, "question"), ("add a crown on my head", False, "wake")]


def test_chatter_during_a_question_is_not_sent_to_claude(tmp_path):
    pl = LivePipeline(CaptureConfig(mode="test", out_height=360, mic=False), LiveConfig(stt=False, speak=False),
                      None, session_dir=tmp_path)
    pl.request_gap = 0.2
    pl.start()
    pl.command({"cmd": "question", "id_q": "q1", "text": "Rabbit ears for Adam?", "speak": False,
                "options": ["Rabbit ears on Adam", "Rabbit ears on me"]})
    pl.bus.publish("voice_command", {"command": "claude", "args": "and thank you guys", "text": "And thank you guys",
                                     "wake": False, "armed": "question"})
    pl.bus.publish("voice_command", {"command": "claude", "args": "rabbit ears on me", "text": "...",
                                     "wake": False, "armed": "question"})  # an answer still counts
    pl.bus.publish("voice_command", {"command": "claude", "args": "write a poem on the screen", "text": "...",
                                     "wake": True, "armed": None})  # "VidAI, ..." is always a request
    time.sleep(0.6)
    pl._flush_request(force=True)
    pl.stop()
    msgs = [e["data"]["message"] for e in pl.bus.history if e["kind"] == "claude"]
    assert msgs == ["write a poem on the screen"]
    assert any(e["kind"] == "answer" and e["data"]["answer"] == "Rabbit ears on me" for e in pl.bus.history)
    assert any(e["kind"] == "action" and e["data"].get("what") == "ignored_chatter" for e in pl.bus.history)


def test_garbled_speech_never_becomes_a_shortcut():
    for good in ["add numbers on my fingers", "remove the background", "and add an orange",
                 "at a title above my head called introduction", "add 1 2 3 4 5"]:
        assert plausible_shortcut(good), good
    for bad in ["to image to making", "and add add add rub it", "so just say vidai vidai vidai meet all the eyes",
                "ad add 1 to 3 for oom", "is end up real cool", "yeah adam",
                "improved the rabbit ears to be more realistic by using ai model"]:
        assert not plausible_shortcut(bad), bad
    p = Profile()
    p.add_macro("add an orange", [{"cmd": "add"}])
    p.add_macro("to image to making", [{"cmd": "add"}])
    p.add_macro("my own words", [{"cmd": "add"}], source="user")  # taught on purpose: kept
    assert p.clean_macros() == ["to image to making"]
    assert {m["phrase"] for m in p.macros()} == {"add an orange", "my own words"}


def test_emoji_in_window_messages():
    from PIL import Image, ImageDraw

    from vidai import gui

    assert gui._runs("New glasses 🕶️ ok") == [(False, "New glasses "), (True, "🕶"), (False, " ok")]
    img = Image.new("RGB", (300, 40), (0, 0, 0))
    d = ImageDraw.Draw(img, "RGBA")
    f = gui._pil_font(15, bold=False)
    gui._text(img, d, (4, 10), "👍", f, (255, 255, 255))
    px = img.getpixel((12, 19))
    assert px != (0, 0, 0) and len(set(px)) > 1  # a coloured emoji, not an empty box
