"""VidAI learns from mistakes and from Claude's solutions, and remembers them in the next video."""
import time

import pytest

from vidai.capture import CaptureConfig
from vidai.live.pipeline import LiveConfig, LivePipeline
from vidai.profile import Profile


def _pipe(tmp_path, name="s"):
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    pl = LivePipeline(CaptureConfig(mode="test", out_height=360, mic=False), LiveConfig(stt=False, speak=False),
                      None, session_dir=d)
    pl.ctx.need_tracking = lambda: None
    return pl


def test_profile_basics(tmp_path):
    p = Profile(tmp_path)
    assert p.learn_correction("promote the apple", "remove the apple")
    assert p.correct("VidAI, promote the apple!") == "vidai remove the apple"
    assert p.correct("hello world") == "hello world"
    p.add_macro("draw a spaceship", [{"cmd": "text", "text": "🚀"}])
    assert p.find_macro("Draw a spaceship.")["phrase"] == "draw a spaceship"
    assert p.find_macro("remove everything") is None
    p.add_lesson("The user wants captions off.")
    p.add_lesson("The user wants captions off!")  # duplicates are not stored twice
    assert len(p.lessons()) == 1
    assert p.nudge_pref("x", 2.0) == 2.0 and p.nudge_pref("x", 1.0) == pytest.approx(1.4)


def test_misheard_request_is_corrected_next_time(tmp_path):
    pl = _pipe(tmp_path)
    pl.start()
    pl.request("add an apple in my head")  # misheard: the apple lands on the head
    pl.command({"cmd": "remove", "name": "fx_apple_head"}, source="voice")  # user removes it right away
    pl.request("add an apple in my hand")  # and says what they meant
    assert any(l["kind"] == "vocabulary" for l in pl.learned)
    pl.stop()
    pl2 = _pipe(tmp_path, "next_video")  # a new video: the same mishearing is fixed before it is understood
    pl2.start()
    out = pl2.request("add an apple in my head")
    pl2.stop()
    assert out["commands"][0]["params"]["to"] == "hand"
    assert any("removed it right away" in l["lesson"] for l in Profile().lessons())


def test_claude_solution_becomes_instant_macro(tmp_path):
    pl = _pipe(tmp_path)
    pl.start()
    assert pl.request("draw a spaceship above me")["handled"] == "claude"
    pl.command({"cmd": "text", "name": "ship", "text": "🚀 to the moon", "position": "top-center"}, source="claude")
    pl.command({"cmd": "done"}, source="claude")
    assert pl._probation
    pl._probation[0]["t_done"] -= 30  # kept for 20 s -> remembered
    pl._check_probation()
    pl.stop()
    assert Profile().find_macro("draw a spaceship above me")
    pl2 = _pipe(tmp_path, "next")
    pl2.start()
    out = pl2.request("draw a spaceship above me")
    assert out["handled"] == "memory" and pl2.chain.get("ship") is not None  # no Claude round trip
    pl2.command({"cmd": "remove", "name": "ship"}, source="voice")  # ...but the user did not want it this time
    pl2.stop()
    assert Profile().find_macro("draw a spaceship above me") is None  # a bad shortcut is forgotten


def test_preferences_carry_over(tmp_path):
    pl = _pipe(tmp_path)
    pl.start()
    pl.request("add an apple in my hand")
    pl.request("bigger apple")
    pl.request("put an orange on my other hand")
    pl.request("swap hands")
    pl.stop()
    prof = Profile()
    assert prof.pref("scale:🍎") == pytest.approx(1.4) and prof.pref("hands_swapped") is True
    pl2 = _pipe(tmp_path, "next")
    pl2.start()
    a = pl2.request("add an apple in my hand")["commands"][0]["params"]
    o = pl2.request("add an orange in my right hand")["commands"][0]["params"]
    pl2.stop()
    assert a["scale"] == pytest.approx(1.4) and o["to"] == "left_hand"


def test_session_history_and_brief_defaults(tmp_path):
    from vidai import service
    from vidai.session import SessionRecorder, create_session

    s = create_session({"title": "t", "language": "en", "audience": "Arab biology students", "style": "screencast"},
                       capture=CaptureConfig(mode="test", out_height=360, mic=False), root=tmp_path,
                       live={"stt": False, "speak": False})
    r = SessionRecorder(s)
    r.start()
    time.sleep(0.5)
    r.pipe.request("put a crown on my head")
    time.sleep(0.5)
    r.stop()
    prof = service.vidai_profile()
    assert prof["recent_sessions"][-1]["requests"] == ["put a crown on my head"]
    assert Profile().pref("brief_defaults")["audience"] == "Arab biology students"
    out = service.studio_start({"title": "next"}, root=str(tmp_path), open_gui=False)
    from vidai.session import Session

    assert Session.load(out["session"]).brief.audience == "Arab biology students"  # not asked again
    service.vidai_learn("lesson", {"lesson": "Keep effects small on camera videos."})
    assert "Keep effects small on camera videos." in service.vidai_profile()["lessons"]
