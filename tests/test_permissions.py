"""VidAI asks 'Master, I need to ...' and waits for confirm / deny, or does everything in full-access mode."""
import threading
import time
from pathlib import Path

import pytest

from vidai import actions, service
from vidai.capture import CaptureConfig
from vidai.live.bus import read_events
from vidai.live.sensors import parse_command
from vidai.session import SessionRecorder, create_session


@pytest.fixture
def rec(tmp_path, monkeypatch):
    s = create_session({"title": "perm"}, capture=CaptureConfig(mode="test", out_height=360, mic=False),
                       root=tmp_path, live={"stt": False, "speak": False})
    r = SessionRecorder(s)
    r.start()
    monkeypatch.setattr(service, "_recorder_alive", lambda session: True)
    time.sleep(0.5)
    yield s, r
    r.stop()


def _answer_when_asked(r, s, what, by="voice"):
    def run():
        for _ in range(100):
            evs = read_events(Path(s.dir) / "live.jsonl", 0, ["action"], 10 ** 6)
            if any(e["data"].get("what") == "asking" for e in evs):
                r.pipe.command({"cmd": what, "by": by}, source=by)
                return
            time.sleep(0.05)
    threading.Thread(target=run, daemon=True).start()


def test_ask_then_confirm_runs_the_action(rec):
    s, r = rec
    _answer_when_asked(r, s, "confirm")
    out = service.vidai_create_file("effects/hello.py", "print('hi')", session=s.dir, reason="test")
    assert out["status"] == "done" and Path(out["path"]).read_text() == "print('hi')"
    said = [e["data"]["text"] for e in read_events(Path(s.dir) / "live.jsonl", 0, ["action"], 10 ** 6)
            if e["data"].get("what") == "vidai_said"]
    assert said[0].startswith("Master, I need to create the file effects/hello.py")


def test_ask_then_deny_does_nothing(rec):
    s, r = rec
    _answer_when_asked(r, s, "deny")
    out = service.vidai_create_file("nope.txt", "x", session=s.dir)
    assert out["status"] == "denied" and not (Path(s.dir) / "nope.txt").exists()


def test_full_access_skips_the_question(rec):
    s, r = rec
    r.pipe.command({"cmd": "mode", "mode": "full"}, source="voice")  # "VidAI, take all actions"
    t0 = time.monotonic()
    out = service.vidai_create_file("fast.txt", "ok", session=s.dir)
    assert out["status"] == "done" and time.monotonic() - t0 < 1.0
    assert not any(e["data"].get("what") == "asking"
                   for e in read_events(Path(s.dir) / "live.jsonl", 0, ["action"], 10 ** 6))
    r.pipe.command({"cmd": "mode", "mode": "ask"}, source="voice")  # "VidAI, ask me first"
    assert actions.get_mode(s.dir) == "ask"


def test_without_recorder_claude_asks_in_chat(tmp_path):
    out = service.vidai_create_file("a.txt", "x", session=None)
    assert out["status"] == "needs_confirmation" and "Allow?" in out["ask_user"]
    out = service.vidai_create_file("a.txt", "x", session=None, confirmed=True)
    assert out["status"] == "done"
    log = (actions.home() / "actions.log").read_text()
    assert "create_file" in log


def test_action_limits():
    with pytest.raises(ValueError):
        actions.install(["-r", "evil.txt"])
    with pytest.raises(ValueError):
        actions.install(["numpy; rm -rf /"])
    with pytest.raises(ValueError):
        actions.create_file("../../outside.txt", "x")
    with pytest.raises(ValueError):
        actions.download("file:///etc/passwd")


def test_voice_phrases():
    assert parse_command("VidAI, take all actions")["command"] == "full_access"
    assert parse_command("VidAI ask me first")["command"] == "ask_first"
    assert parse_command("VidAI yes")["command"] == "confirm"
    assert parse_command("VidAI deny")["command"] == "deny"
