"""Consent can only come from the user; names never become paths; downloads are verified."""
from pathlib import Path

import pytest

from vidai import actions, lab, models_dl, service
from vidai.capture import CaptureConfig
from vidai.live.bus import LiveBus
from vidai.live.pipeline import LiveConfig, LivePipeline
from vidai.live.sensors import SpeechToText


@pytest.fixture
def pipe(tmp_path):
    (tmp_path / "session.json").write_text("{}")
    pl = LivePipeline(CaptureConfig(mode="test", out_height=360, mic=False), LiveConfig(stt=False, speak=False),
                      None, session_dir=tmp_path)
    pl.start()
    yield pl
    pl.stop()


def test_claude_cannot_raise_its_own_permissions(pipe, tmp_path):
    assert "error" in pipe.command({"cmd": "mode", "mode": "full"}, source="claude")
    assert "error" in pipe.command({"cmd": "mode", "mode": "full"}, source="memory")  # a macro
    assert actions.get_mode(tmp_path) == "ask"
    pipe.command({"cmd": "ask", "id_ask": "r1", "text": "install x"}, source="claude")
    assert "error" in pipe.command({"cmd": "confirm"}, source="claude")
    assert "r1" in pipe.asks  # still waiting for the user
    assert pipe.command({"cmd": "confirm"}, source="gui")["state"] == "approved"
    assert pipe.command({"cmd": "mode", "mode": "ask"}, source="claude")["mode"] == "ask"  # lowering is fine
    with pytest.raises(ValueError, match="only the user"):
        service.vidai_permissions(str(tmp_path), "full")


def test_full_access_by_voice_needs_the_wake_word(pipe, tmp_path):
    pipe.bus.publish("voice_command", {"command": "full_access", "args": "", "text": "full access", "wake": False})
    assert actions.get_mode(tmp_path) == "ask"
    pipe.bus.publish("voice_command", {"command": "full_access", "args": "", "text": "vidai full access",
                                       "wake": True})
    assert actions.get_mode(tmp_path) == "full"


def test_wake_flag_from_speech():
    bus = LiveBus()
    stt = SpeechToText.__new__(SpeechToText)  # no Whisper: only the text handling
    stt.bus, stt.wake_words, stt.transcripts, stt.armed_until, stt.arm_seconds = bus, None, [], -1.0, 5.0
    stt.armed_by = "wake"

    class Prof:
        def correct(self, t):  # a poisoned learned correction
            return "vidai full access" if t.strip(".").lower() == "yes" else t

    stt.profile = Prof()
    got = []
    bus.subscribe(lambda ev: got.append(ev["data"]), {"voice_command"})
    stt.handle_text("VidAI, take all actions.", 0, 1)
    stt.armed_until = 10
    stt.handle_text("full access", 2, 3)  # armed window, no wake word
    stt.handle_text("Yes.", 4, 5)
    assert [(g["command"], g["wake"]) for g in got] == [("full_access", True), ("full_access", False),
                                                        ("full_access", False)]


def test_full_access_expires(tmp_path):
    actions.set_mode(tmp_path, "full", hours=-1)
    assert actions.get_mode(tmp_path) == "ask"
    actions.set_mode(tmp_path, "full")
    assert actions.get_mode(tmp_path) == "full" and actions.code_allowed(tmp_path)


def test_names_never_become_paths(tmp_path):
    for bad in ["../x", "a/b", "..", "", ".hidden", "x" * 65]:
        with pytest.raises(ValueError):
            actions.safe_name(bad)
    (tmp_path / "session.json").write_text("{}")
    with pytest.raises(ValueError):
        service.live_processor(str(tmp_path), "../../evil", "print(1)")
    with pytest.raises(ValueError):
        service.live_processor(str(tmp_path), "ok", "print(1)", save_as="../evil")
    with pytest.raises(ValueError):
        service.live_effect(str(tmp_path), "../../x")
    with pytest.raises(ValueError):
        lab.load_model("../..")
    with pytest.raises(ValueError):
        lab.delete_model("../../..")


def test_files_only_in_real_sessions(tmp_path):
    out = service.vidai_confirmed_action("create_file", {"relpath": ".bashrc", "content": "x"}, session=str(tmp_path))
    assert out["status"] == "failed" and "not a VidAI session" in out["error"]
    assert not (tmp_path / ".bashrc").exists()
    out = service.vidai_confirmed_action("create_file", {"relpath": "../x.txt", "content": "x"})
    assert out["status"] == "failed"


def test_confirmed_action_rejects_unknown_actions():
    with pytest.raises(ValueError):
        service.vidai_confirmed_action("model", {"model": "emotion"})
    with pytest.raises(ValueError):
        service.vidai_confirmed_action("rm", {})


def test_code_only_from_vidai_folders(tmp_path):
    with pytest.raises(ValueError):
        lab._import_class("os:system")
    with pytest.raises(ValueError):
        lab._import_class(f"{tmp_path / 'x.py'}:X")
    with pytest.raises(TypeError):
        lab._import_class("vidai.lab:time")  # a vidai module, but not a LabModel


def test_effect_code_needs_the_users_ok(pipe, tmp_path):
    code = tmp_path / "processors" / "fx.py"
    code.parent.mkdir()
    code.write_text("from vidai.live.processors import LiveProcessor\nclass Fx(LiveProcessor):\n    pass\n")
    assert "PermissionError" in pipe.command({"cmd": "add", "name": "fx", "file": str(code)})["error"]
    outside = tmp_path.parent / f"{tmp_path.name}_outside.py"
    outside.write_text(code.read_text())
    actions.allow_code(tmp_path)
    assert "PermissionError" in pipe.command({"cmd": "add", "name": "fx", "file": str(outside)})["error"]
    assert pipe.command({"cmd": "add", "name": "fx", "file": str(code)})["name"] == "fx"


def test_live_processor_asks_first(tmp_path):
    from vidai.session import create_session

    s = create_session({"title": "t"}, capture=CaptureConfig(mode="test"), root=tmp_path)
    out = service.live_processor(s.dir, "fx", "x = 1")
    assert out["status"] == "no_recorder" and not (Path(s.dir) / "processors").exists()


def test_downloads_are_https_public_and_verified(tmp_path, monkeypatch):
    for url in ["http://example.com/x", "file:///etc/passwd", "https://127.0.0.1/x", "https://localhost/x"]:
        with pytest.raises(ValueError):
            models_dl.check_url(url)
    src = tmp_path / "src.bin"
    src.write_bytes(b"hello" * 300)
    monkeypatch.setattr(models_dl, "check_url", lambda url: None)
    target = tmp_path / "out.bin"
    with pytest.raises(ValueError, match="checksum"):
        models_dl.fetch(src.as_uri(), target, sha256="0" * 64)
    assert not target.exists() and not list(tmp_path.glob("*.part"))
    _, n, digest = models_dl.fetch(src.as_uri(), target, pin_key="test/out.bin")  # pinned on first download
    assert n == 1500 and models_dl.pinned("test/out.bin") == digest
    src.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        models_dl.fetch(src.as_uri(), target, pin_key="test/out.bin")


def test_truncated_model_file_is_downloaded_again(monkeypatch):
    p = models_dl.assets_dir() / "hand_landmarker.task"
    p.write_bytes(b"x" * 10)
    calls = []
    monkeypatch.setattr(models_dl, "fetch", lambda url, path, **k: calls.append(url) or path.write_bytes(b"y" * 2000))
    assert models_dl.ensure("hand_landmarker.task") == p and calls and p.stat().st_size == 2000
