"""Live pipeline and editing robustness: processor lifecycle, undo, control handoff, plan locking, errors."""
import threading
import time
from pathlib import Path

import pytest

from vidai import service
from vidai.capture import CaptureConfig
from vidai.live.bus import LiveBus
from vidai.live.pipeline import LiveConfig, LivePipeline
from vidai.live.processors import REGISTRY, LiveProcessor


def _pipe(tmp_path):
    return LivePipeline(CaptureConfig(mode="test", out_height=360, mic=False), LiveConfig(stt=False, speak=False),
                        None, session_dir=tmp_path)


class _Closing(LiveProcessor):
    listens = {"marker"}
    closed_n = 0

    def close(self):
        super().close()
        _Closing.closed_n += 1

    def on_event(self, ev, ctx):
        self.seen = getattr(self, "seen", 0) + 1


def test_removed_processors_are_closed_and_unsubscribed(tmp_path, monkeypatch):
    monkeypatch.setitem(REGISTRY, "closing", _Closing)
    _Closing.closed_n = 0
    pl = _pipe(tmp_path)
    pl.start()
    n_subs = len(pl.bus._subs)
    pl.command({"cmd": "add", "name": "c", "type": "closing"})
    p = pl.chain.get("c")
    assert len(pl.bus._subs) == n_subs + 1
    pl.command({"cmd": "add", "name": "c", "type": "closing"})  # replaced -> the old one is closed
    assert _Closing.closed_n == 1 and p.closed
    pl.command({"cmd": "remove", "name": "c"})
    assert _Closing.closed_n == 2 and len(pl.bus._subs) == n_subs
    pl.command({"cmd": "add", "name": "d", "type": "closing"})
    pl.stop()
    assert _Closing.closed_n == 3


def test_undo_removes_exactly_the_unnamed_effect(tmp_path):
    pl = _pipe(tmp_path)
    pl.start()
    pl._txn += 1
    name = pl.command({"cmd": "text", "text": "hello"})["name"]
    pl._txn += 1
    pl.command({"cmd": "add", "name": "later", "type": "text", "params": {"text": "x"}}, source="rule:x")
    pl.history.pop()  # only the first request is undone below
    pl.undo(quiet=True)
    pl.stop()
    assert pl.chain.get(name) is None and pl.chain.get("later") is not None


def test_concurrent_commands_keep_every_effect(tmp_path):
    pl = _pipe(tmp_path)
    pl.start()
    ths = [threading.Thread(target=lambda i=i: pl.command({"cmd": "text", "name": f"t{i}", "text": str(i)}))
           for i in range(40)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    pl.stop()
    assert {f"t{i}" for i in range(40)} <= {p.name for p in pl.chain.items}


def test_commands_between_preview_and_recording_are_not_lost(tmp_path):
    ctl = tmp_path / "control.jsonl"
    a = _pipe(tmp_path)
    a.start()
    time.sleep(0.3)
    a.stop()
    with open(ctl, "a") as f:  # Claude writes while no pipeline is running
        f.write('{"cmd": "text", "name": "late", "text": "hi"}\n{"cmd": "text", "na')  # + a half-written line
    b = _pipe(tmp_path)
    b.start()
    deadline = time.monotonic() + 3
    while b.chain.get("late") is None and time.monotonic() < deadline:
        time.sleep(0.05)
    with open(ctl, "a") as f:
        f.write('me": "rest", "text": "ok"}\n')
    while b.chain.get("rest") is None and time.monotonic() < deadline:
        time.sleep(0.05)
    b.stop()
    assert b.chain.get("late") is not None and b.chain.get("rest") is not None
    assert not any(e["kind"] == "error" and e["data"].get("where") == "control" for e in b.bus.history)


def test_frame_loop_survives_a_broken_preview_callback(tmp_path):
    pl = _pipe(tmp_path)
    pl.on_frame = lambda f: 1 / 0
    pl.start()
    time.sleep(1.0)
    frames = pl.frames
    time.sleep(0.5)
    pl.stop()
    assert pl.frames > frames  # still running
    errs = [e for e in pl.bus.history if e["kind"] == "error" and e["data"].get("where") == "preview"]
    assert 1 <= len(errs) <= 3  # reported, rate-limited


def test_failed_start_cleans_up(tmp_path, monkeypatch):
    import vidai.live.pipeline as P

    def boom(*a, **k):
        raise RuntimeError("no camera")

    monkeypatch.setattr(P, "capture_inputs", boom)
    pl = LivePipeline(CaptureConfig(mode="test", out_height=360, mic=False), LiveConfig(stt=False, speak=False),
                      tmp_path / "v.mkv", session_dir=tmp_path)
    with pytest.raises(RuntimeError, match="no camera"):
        pl.start()
    assert pl.stopped and not Path(pl._tmp).exists()


def test_bus_unsubscribe():
    bus = LiveBus()
    got = []
    fn = bus.subscribe(lambda ev: got.append(ev), {"x"})
    bus.publish("x")
    bus.unsubscribe(fn)
    bus.publish("x")
    assert len(got) == 1


def test_plan_errors_and_parallel_adds(tmp_path, monkeypatch):
    video = str(tmp_path / "v.mkv")
    Path(video).write_bytes(b"")
    import vidai.edit as E

    monkeypatch.setattr(service, "new_plan", lambda v: E.EditPlan(source=v, duration=100.0))
    service.plan(video, "new")
    with pytest.raises(ValueError, match="unknown action"):
        service.plan(video, "delete")
    with pytest.raises(ValueError, match="needs index"):
        service.plan(video, "remove")
    ths = [threading.Thread(target=lambda i=i: service.plan(video, "add", ops=[
        {"op": "text", "start": i, "end": i + 1, "text": f"t{i}"}])) for i in range(20)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    shown = service.plan(video, "show")
    assert len(shown["ops"]) == 20  # no edit lost
    out = service.plan(video, "remove", index=[0, 1, 2])
    assert len(out["removed"]) == 3 and len(out["ops"]) == 17
    with pytest.raises(ValueError, match="no op at index"):
        service.plan(video, "remove", index=99)


def test_mcp_errors_reach_claude():
    import asyncio

    from vidai.mcp_server import build_server

    async def run():
        s = build_server()
        with pytest.raises(Exception) as ei:
            await s.call_tool("anchors", {"video": "/nonexistent.mkv", "mode": "summary"})
        return str(ei.value)

    msg = asyncio.run(run())
    assert "FileNotFoundError" in msg and "nonexistent" in msg
