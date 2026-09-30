import asyncio
import json

from vidai import cli, service
from vidai.mcp_server import build_server


def test_cli_end_to_end(video, tmp_path, capsys):
    def run(*args):
        cli.main([str(a) for a in args])
        return json.loads(capsys.readouterr().out)

    s = run("analyze", video)
    assert s["summary"]["segments"]["silence"]["count"] == 3
    assert run("anchors", video, "--mode", "chunks", "--t", 3)["chunks"][1][0] == 6.5
    assert run("plan", video, "remove_gaps")["cuts_added"] == 3
    ops = json.dumps([{"op": "text", "start": 0, "end": 2, "text": "Hi"}])
    assert run("plan", video, "add", "--ops", ops)["summary"]["ops"]["text"] == 1
    assert run("plan", video, "chapters", "--titles", "Intro", "Blue")["chapters_added"] == 2
    r = run("render", video, "--out", tmp_path / "final.mp4")
    assert r["duration"] < 25 and r["youtube"]["package"].endswith(".youtube.txt")
    sheet = run("sheet", video, 1, 10, 20)
    assert sheet["image"].endswith(".png")


def test_brief_and_stats(tmp_path):
    r = service.save_brief(str(tmp_path / "b.json"), {"title": "x", "style": "talking_head", "language": "ar"})
    assert "motion" in r["suggested_stats"]
    assert "eye_state" in service.suggest_stats()["all"]


def test_mcp_tools_registered():
    tools = asyncio.run(build_server().list_tools())
    names = {t.name for t in tools}
    assert {"analyze_video", "anchors", "plan", "render_video", "train", "vidai_confirmed_action"} <= names
    assert "record_start" not in names  # legacy OBS backend: CLI / Python only
