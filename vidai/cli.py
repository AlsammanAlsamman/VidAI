"""`vidai` command line. Every command prints JSON so Claude can read it."""
from __future__ import annotations

import argparse
import json
import sys

from . import service


def _print(obj) -> None:
    json.dump(obj, sys.stdout, indent=2, ensure_ascii=False, default=str)
    print()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="vidai", description="Claude video-editing plugin")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("brief", help="ask the brief questions in the terminal and save brief.json")
    s.add_argument("out", nargs="?", default="brief.json")

    s = sub.add_parser("stats", help="list stats / suggest stats for a brief")
    s.add_argument("--brief")

    s = sub.add_parser("studio", help="open the VidAI Recorder for a brief (asks the brief if none given)")
    s.add_argument("--brief", help="brief.json (default: ask the questions)")
    s.add_argument("--mode", choices=["camera", "screen", "screen+camera", "test"])
    s.add_argument("--stats", nargs="*")
    s.add_argument("--wait", action="store_true", help="wait until the recording is done")

    sub.add_parser("devices", help="list cameras, screen and mic capture method")

    s = sub.add_parser("setup", help="download all models once (speech, hands, face)")
    s.add_argument("--whisper", nargs="*", default=["base"])

    s = sub.add_parser("recover", help="repair a session whose recorder window died")
    s.add_argument("session")

    s = sub.add_parser("record", help="record with OBS until Enter is pressed")
    s.add_argument("--brief")
    s.add_argument("--stats", nargs="*")
    s.add_argument("--host", default="localhost")
    s.add_argument("--port", type=int, default=4455)
    s.add_argument("--password", default="")

    s = sub.add_parser("analyze", help="compute anchors for a video")
    s.add_argument("video")
    s.add_argument("--stats", nargs="*")
    s.add_argument("--workers", type=int, default=3)
    s.add_argument("--silence-db", type=float)
    s.add_argument("--fast", action="store_true", help="keyframe-only visual analysis (~10x faster)")

    s = sub.add_parser("anchors", help="query anchors")
    s.add_argument("video")
    s.add_argument("--mode", default="summary", choices=["summary", "at", "segments", "events", "series", "chunks"])
    s.add_argument("--t", type=float, default=0.0)
    s.add_argument("--kind")
    s.add_argument("--t0", type=float, default=0.0)
    s.add_argument("--t1", type=float)
    s.add_argument("--workers", type=int, default=3, help="for --mode chunks")

    s = sub.add_parser("frame", help="extract a frame (png)")
    s.add_argument("video")
    s.add_argument("t", type=float)
    s.add_argument("--out")

    s = sub.add_parser("sheet", help="contact sheet of frames at times")
    s.add_argument("video")
    s.add_argument("times", type=float, nargs="+")
    s.add_argument("--out")

    s = sub.add_parser("plan", help="build the edit plan")
    s.add_argument("video")
    s.add_argument("action", choices=["new", "show", "add", "remove", "remove_gaps", "cut_mistakes", "chapters",
                                      "note"])
    s.add_argument("--ops", help="JSON list of ops (for add)")
    s.add_argument("--index", type=int, nargs="+", help="op index(es) to remove (see: plan VIDEO show)")
    s.add_argument("--min-gap", type=float, default=0.8)
    s.add_argument("--keep", type=float, default=0.3)
    s.add_argument("--titles", nargs="*")
    s.add_argument("--note")
    s.add_argument("--burn-subtitles", action=argparse.BooleanOptionalAction, default=None)

    s = sub.add_parser("render", help="render the plan (parallel, YouTube-ready)")
    s.add_argument("video")
    s.add_argument("--out")
    s.add_argument("--workers", type=int, default=3)
    s.add_argument("--crf", type=int, default=20)
    s.add_argument("--height", type=int)
    s.add_argument("--no-youtube", action="store_true", help="skip the .srt and YouTube description")

    sub.add_parser("models", help="list lab models")

    s = sub.add_parser("classify", help="run a frame_classifier lab model over a video -> anchor segments")
    s.add_argument("video")
    s.add_argument("model")
    s.add_argument("--kind")
    s.add_argument("--fps", type=float, default=2.0)

    s = sub.add_parser("sessions", help="recent recording sessions")
    s.add_argument("--limit", type=int, default=10)

    s = sub.add_parser("status", help="state of a recording session")
    s.add_argument("session")

    s = sub.add_parser("permissions", help="show or set what VidAI may do on its own (you decide, not Claude)")
    s.add_argument("mode", nargs="?", choices=["ask", "full"])
    s.add_argument("--session", help="a session folder (default: outside recordings)")
    s.add_argument("--hours", type=float, default=4.0, help="full access lasts this long")

    s = sub.add_parser("train", help="train a lab model (one or more rounds)")
    s.add_argument("class_path")
    s.add_argument("data")
    s.add_argument("--metric", required=True)
    s.add_argument("--target", type=float, required=True)
    s.add_argument("--lower-is-better", action="store_true")
    s.add_argument("--hparams", default="{}")
    s.add_argument("--save-as")
    s.add_argument("--rounds", type=int, default=1)

    sub.add_parser("mcp", help="run the MCP server (stdio)")

    a = ap.parse_args(argv)
    try:
        _run(a)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:  # JSON, like every other output, so Claude can read what went wrong
        _print({"error": f"{type(e).__name__}: {e}"})
        sys.exit(1)


def _run(a: argparse.Namespace) -> None:
    if a.cmd == "brief":
        from .brief import ask_interactive

        b = ask_interactive()
        _print(service.save_brief(a.out, b.model_dump()))
    elif a.cmd == "stats":
        from .brief import Brief

        _print(service.suggest_stats(Brief.load(a.brief).model_dump() if a.brief else None))
    elif a.cmd == "studio":
        from .brief import Brief, ask_interactive

        b = Brief.load(a.brief) if a.brief else ask_interactive()
        res = service.studio_start(b.model_dump(), a.stats, {"mode": a.mode} if a.mode else None)
        _print(res)
        if a.wait:
            _print(service.studio_wait(res["session"], timeout=6 * 3600))
    elif a.cmd == "recover":
        _print(service.studio_recover(a.session))
    elif a.cmd == "setup":
        from .models_dl import setup

        _print(setup(tuple(a.whisper)))
    elif a.cmd == "devices":
        _print(service.studio_devices())
    elif a.cmd == "record":
        _print(service.record_start(a.brief, a.stats, a.host, a.port, a.password))
        input("Recording... press Enter to stop.\n")
        _print(service.record_stop())
    elif a.cmd == "analyze":
        _print(service.analyze_video(a.video, a.stats, a.workers, a.silence_db, fast=a.fast))
    elif a.cmd == "anchors":
        _print(service.anchors(a.video, a.mode, a.t, a.kind, a.t0, a.t1, a.workers))
    elif a.cmd == "frame":
        _print(service.frame(a.video, a.t, a.out))
    elif a.cmd == "sheet":
        _print(service.contact_sheet(a.video, a.times, a.out))
    elif a.cmd == "plan":
        index = a.index[0] if a.index and len(a.index) == 1 else a.index
        _print(service.plan(a.video, a.action, json.loads(a.ops) if a.ops else None, index, a.min_gap, a.keep,
                            a.titles, note=a.note, burn_subtitles=a.burn_subtitles))
    elif a.cmd == "render":
        _print(service.render_video(a.video, a.out, a.workers, a.crf, a.height, youtube=not a.no_youtube))
    elif a.cmd == "classify":
        _print(service.classify_video(a.video, a.model, a.kind, a.fps))
    elif a.cmd == "sessions":
        _print(service.studio_sessions(limit=a.limit))
    elif a.cmd == "status":
        _print(service.studio_status(a.session))
    elif a.cmd == "permissions":
        from . import actions

        if a.mode:  # typed by the user in a terminal: this IS their consent
            actions.set_mode(a.session, a.mode, hours=a.hours)
        _print(service.vidai_permissions(a.session))
    elif a.cmd == "models":
        _print(service.models())
    elif a.cmd == "train":
        _print(service.train(a.class_path, a.data, a.metric, a.target, not a.lower_is_better,
                             json.loads(a.hparams), a.save_as, max_rounds=a.rounds))
    elif a.cmd == "mcp":
        from .mcp_server import main as mcp_main

        mcp_main()


if __name__ == "__main__":
    main()
