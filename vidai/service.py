"""High-level operations returning JSON-friendly dicts. Shared by the MCP server and the CLI."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from . import lab, preview
from .analyze import analyze
from .anchors import STATS, AnchorFile, Event, list_stats, select_stats
from .brief import QUESTIONS, Brief
from .edit import EditPlan, new_plan
from .export import youtube_package
from .render import render

_recorder = None


# ---------- brief ----------
def brief_questions() -> dict:
    return {"questions": QUESTIONS, "styles_note": "style decides which stats are useful"}


def save_brief(path: str, brief: dict) -> dict:
    b = Brief(**brief)
    return {"saved": str(b.save(path)), "suggested_stats": select_stats(b)}


def suggest_stats(brief: dict | None = None) -> dict:
    b = Brief(**(brief or {}))
    return {"suggested": select_stats(b),
            "all": {s.name: {"description": s.description, "source": s.source, "kind": s.kind,
                             "available": s.available} for s in list_stats(include_unavailable=True)}}


# ---------- studio (VidAI's own recorder + GUI) ----------
def studio_devices() -> dict:
    """Cameras (with formats), screen size and how the mic will be captured."""
    from .capture import devices

    return devices()


def studio_start(brief: dict, stats: dict[str, str] | list[str] | None = None, capture: dict | None = None,
                 silence_db: float | None = None, min_silence: float = 0.5, root: str | None = None,
                 open_gui: bool = True, autostart: bool = False, live: dict | None = None) -> dict:
    """Create a recording session from the brief + Claude's anchor configuration and open the VidAI Recorder.

    stats: {stat: why} chosen by Claude (None = default selection from the brief).
    capture: {"mode": "camera"|"screen"|"screen+camera", "camera": "/dev/video0", "camera_size": "1280x720",
              "pip_position": "bottom-right", "pip_width": 0.22, "mic": true, "out_height": 1080, ...}
    live: live processing from the start (see live_guide): {"stt": true, "stt_language": "ar", "ocr": true,
          "processors": [{"name","type"|"file","params"}], "rules": [...], "learners": [...]}
    Then follow the live stream with live_stats / live_control, and studio_wait until it is done."""
    from .session import create_session, launch_gui

    s = create_session(brief, stats, capture, root=root, live=live, silence_db=silence_db, min_silence=min_silence)
    out = {"session": s.dir, "video": str(s.video_path), "anchor_config": s.anchors.model_dump(),
           "capture": s.capture.model_dump(), "live": s.live.model_dump()}
    if open_gui:
        out["gui_pid"] = launch_gui(s, autostart=autostart)
        out["tell_user"] = ("The VidAI Recorder window is open. Press Record (or space). While recording: "
                            "ctrl+alt+m marker, ctrl+alt+x mistake, ctrl+alt+n new section, ctrl+alt+i important, "
                            "ctrl+alt+s stop. Tell me when you are done.")
    return out


def studio_status(session: str) -> dict:
    """Session state: ready | recording | saving | done | error | cancelled (auto-recovers a dead recorder)."""
    from .session import check_session

    return {"session": session, **check_session(session).model_dump()}


def studio_recover(session: str) -> dict:
    """Repair a session whose recorder window died: fix the video file and rebuild anchors."""
    from .session import recover_session

    return {"session": session, **recover_session(session).model_dump()}


def studio_wait(session: str, timeout: float = 600) -> dict:
    """Wait until the recording is finished (done / error / cancelled) or timeout seconds pass.
    If the recorder window closed or crashed, the session is recovered automatically."""
    from .session import wait_session

    st = wait_session(session, timeout)
    out = {"session": session, **st.model_dump()}
    if st.state == "done":
        out["anchors_summary"] = AnchorFile.load(st.video).summary()
    return out


def studio_sessions(root: str | None = None, limit: int = 10) -> dict:
    """Recent recording sessions (newest first)."""
    from .session import Session, default_root

    base = Path(root) if root else default_root()
    items = []
    for d in sorted(base.glob("*/session.json"), reverse=True)[:limit]:
        s = Session.load(d)
        items.append({"session": s.dir, "title": s.brief.title, "state": s.status.state,
                      "duration": s.status.duration})
    return {"root": str(base), "sessions": items}


# ---------- live processing (Claude's slow loop) ----------
def live_guide(topic: str | None = None) -> dict:
    """How to drive live processing. topics: overview, stats, commands, rules, processors, models (None = all)."""
    from .live.guide import guide

    return {"guide": guide(topic)}


def _live_log(session: str) -> Path:
    return Path(session) / "live.jsonl"


def live_stats(session: str, since: int = 0, kinds: list[str] | None = None, limit: int = 100) -> dict:
    """New live events since seq `since` (pass back `next` next time). Also the latest value per kind."""
    from .live.bus import read_events

    evs = read_events(_live_log(session), since, None, 10 ** 7)
    latest: dict[str, Any] = {}
    for e in evs:
        latest[e["kind"]] = {"t": e["t"], **e["data"]}
    sel = [e for e in evs if not kinds or e["kind"] in kinds][-limit:]
    nxt = evs[-1]["seq"] if evs else since
    from .session import Session

    st = Session.load(session).status.state
    return {"state": st, "next": nxt, "events": sel, "latest": latest,
            "claude_requests": [e for e in evs if e["kind"] == "claude"]}


def live_control(session: str, commands: list[dict], wait: float = 3.0, done: bool = True) -> dict:
    """Send commands to the running recorder (see live_guide('commands')). Waits for their ack/error.
    done=True also tells the recorder the user's pending request is answered (hides the thinking icon);
    pass done=False when you will send more commands for the same request."""
    import json
    import time as _t

    from .live.bus import read_events
    from .session import Session

    state = Session.load(session).status.state
    if state not in ("recording", "ready"):  # "ready" = the recorder window is open in preview
        return {"error": f"not recording (state={state})"}
    before = read_events(_live_log(session), 0, None, 1)
    seq0 = before[-1]["seq"] if before else 0
    commands = list(commands) + ([{"cmd": "done"}] if done else [])
    with open(Path(session) / "control.jsonl", "a", encoding="utf-8") as f:
        for c in commands:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    results: list[dict] = []
    end = _t.time() + wait
    while _t.time() < end and len(results) < len(commands):
        _t.sleep(0.15)
        results = [e for e in read_events(_live_log(session), seq0, ["ack", "error"], 10 ** 6)
                   if e["data"].get("source") in ("claude", None) or e["kind"] == "error"]
    shown = [e for e in results if e["data"].get("command") != "done"]
    return {"results": [{"kind": e["kind"], **e["data"]} for e in shown],
            "missing": max(0, len(commands) - len(results))}


def live_processor(session: str, name: str, code: str, params: dict | None = None, duration: float | None = None,
                   wait: float = 4.0, done: bool = True) -> dict:
    """Write a processor (Python code, see live_guide('processors')) and load it into the running recorder.
    Replaces a processor with the same name. Returns the ack or the error to fix."""
    from .live.bus import read_events

    last = read_events(_live_log(session), 0, None, 1)
    seq0 = last[-1]["seq"] if last else 0  # only errors of *this* version count
    d = Path(session) / "processors"
    d.mkdir(exist_ok=True)
    path = d / f"{name}.py"
    path.write_text(code, encoding="utf-8")
    cmd = {"cmd": "add", "name": name, "file": str(path), "params": params or {}}
    if duration:
        cmd["for"] = duration
    out = live_control(session, [cmd], wait, done=False)
    import time as _t

    _t.sleep(0.8)  # let it run a few frames: runtime errors show up as error events
    errs = [e["data"] for e in read_events(_live_log(session), seq0, ["error"], 10 ** 6)
            if e["data"].get("processor") == name][-1:]
    out["file"] = str(path)
    if errs:
        out["runtime_error"] = errs[0]  # keep the icon on: fix the code and call again
    elif done:
        live_control(session, [], wait=1.0, done=True)
    return out


def live_status(session: str) -> dict:
    """Processors, rules, learners and performance of the running recorder."""
    r = live_control(session, [{"cmd": "status"}], done=False)
    return r["results"][0] if r.get("results") else r


# ---------- recording with OBS (optional backend) ----------
def record_start(brief_path: str | None = None, stats: list[str] | None = None, host: str = "localhost",
                 port: int = 4455, password: str = "") -> dict:
    global _recorder
    from .recorder import Recorder

    b = Brief.load(brief_path) if brief_path else Brief()
    sel = {s: select_stats(b).get(s, "chosen by Claude") for s in stats} if stats else None
    _recorder = Recorder(b, sel, host=host, port=port, password=password)
    _recorder.start()
    return {"recording": True, "stats": _recorder.stats,
            "hotkeys": {"ctrl+alt+m": "marker", "ctrl+alt+x": "mistake", "ctrl+alt+n": "section",
                        "ctrl+alt+i": "important"}}


def record_mark(kind: str = "marker", note: str = "") -> dict:
    if not _recorder or not _recorder.recording:
        return {"error": "not recording"}
    _recorder.mark(kind, note)
    return {"marked": kind, "t": round(_recorder.clock(), 2)}


def record_stop(workers: int = 3) -> dict:
    global _recorder
    if not _recorder or not _recorder.recording:
        return {"error": "not recording"}
    a = _recorder.stop(workers=workers)
    _recorder = None
    return {"video": a.video, "anchors": str(AnchorFile.path_for(a.video)) if a.video else None,
            "summary": a.summary()}


# ---------- anchors ----------
def analyze_video(video: str, stats: list[str] | None = None, workers: int = 3,
                  silence_db: float | None = None, min_silence: float = 0.5, fast: bool = False) -> dict:
    """fast=True: keyframe-only decoding for visual stats (~10x faster, scene cuts located to ~1-2 s)."""
    a = analyze(video, stats, workers=workers, silence_db=silence_db, min_silence=min_silence, fast=fast)
    path = a.save()
    return {"anchors": str(path), "summary": a.summary()}


def anchors(video: str, mode: str = "summary", t: float = 0.0, kind: str | None = None,
            t0: float = 0.0, t1: float | None = None) -> dict:
    a = AnchorFile.load(video)
    if mode == "summary":
        return a.summary()
    if mode == "at":
        return a.at(t)
    if mode == "segments":
        return {"segments": [s.model_dump() for s in a.segments if (kind is None or s.kind == kind)
                             and s.end > t0 and (t1 is None or s.start < t1)]}
    if mode == "events":
        return {"events": [e.model_dump() for e in a.events_of(kind, t0, t1)]}
    if mode == "series":
        s = a.series[kind or "audio_level"]
        w = s.window(t0, t1 if t1 is not None else a.duration)
        return {"rate_hz": s.rate_hz, "t0": t0, "values": [round(float(x), 3) for x in w]}
    if mode == "chunks":
        return {"chunks": a.chunks(int(t) or 3)}
    raise ValueError(f"unknown mode {mode}")


def add_anchor_events(video: str, events: list[dict]) -> dict:
    """Let Claude (or a lab model) write its own anchors, e.g. 'topic_start' at 12.4 s."""
    a = AnchorFile.load(video)
    evs = [Event(**e) for e in events]
    a.add_events(evs, replace_kinds=False)
    a.save()
    return {"added": len(evs)}


# ---------- previews ----------
def frame(video: str, t: float, out: str | None = None, width: int = 640) -> dict:
    return {"image": str(preview.frame(video, t, out, width))}


def contact_sheet(video: str, times: list[float], out: str | None = None, cols: int = 4) -> dict:
    return {"image": str(preview.contact_sheet(video, times, out, cols))}


# ---------- edit plan ----------
def _plan(video: str) -> EditPlan:
    p = EditPlan.path_for(video)
    return EditPlan.load(p) if p.exists() else new_plan(video)


def plan(video: str, action: str = "show", ops: list[dict] | None = None, index: int | None = None,
         min_gap: float = 0.8, keep: float = 0.3, titles: list[str] | None = None,
         note: str | None = None, burn_subtitles: bool | None = None) -> dict:
    """Actions: new, show, add, remove (index), remove_gaps, cut_mistakes, chapters, note."""
    p = new_plan(video) if action == "new" else _plan(video)
    info: dict[str, Any] = {}
    if action == "add":
        p.add(*(ops or []))
    elif action == "remove" and index is not None:
        info["removed"] = p.ops.pop(index).model_dump()
    elif action in ("remove_gaps", "cut_mistakes", "chapters"):
        a = AnchorFile.load(video)
        if action == "remove_gaps":
            info["cuts_added"] = p.remove_gaps(a, min_gap, keep)
        elif action == "cut_mistakes":
            info["cuts_added"] = p.cut_mistakes(a)
        else:
            info["chapters_added"] = p.chapters_from_anchors(a, titles)
    if note:
        p.notes.append(note)
    if burn_subtitles is not None:
        p.burn_subtitles = burn_subtitles
    path = p.save()
    out = {"plan": str(path), "summary": p.summary(), **info}
    if action == "show":
        out["ops"] = [f"{i}: {o.model_dump()}" for i, o in enumerate(p.ops)]
    return out


def render_video(video: str, output: str | None = None, workers: int = 3, crf: int = 20,
                 scale_height: int | None = None, youtube: bool = True) -> dict:
    p = _plan(video)
    output = output or str(Path(video).with_name(Path(video).stem + "_vidai.mp4"))
    r = render(p, output, workers=workers, crf=crf, scale_height=scale_height)
    a_path = AnchorFile.path_for(video)
    brief = AnchorFile.load(a_path).brief if a_path.exists() else None
    out = {"output": r.output, "duration": r.duration, "chunks": r.chunks, "seconds": r.seconds, "srt": r.srt}
    if youtube:
        out["youtube"] = youtube_package(r, brief)
    return out


# ---------- lab ----------
def models() -> dict:
    from . import native

    return {"models": lab.list_models(), "home": str(lab.models_dir()),
            "native_c": native.AVAILABLE, "native_error": native.BUILD_ERROR}


def train(class_path: str, data: str, metric: str, target: float, higher_is_better: bool = True,
          hparams: dict | None = None, save_as: str | None = None, description: str = "",
          max_rounds: int = 1) -> dict:
    """Train a lab model. data = .npz with X_train, Y_train, X_val, Y_val.
    class_path = 'vidai.lab.examples:ColorMatch' or '/path/model.py:MyModel' (code is copied into the registry).
    Claude usually runs one round at a time (max_rounds=1), reads the metrics, changes hparams, repeats."""
    mod = class_path.partition(":")[0]
    code_file = mod if mod.endswith(".py") else None
    cls = lab._import_class(class_path)
    with np.load(data) as z:
        tr, va = (z["X_train"], z["Y_train"]), (z["X_val"], z["Y_val"])
    rep = lab.train_until_suitable(cls, tr, va, metric, target, higher_is_better, hparams,
                                   max_rounds=max_rounds, save_as=save_as, description=description,
                                   code_file=code_file)
    return {"suitable": rep.suitable, "metrics": rep.best_metrics, "hparams": rep.best_hparams,
            "rounds": rep.rounds, "saved_to": rep.saved_to,
            "next": None if rep.suitable else "adjust hparams (or the model code) and train again"}


def classify_video(video: str, model: str, kind: str | None = None, fps: float = 2.0,
                   min_len: float = 0.5) -> dict:
    """Run a frame_classifier lab model over the video and store positive ranges as anchor segments."""
    from .anchors import Segment

    m, spec = lab.load_model(model)
    X, ts = preview.frame_features(video, fps)
    y = np.asarray(m.predict(X)).astype(bool)
    kind = kind or model
    segs, i = [], 0
    while i < len(y):
        if y[i]:
            j = i
            while j < len(y) and y[j]:
                j += 1
            if (j - i) / fps >= min_len:
                segs.append(Segment(start=float(ts[i]), end=float(j / fps), kind=kind, data={"model": model}))
            i = j
        else:
            i += 1
    if AnchorFile.path_for(video).exists():
        a = AnchorFile.load(video)
    else:
        from .ffmpeg import probe

        a = AnchorFile(video=video, duration=probe(video).duration)
    if segs:
        a.add_segments(segs)
    a.save()
    return {"segments": len(segs), "kind": kind}
