"""A recording session: one folder with everything Claude needs.

    <root>/<YYYY-mm-dd_HHMM>_<slug>/
        session.json            brief + anchor config + capture config + status
        video.mkv               the recording
        video.mkv.anchors.json  anchors (live stats + markers + derived segments/events)

Flow: Claude asks the brief questions -> `create_session` (anchor config from the answers)
-> `launch_gui` (VidAI Recorder window) -> user records -> `wait_session` -> Claude edits.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .anchors import AnchorFile, select_stats
from .brief import Brief
from .capture import CaptureConfig
from .live.pipeline import LiveConfig, LivePipeline, mux_parts


class AnchorConfig(BaseModel):
    """Which stats to record and how Claude wants them tuned for this video."""
    stats: dict[str, str] = Field(default_factory=dict)  # stat -> why Claude chose it
    silence_db: float | None = None  # None = adaptive
    min_silence: float = 0.5
    scene_threshold: float = 0.12
    hotkeys: bool = True


FINAL_STATES = ("done", "error", "cancelled")


class SessionStatus(BaseModel):
    state: str = "ready"  # ready | recording | saving | done | error | cancelled
    started: str = ""
    finished: str = ""
    duration: float = 0.0
    video: str = ""
    anchors: str = ""
    message: str = ""
    gui_pid: int = 0
    recovered: bool = False


class Session(BaseModel):
    dir: str
    brief: Brief
    anchors: AnchorConfig
    capture: CaptureConfig
    live: LiveConfig = Field(default_factory=LiveConfig)
    status: SessionStatus = Field(default_factory=SessionStatus)

    @property
    def path(self) -> Path:
        return Path(self.dir) / "session.json"

    @property
    def video_path(self) -> Path:
        return Path(self.dir) / "video.mkv"

    def save(self) -> Path:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp, self.path)
        return self.path

    @classmethod
    def load(cls, path: str | Path) -> "Session":
        path = Path(path)
        if path.is_dir():
            path = path / "session.json"
        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def set_status(self, **kw: Any) -> None:
        for k, v in kw.items():
            setattr(self.status, k, v)
        self.save()


def default_root() -> Path:
    env = os.environ.get("VIDAI_VIDEOS")
    if env:
        return Path(env)
    videos = Path.home() / "Videos"
    return (videos if videos.exists() else Path.home()) / "VidAI"


def default_capture(brief: Brief) -> CaptureConfig:
    mode = {"talking_head": "camera", "vlog": "camera", "screencast": "screen",
            "slides": "screen", "screencast_with_camera": "screen+camera"}.get(brief.style, "screen+camera")
    return CaptureConfig(mode=mode)


def default_live(brief: Brief, capture: CaptureConfig) -> LiveConfig:
    langs = brief.languages
    # English only for now (reliable wake word + commands); set stt_language="ar" / None to change
    return LiveConfig(stt=capture.mic, stt_language="en", stt_languages=["en"],
                      ocr=capture.mode in ("screen", "screen+camera"))


def create_session(brief: Brief | dict, stats: dict[str, str] | list[str] | None = None,
                   capture: CaptureConfig | dict | None = None, root: str | Path | None = None,
                   live: LiveConfig | dict | None = None, **anchor_params: Any) -> Session:
    brief = brief if isinstance(brief, Brief) else Brief(**brief)
    if stats is None:
        stats = select_stats(brief)
    elif isinstance(stats, list):
        sug = select_stats(brief)
        stats = {s: sug.get(s, "chosen by Claude") for s in stats}
    stats.setdefault("markers", "user hotkeys/buttons")
    cap = capture if isinstance(capture, CaptureConfig) else (
        CaptureConfig(**capture) if capture else default_capture(brief))
    slug = re.sub(r"[^\w\-]+", "_", brief.title or "recording", flags=re.UNICODE).strip("_")[:40] or "recording"
    d = Path(root or default_root()) / f"{time.strftime('%Y-%m-%d_%H%M%S')}_{slug}"
    d.mkdir(parents=True, exist_ok=True)
    lv = live if isinstance(live, LiveConfig) else (
        default_live(brief, cap).model_copy(update=live) if live else default_live(brief, cap))
    s = Session(dir=str(d), brief=brief, anchors=AnchorConfig(stats=stats, **anchor_params), capture=cap, live=lv)
    s.save()
    return s


class SessionRecorder:
    """Runs one recording for a session: live pipeline + samplers -> video + anchors. Used by the GUI."""

    def __init__(self, session: Session, on_frame=None, on_stop_request=None) -> None:
        self.session = session
        self.on_frame = on_frame
        self.on_stop_request = on_stop_request
        self.pipe: LivePipeline | None = None
        self.samplers: list = []
        self._hotkeys = None
        self._wall_start = ""

    @property
    def cap(self) -> LivePipeline | None:  # kept for callers that used the old capture object
        return self.pipe

    def start(self) -> None:
        from .recorder import HOTKEYS, InputActivitySampler, WindowFocusSampler

        s = self.session
        self.pipe = LivePipeline(s.capture, s.live, s.video_path, s.dir, on_frame=self.on_frame,
                                 min_silence=s.anchors.min_silence, silence_db=s.anchors.silence_db,
                                 scene_threshold=s.anchors.scene_threshold, on_stop_request=self.on_stop_request)
        self.pipe.start()
        clock = self.pipe.clock
        if "input_activity" in s.anchors.stats and s.capture.mode != "test":
            self.samplers.append(InputActivitySampler(clock))
        if "window_focus" in s.anchors.stats and s.capture.mode != "test":
            self.samplers.append(WindowFocusSampler(clock))
        for smp in self.samplers:
            smp.bus = self.pipe.bus
            smp.start()
        if s.anchors.hotkeys and s.capture.mode != "test":
            try:
                from pynput import keyboard

                self._hotkeys = keyboard.GlobalHotKeys({k: (lambda t=t: self.mark(t)) for k, t in HOTKEYS.items()})
                self._hotkeys.start()
            except Exception:
                self._hotkeys = None
        self._wall_start = time.strftime("%Y-%m-%d %H:%M:%S")
        s.set_status(state="recording", started=self._wall_start, video=str(s.video_path), message="")

    def mark(self, kind: str, note: str = "") -> float:
        if self.pipe:
            self.pipe.command({"cmd": "mark", "type": kind, "note": note, "source": "user"}, source="user")
            return self.pipe.clock()
        return 0.0

    def command(self, c: dict) -> dict:
        return self.pipe.command(c, source="gui") if self.pipe else {"error": "not recording"}

    @property
    def elapsed(self) -> float:
        return self.pipe.clock() if self.pipe else 0.0

    def stop(self) -> AnchorFile:
        from .analyze import scene_changes, silence_from_level
        from .anchors import Event, Segment, Series
        from .ffmpeg import probe
        from .live.bus import read_events

        s = self.session
        s.set_status(state="saving", message="finishing the video and writing anchors")
        assert self.pipe is not None
        if self._hotkeys:
            self._hotkeys.stop()
        self.pipe.stop()
        for smp in self.samplers:
            smp.stop()
        if not s.video_path.exists():
            s.set_status(state="error", message=self.pipe.error or "no video was written")
            raise RuntimeError(s.status.message)
        dur = probe(str(s.video_path)).duration
        a = AnchorFile(video=str(s.video_path), duration=dur, brief=s.brief, selected_stats=dict(s.anchors.stats),
                       recorded_at=self._wall_start)
        wanted = set(s.anchors.stats)
        levels = Series(rate_hz=10.0, values=list(self.pipe.audio.levels))
        if levels.values:
            if "audio_level" in wanted:
                a.series["audio_level"] = levels
            if "silence" in wanted:
                a.add_segments(silence_from_level(levels, s.anchors.silence_db, s.anchors.min_silence))
        mot = Series(rate_hz=self.pipe.motion.rate, values=list(self.pipe.motion.values))
        if mot.values and wanted & {"motion", "scene_change"}:
            if "motion" in wanted:
                a.series["motion"] = mot
            if "scene_change" in wanted:
                a.add_events(scene_changes(mot, s.anchors.scene_threshold))
        for smp in self.samplers:
            evs, series = smp.result(dur)
            if evs:
                a.add_events(evs)
            if series is not None:
                a.series[smp.stat] = series
        # everything that happened live becomes anchors
        live = read_events(Path(s.dir) / "live.jsonl", limit=10 ** 9)
        a.add_events([Event(t=e["t"], kind="markers", data=e["data"]) for e in live if e["kind"] == "marker"])
        a.add_segments([Segment(start=e["data"]["start"], end=e["data"]["end"], kind="speech",
                                data={"text": e["data"]["text"], "lang": e["data"].get("lang"),
                                      "is_command": e["data"].get("is_command", False)})
                        for e in live if e["kind"] == "transcript"])
        for kind, akind in (("screen_text", "screen_text"), ("voice_command", "voice_command"),
                            ("claude", "claude_request"), ("action", "live_action"), ("error", "live_error")):
            evs = [Event(t=e["t"], kind=akind, data=e["data"]) for e in live if e["kind"] == kind]
            if evs:
                a.add_events(evs)
        for name, learner in self.pipe.learners.items():
            if learner.segments:
                a.add_segments([Segment(start=round(x, 3), end=round(y, 3), kind=f"learner:{name}",
                                        data={"label": lab}) for x, y, lab in learner.segments])
            if learner.model.ready:  # keep the instant model for later videos / offline use
                from .lab import ModelSpec, save_model

                save_model(learner.model, ModelSpec(name=f"{Path(s.dir).name}_{name}"[:80],
                                                    class_path="vidai.live.learn:InstantKNN",
                                                    task="frame_classifier", hparams=learner.model.hparams,
                                                    description=f"instant model '{name}' from {s.brief.title}"))
        for kind in ("speech", "screen_text", "voice_command"):
            if (a.segments_of(kind) or a.events_of(kind)) and kind not in a.selected_stats:
                a.selected_stats[kind] = "live"
        path = a.save()
        from .profile import Profile

        prof = Profile()  # remember the answers the user gives every time
        prof.set_pref("brief_defaults", {k: v for k, v in {"language": s.brief.language, "style": s.brief.style,
                                                           "audience": s.brief.audience}.items() if v})
        prof.set_pref("capture_mode", s.capture.mode if s.capture.mode != "test" else prof.pref("capture_mode"))
        s.set_status(state="done", finished=time.strftime("%Y-%m-%d %H:%M:%S"), duration=round(dur, 2),
                     anchors=str(path), message="recording saved; back to Claude for editing")
        return a


def launch_gui(session: Session | str | Path, autostart: bool = False) -> int:
    """Open the VidAI Recorder window in its own process (returns immediately with the PID)."""
    sdir = session.dir if isinstance(session, Session) else str(session)
    cmd = [sys.executable, "-m", "vidai.gui", sdir] + (["--autostart"] if autostart else [])
    log = open(Path(sdir) / "gui.log", "ab")
    p = subprocess.Popen(cmd, stdout=log, stderr=log, start_new_session=True)
    s = Session.load(sdir)
    s.set_status(gui_pid=p.pid)
    return p.pid


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:  # a zombie (exited child not yet reaped) is not alive
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except OSError:
        return True


def recover_session(session_dir: str | Path) -> SessionStatus:
    """Finish a session whose recorder window died (crash, killed, closed mid-save).

    Repairs a truncated video (remux, no re-encode), rebuilds the anchors from the file
    (live-only stats such as markers are lost), and marks the session done or cancelled."""
    from .analyze import analyze
    from . import ffmpeg

    s = Session.load(session_dir)
    v = s.video_path
    vpart = v.with_name(v.stem + ".part-video.mkv")
    if not v.exists() and vpart.exists():
        try:  # the live pipeline records audio and video separately; join what was written
            mux_parts(vpart, v.with_name(v.stem + ".part-audio.mka"), v)
        except Exception as e:
            s.set_status(state="error", message=f"recovery failed: {e}")
            return s.status
    if not v.exists() or v.stat().st_size < 1024:
        s.set_status(state="cancelled", message="the recorder was closed without a recording")
        return s.status
    fixed = v.with_name("video.fixed.mkv")
    try:
        ffmpeg.run(["-err_detect", "ignore_err", "-i", str(v), "-map", "0", "-c", "copy", str(fixed)])
        os.replace(v, v.with_name("video.damaged.mkv"))  # keep the original until the user deletes it
        os.replace(fixed, v)
        a = analyze(str(v), s.anchors.stats, silence_db=s.anchors.silence_db, min_silence=s.anchors.min_silence)
        a.brief = s.brief
        a.recorded_at = s.status.started
        path = a.save()
    except Exception as e:
        fixed.unlink(missing_ok=True)
        s.set_status(state="error", message=f"recovery failed: {e}")
        return s.status
    s.set_status(state="done", recovered=True, duration=round(a.duration, 2), anchors=str(path),
                 finished=time.strftime("%Y-%m-%d %H:%M:%S"),
                 message="recovered after the recorder window closed unexpectedly (markers were lost)")
    return s.status


def check_session(session_dir: str | Path) -> SessionStatus:
    """Current status; if the recorder window is gone but the state is not final, recover it."""
    st = Session.load(session_dir).status
    if st.state not in FINAL_STATES and st.gui_pid and not pid_alive(st.gui_pid):
        time.sleep(1.0)  # the window may be writing its final status right now
        st = Session.load(session_dir).status
        if st.state not in FINAL_STATES:
            return recover_session(session_dir)
    return st


def wait_session(session_dir: str | Path, timeout: float = 3600, poll: float = 1.0) -> SessionStatus:
    """Block until the recording is done/error/cancelled (or timeout); returns the status.
    Notices when the recorder window dies and recovers the session."""
    end = time.time() + timeout
    while True:
        st = check_session(session_dir)
        if st.state in FINAL_STATES or time.time() >= end:
            return st
        time.sleep(poll)
