"""VidAI's own recorder, run with synthetic sources (mode='test'): no camera/screen/mic needed."""
import time

import pytest

from vidai.ffmpeg import probe as _probe


def ffmpeg_probe(p):
    return _probe(str(p))

from vidai import service
from vidai.capture import CaptureConfig, LiveCapture, build_command
from vidai.session import Session, SessionRecorder, create_session


def test_build_command_screen_camera_pip():
    cfg = CaptureConfig(mode="screen+camera", screen_size="1366x768", display=":0")
    cmd = " ".join(build_command(cfg, "out.mkv", "V", "A", audio_from_stdin=True))
    assert "x11grab" in cmd and "v4l2" in cmd and "pipe:0" in cmd
    assert "overlay=W-w-24:H-h-24" in cmd and "asplit[arec][atapin]" in cmd
    assert "out.mkv" in cmd and cmd.endswith("-f f32le A")
    prev = " ".join(build_command(cfg, None, "V", "A", audio_from_stdin=True))
    assert "-f null -" in prev and "asplit" not in prev


def test_create_session_uses_brief(tmp_path):
    s = create_session({"title": "BLAST شرح", "style": "talking_head", "language": "ar"}, root=tmp_path)
    assert s.capture.mode == "camera"
    assert "markers" in s.anchors.stats and "motion" in s.anchors.stats
    s2 = Session.load(s.dir)
    assert s2.brief.title == "BLAST شرح" and s2.status.state == "ready"


def test_preview_delivers_frames_and_levels():
    frames = []
    c = LiveCapture(CaptureConfig(mode="test"), None, on_frame=frames.append)
    c.start()
    time.sleep(2.5)
    running = c.running
    c.stop()
    assert running and len(frames) >= 8 and frames[0].shape == (180, 320, 3)
    assert c.level_db > -60


def test_session_recording_writes_video_and_live_anchors(tmp_path):
    s = create_session({"title": "t", "style": "screencast"}, stats=["audio_level", "silence", "motion", "scene_change"],
                       capture=CaptureConfig(mode="test", out_height=360), root=tmp_path)
    r = SessionRecorder(s)
    r.start()
    time.sleep(2.0)
    r.mark("section", "part 2")
    time.sleep(4.5)
    a = r.stop()
    st = Session.load(s.dir).status
    assert st.state == "done" and st.duration == pytest.approx(a.duration)
    assert 5.5 < a.duration < 9
    assert "audio_level" in a.series and "motion" in a.series
    # test tone is silent for 1.5 s every 4 s
    assert len(a.segments_of("silence")) >= 1
    m = a.events_of("markers")
    assert m and m[0].data["type"] == "section" and m[0].data["note"] == "part 2" and 1.5 < m[0].t < 3.5
    assert service.anchors(str(s.video_path))["segments"]["silence"]["count"] >= 1


def test_studio_start_without_gui(tmp_path):
    r = service.studio_start({"title": "x", "style": "screencast_with_camera"}, stats={"silence": "gaps"},
                             root=str(tmp_path), open_gui=False)
    assert r["capture"]["mode"] == "screen+camera"
    assert set(r["anchor_config"]["stats"]) == {"silence", "markers"}
    assert service.studio_status(r["session"])["state"] == "ready"
    assert service.studio_sessions(str(tmp_path))["sessions"][0]["title"] == "x"


def test_dead_recorder_is_detected_and_recovered(tmp_path):
    """Window died mid-save: state stuck at 'saving' and a truncated MKV -> recovered to 'done'."""
    import subprocess

    from vidai.session import FINAL_STATES, check_session, wait_session

    s = create_session({"title": "crash"}, stats=["silence", "audio_level"],
                       capture=CaptureConfig(mode="test", out_height=360), root=tmp_path)
    r = SessionRecorder(s)
    r.start()
    time.sleep(4)
    for p in (r.pipe.cap, r.pipe.enc):  # simulate a crash: both ffmpeg processes die, no clean end of file
        p.kill()
        p.wait()
    assert not s.video_path.exists()  # only the separate audio/video parts are on disk
    dead = subprocess.Popen(["true"])
    dead.wait()
    s.set_status(state="saving", gui_pid=dead.pid)
    st = wait_session(s.dir, timeout=60)
    assert st.state == "done" and st.recovered and st.duration > 2
    assert s.video_path.exists() and ffmpeg_probe(s.video_path).has_audio
    from vidai import AnchorFile
    assert AnchorFile.load(s.video_path).series["audio_level"].values
    assert check_session(s.dir).state in FINAL_STATES


def test_closed_without_recording_is_cancelled(tmp_path):
    import subprocess

    from vidai.session import wait_session

    s = create_session({"title": "nothing"}, capture=CaptureConfig(mode="test"), root=tmp_path)
    dead = subprocess.Popen(["true"])
    dead.wait()
    s.set_status(gui_pid=dead.pid)
    assert wait_session(s.dir, timeout=10).state == "cancelled"
