import shutil
from types import SimpleNamespace

from vidai import AnchorFile, Brief
from vidai.recorder import Recorder


class FakeOBS:
    def __init__(self, video, out):
        self.video, self.out, self.calls = video, out, []

    def start_record(self):
        self.calls.append("start")

    def stop_record(self):
        self.calls.append("stop")
        shutil.copy(self.video, self.out)
        return SimpleNamespace(output_path=str(self.out))


def test_record_with_fake_obs(video, tmp_path):
    out = tmp_path / "rec.mp4"
    obs = FakeOBS(video, out)
    titles = iter(["Terminal", "Terminal", "Firefox"] + ["Firefox"] * 100)
    rec = Recorder(Brief(style="screencast", title="demo"), client=obs, window_probe=lambda: next(titles))
    assert "input_activity" in rec.stats and "window_focus" in rec.stats
    rec.start()
    rec.mark("section", "part 2")
    rec.mark("mistake")
    import time
    time.sleep(1.2)
    rec.stop()
    assert obs.calls == ["start", "stop"]
    saved = AnchorFile.load(out)
    assert saved.brief.title == "demo"
    kinds = [e.data["type"] for e in saved.events_of("markers")]
    assert kinds == ["section", "mistake"]
    assert [e.data["title"] for e in saved.events_of("window_focus")] == ["Terminal", "Firefox"]
    assert saved.segments_of("silence")  # analysis ran after recording
    assert saved.duration == 30.0 or saved.duration > 29  # replaced by real file duration
