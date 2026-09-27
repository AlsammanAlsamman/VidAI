import shutil

import pytest

from vidai.testing import make_test_video


@pytest.fixture(scope="session")
def _master_video(tmp_path_factory):
    return make_test_video(tmp_path_factory.mktemp("media") / "master.mp4")


@pytest.fixture
def video(tmp_path, _master_video):
    """A fresh copy per test (anchors/plans are written next to the video)."""
    dst = tmp_path / "talk.mp4"
    shutil.copy(_master_video, dst)
    return dst


@pytest.fixture(autouse=True)
def vidai_home(tmp_path, monkeypatch):
    monkeypatch.setenv("VIDAI_HOME", str(tmp_path / "vidai_home"))
    return tmp_path / "vidai_home"
