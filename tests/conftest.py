import os
import shutil
import tempfile

import pytest

# Downloaded models ($VIDAI_ASSETS, default ~/.vidai/assets) are redirected to a throwaway dir so tests never
# read or write the user's real model cache. This runs at conftest import, i.e. before test modules are collected,
# so module-level `skipif(not hub.installed(...))` checks see the empty dir and skip cleanly.
# Set VIDAI_TEST_REAL_ASSETS=1 to run against the real (already downloaded) models instead.
_TMP_ASSETS = None
if os.environ.get("VIDAI_TEST_REAL_ASSETS") != "1":
    _TMP_ASSETS = tempfile.mkdtemp(prefix="vidai-test-assets-")
    os.environ["VIDAI_ASSETS"] = _TMP_ASSETS

from vidai.testing import make_test_video  # noqa: E402


def pytest_unconfigure(config):
    if _TMP_ASSETS:
        shutil.rmtree(_TMP_ASSETS, ignore_errors=True)


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
