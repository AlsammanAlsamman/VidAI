import importlib
import re
import subprocess

import numpy as np
import pytest

from vidai import AnchorFile, Chapter, Subtitle, Zoom, ffmpeg, native, new_plan
from vidai.anchors import Event
from vidai.edit import Cut, EditPlan
from vidai.render import _chapters_meta, _output_chapters, _zoom_crop, render, snap_ranges, srt_items


analyze_mod = importlib.import_module("vidai.analyze")
render_mod = importlib.import_module("vidai.render")


def _ff(*args) -> str:
    return subprocess.run([ffmpeg.ffmpeg_exe(), "-hide_banner", *args], capture_output=True, text=True).stderr


def _frames_and_samples(path) -> tuple[int, int]:
    err = _ff("-i", str(path), "-map", "0:v:0", "-f", "null", "-")
    frames = int(re.findall(r"frame=\s*(\d+)", err)[-1])
    return frames, ffmpeg.read_audio(str(path), 48000).size


# 1. chapter titles are escaped in the FFMETADATA file
def test_chapter_title_escaped(video, tmp_path):
    title = r"a=b;c#d\e"
    meta = _chapters_meta([(0.0, title), (10.0, "two\nlines")], 30.0, tmp_path / "meta.txt")
    assert r"title=a\=b\;c\#d\\e" in meta.read_text()
    out = tmp_path / "ch.mp4"
    ffmpeg.run(["-i", str(video), "-i", str(meta), "-map", "0", "-map_chapters", "1", "-c", "copy", str(out)])
    err = _ff("-i", str(out))
    assert title in err and err.count("Chapter #") == 2


# 2. a subtitle starting inside a cut keeps its surviving part in the .srt (like the burned-in one)
def test_srt_keeps_subtitle_starting_in_cut():
    p = EditPlan(source="x", duration=30.0)
    p.add(Cut(start=0.5, end=2.0), Cut(start=10, end=12),
          Subtitle(start=1, end=3, text="partly cut"), Subtitle(start=10.5, end=11.5, text="fully cut"))
    assert srt_items(p) == [(0.5, 1.5, "partly cut")]


# 3. chapters_from_anchors is idempotent, spaced on the output timeline, first chapter at 0:00 in the MP4
def test_chapters_from_anchors_idempotent_and_output_spacing():
    a = AnchorFile(video="x", duration=100.0)
    a.add_events([Event(t=12, kind="scene_change"), Event(t=30, kind="scene_change"), Event(t=60, kind="scene_change")])
    p = EditPlan(source="x", duration=100.0)
    p.add(Chapter(t=80, title="Manual"), Cut(start=0, end=5))
    assert p.chapters_from_anchors(a) == 3  # 12 s -> 7 s after the cut: too close to 0:00
    assert p.chapters_from_anchors(a, ["Intro", "A", "B"]) == 3
    ch = p.of(Chapter)
    assert [c.title for c in ch] == ["Manual", "Intro", "A", "B"]
    assert [c.t for c in ch if c.auto] == [0.0, 30, 60]


def test_mp4_chapters_start_at_zero():
    p = EditPlan(source="x", duration=60.0)
    p.add(Chapter(t=5, title="First"), Chapter(t=20, title="Second"), Chapter(t=21, title="Same"),
          Cut(start=20, end=21), Chapter(t=59.9999, title="At end"))
    assert _output_chapters(p, p.output_duration()) == [(0.0, "First"), (20.0, "Second")]


# 4. model pre-pass: decoder failures are reported and both ffmpeg processes are always reaped
def test_apply_models_decoder_failure_and_cleanup(tmp_path, monkeypatch):
    import vidai.lab as lab_mod

    class Boom:
        def transform_frame(self, frame, t=0.0, **_):
            raise RuntimeError("model crashed")

    monkeypatch.setattr(lab_mod, "load_model", lambda name: (Boom(), None))
    src = tmp_path / "src.mp4"
    ffmpeg.run(["-f", "lavfi", "-i", "testsrc2=s=64x36:r=30:d=2", "-pix_fmt", "yuv420p", str(src)])
    procs = []
    real = subprocess.Popen

    def popen(*a, **k):
        procs.append(real(*a, **k))
        return procs[-1]

    monkeypatch.setattr(render_mod.subprocess, "Popen", popen)
    op = render_mod.ApplyModel(name="boom")
    with pytest.raises(ffmpeg.FFmpegError, match="decode failed"):
        render_mod._apply_models(str(tmp_path / "missing.mp4"), 0, 2, [op], 30, 64, 36, tmp_path / "o.mkv", False)
    with pytest.raises(RuntimeError, match="model crashed"):
        render_mod._apply_models(str(src), 0, 2, [op], 30, 64, 36, tmp_path / "o.mkv", False)
    assert len(procs) == 4 and all(p.returncode is not None for p in procs)


# 5. many cuts: audio and video stay in sync (cuts on the frame grid, sample-exact audio)
def test_many_cuts_no_av_drift(video, tmp_path):
    p = new_plan(video)
    for i in range(75):
        a = 0.3 + i * 0.37
        p.add(Cut(start=round(a, 3), end=round(a + 0.137, 3)))
    fps = 30.0
    r = render(p, tmp_path / "many.mp4", workers=2)
    frames, samples = _frames_and_samples(r.output)
    expected = sum(b - a for a, b in snap_ranges(p.keep_ranges(), fps))
    v, a = frames / fps, samples / 48000
    assert abs(v - a) <= 1 / fps
    assert abs(v - expected) <= 1 / fps and abs(a - expected) <= 1 / fps


# 6. cuts past the end are clamped: the output is never longer than the source
def test_keep_ranges_clamped_to_source():
    p = EditPlan(source="x", duration=30.0)
    p.add(Cut(start=40, end=50))
    assert p.keep_ranges() == [(0.0, 30.0)] and p.output_duration() == 30.0
    p.add(Cut(start=25, end=45), Cut(start=-5, end=-1))
    assert p.keep_ranges() == [(0.0, 25.0)]
    assert EditPlan(source="x", duration=10.0004).add(Cut(start=20, end=30)).keep_ranges()[0][1] <= 10.0004


# 7. zoom crop: never 0 px, even sizes, output aspect kept
@pytest.mark.parametrize("z", [Zoom(start=0, end=1, x=0.5, y=0.5, w=0.0001, h=0.0001),
                               Zoom(start=0, end=1, x=0.0, y=0.0, w=0.5, h=0.1),
                               Zoom(start=0, end=1, x=0.9, y=0.9, w=0.1, h=0.9),
                               Zoom(start=0, end=1, x=0.0, y=0.0, w=1.0, h=1.0)])
def test_zoom_crop(z):
    W, H = 640, 360
    cw, ch, cx, cy = _zoom_crop(z, W, H)
    assert cw >= 2 and ch >= 2 and cw % 2 == 0 and ch % 2 == 0
    assert 0 <= cx <= W - cw and 0 <= cy <= H - ch
    assert cw / ch == pytest.approx(W / H, rel=0.02) or cw < 20


def test_tiny_zoom_renders(video, tmp_path):
    p = new_plan(video)
    p.add(Cut(start=4, end=30), Zoom(start=1, end=2, x=0.2, y=0.2, w=0.001, h=0.3))
    assert render(p, tmp_path / "z.mp4", workers=1).duration == pytest.approx(4.0, abs=0.1)


# 8. probe copes with "Duration: N/A" (e.g. a recording that crashed)
def test_probe_duration_na(tmp_path):
    path = tmp_path / "crashed.mkv"
    proc = subprocess.run([ffmpeg.ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                           "testsrc2=s=160x90:r=30:d=4", "-f", "lavfi", "-i", "sine=d=4", "-c:v", "libx264",
                           "-c:a", "aac", "-f", "matroska", "-"], capture_output=True)
    path.write_bytes(proc.stdout)
    assert "Duration: N/A" in _ff("-i", str(path))
    info = ffmpeg.probe(str(path))
    assert info.duration == pytest.approx(4.0, abs=0.2) and info.width == 160 and info.has_audio
    empty = tmp_path / "empty.mkv"
    empty.write_bytes(proc.stdout[:300])
    with pytest.raises(ffmpeg.FFmpegError):
        ffmpeg.probe(str(empty))


# 9. native wrappers validate their inputs; the .so cache key includes the CPU
def test_native_validation():
    f = np.zeros((10, 12, 3), np.uint8)
    with pytest.raises(ValueError, match="overlay"):
        native.alpha_blend(f, np.zeros((10, 12, 3), np.uint8))
    with pytest.raises(ValueError, match="frame"):
        native.alpha_blend(np.zeros((10, 12, 4), np.uint8), np.zeros((10, 12, 4), np.uint8))
    with pytest.raises(ValueError, match="frame"):
        native.alpha_blend(f.astype(np.float32), np.zeros((10, 12, 4), np.uint8))
    native.alpha_blend(f, np.full((50, 50, 4), 255, np.uint8), -20, -20)  # larger overlay: clipped ROI
    assert (f == 255).all()
    with pytest.raises(ValueError, match="background"):
        native.mask_blend(f, np.zeros((5, 5, 3), np.uint8), np.zeros((10, 12), np.uint8))
    with pytest.raises(ValueError, match="mask"):
        native.mask_blend(f, np.zeros_like(f), np.zeros((10, 11), np.uint8))
    native.mask_blend(f, np.zeros_like(f), np.zeros((10, 12, 1), np.uint8))
    assert (f == 0).all()
    with pytest.raises(ValueError, match="T"):
        native.lut3x3(f, np.zeros((3, 256), np.float32))
    with pytest.raises(ValueError, match="frame"):
        native.lut3x3(np.zeros((4, 4), np.uint8), np.zeros((3, 3, 256), np.float32))
    with pytest.raises(ValueError):
        native.rms_db(np.zeros(100, np.float32), 0)
    if native.AVAILABLE:
        with pytest.raises(ValueError, match="W"):
            native.affine_color(f, np.zeros((7, 3), np.float32), degree=1)
        with pytest.raises(ValueError, match="frame"):
            native.affine_color(np.zeros((4, 4, 4), np.uint8), np.zeros((4, 3), np.float32))


@pytest.mark.skipif(not native.AVAILABLE, reason="no C compiler")
def test_native_cache_key_includes_cpu(vidai_home, monkeypatch):
    import platform

    assert native._cpu_id().startswith(platform.machine().encode())
    assert native._build() is not None
    monkeypatch.setattr(native, "_cpu_id", lambda: b"other-cpu")
    assert native._build() is not None
    assert len(list((vidai_home / "native").glob("fastops_*.so"))) == 2


# 10. chunked analysis: series are on the absolute time grid (no drift at chunk borders)
def test_analysis_chunks_do_not_drift(monkeypatch):
    def fake_audio(path, sr=16000, start=0.0, duration=None):
        t = start + np.arange(int(round(duration * sr))) / sr
        block = np.floor(t * 10 + 1e-6).astype(int)
        return (10 ** (-(block % 40) / 20) * np.where(np.arange(t.size) % 2, 1, -1)).astype(np.float32)

    def fake_frames(path, fps=2.0, width=160, height=90, start=0.0, duration=None, keyframes_only=False):
        idx = np.round(start * fps) + np.arange(int(round(duration * fps)))
        return np.broadcast_to(((idx * 37) % 256).astype(np.uint8)[:, None, None], (idx.size, 4, 4)).copy()

    monkeypatch.setattr(ffmpeg, "read_audio", fake_audio)
    monkeypatch.setattr(ffmpeg, "read_frames", fake_frames)
    dur = 73.37
    one, many = analyze_mod.audio_level("x", dur, 1), analyze_mod.audio_level("x", dur, 7)
    assert len(one.values) == len(many.values) == round(dur * 10)
    assert np.allclose(one.values, many.values, atol=0.11)
    m1, m7 = analyze_mod.motion("x", dur, 1), analyze_mod.motion("x", dur, 7)
    assert len(m1.values) == len(m7.values) == round(dur * 2)
    assert np.allclose(m1.values, m7.values, atol=1e-3)
