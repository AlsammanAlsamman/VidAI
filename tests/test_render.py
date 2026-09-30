import subprocess

import numpy as np
import pytest

from vidai import Audio, Chapter, ApplyModel, Shape, Subtitle, Text, Zoom, ffmpeg, lab, new_plan
from vidai.analyze import analyze
from vidai.lab.examples import ColorMatch
from vidai.render import render


def _frame(path, t):
    return ffmpeg.read_frames(str(path), fps=30, width=160, height=90, start=t, duration=0.04)[0].astype(int)


def test_parallel_render_matches_serial(video, tmp_path):
    a = analyze(video)
    p = new_plan(video)
    p.remove_gaps(a)
    p.add(Text(start=1, end=4, text="Hello"), Audio(filter="loudnorm"), Audio(filter="denoise"))
    r3 = render(p, tmp_path / "p.mp4", workers=3, anchors=a)
    r1 = render(p, tmp_path / "s.mp4", workers=1, anchors=a)
    assert len(r3.chunks) == 3 and len(r1.chunks) == 1
    # chunk borders are inside silences
    for cs, _ in r3.chunks[1:]:
        assert any(s.start < cs < s.end for s in a.segments_of("silence"))
    exp = p.output_duration()
    assert r3.duration == pytest.approx(exp, abs=0.25)
    assert r1.duration == pytest.approx(exp, abs=0.25)
    info = ffmpeg.probe(r3.output)
    assert info.has_audio and info.has_video and info.width == 640


def test_overlays_and_zoom_change_pixels(video, tmp_path):
    p = new_plan(video)
    p.add(Text(start=0, end=3, text="BIG TEXT", position="center", size=0.2, color="#FF00FF", box=None),
          Shape(start=0, end=3, shape="box", x=0.5, y=0.5, w=0.9, h=0.9, color="#00FF00", thickness=0.03),
          Zoom(start=16, end=20, x=0.0, y=0.0, w=0.25, h=0.25))
    r = render(p, tmp_path / "o.mp4", workers=2)
    src_a, out_a = _frame(video, 1.0), _frame(r.output, 1.0)
    assert np.abs(src_a - out_a).mean() > 5
    src_b, out_b = _frame(video, 22.0), _frame(r.output, 22.0)  # after zoom: unchanged
    assert np.abs(src_b - out_b).mean() < 5


def test_chapters_and_srt(video, tmp_path):
    a = analyze(video)
    p = new_plan(video)
    p.remove_gaps(a)
    p.add(Chapter(t=0, title="Intro"), Chapter(t=16, title="Blue part"), Subtitle(start=1, end=3, text="hello"),
          Subtitle(start=17, end=19, text="السلام عليكم"))
    r = render(p, tmp_path / "c.mp4", workers=3)
    err = subprocess.run([ffmpeg.ffmpeg_exe(), "-hide_banner", "-i", r.output], capture_output=True, text=True).stderr
    assert "Chapter #0" in err and "Blue part" in err
    assert r.chapters[1][0] < 16  # moved earlier by the cuts
    srt = open(r.srt, encoding="utf-8").read()
    assert "السلام عليكم" in srt and "00:00:01,000" in srt


def test_apply_lab_model(video, tmp_path):
    rng = np.random.default_rng(0)
    X = rng.integers(0, 256, (2000, 3)).astype(np.uint8)
    Y = (255 - X).astype(np.uint8)  # "invert" look
    rep = lab.train_until_suitable(ColorMatch, (X, Y), (X, Y), "mae", 1.5, higher_is_better=False, save_as="invert")
    assert rep.suitable
    p = new_plan(video)
    p.add(ApplyModel(name="invert", start=0, end=4))
    r = render(p, tmp_path / "m.mp4", workers=3)
    a, b = _frame(video, 1.0), _frame(r.output, 1.0)
    assert np.abs((255 - a) - b).mean() < 12  # inverted
    assert np.abs(_frame(video, 10.0) - _frame(r.output, 10.0)).mean() < 5  # untouched outside range
    assert r.duration == pytest.approx(30.0, abs=0.2)


def test_whole_video_cut_raises(video, tmp_path):
    p = new_plan(video)
    p.add({"op": "cut", "start": 0, "end": 30})
    with pytest.raises(ValueError):
        render(p, tmp_path / "x.mp4", workers=2)
