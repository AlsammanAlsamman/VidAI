import pytest

from vidai.analyze import analyze
from vidai.testing import DURATION, SCENE_CUT, SILENCES


def test_analyze_finds_silences_and_scene_cut(video):
    a = analyze(video, workers=3)
    assert a.duration == pytest.approx(DURATION, abs=0.1)
    sil = [(s.start, s.end) for s in a.segments_of("silence")]
    assert len(sil) == len(SILENCES)
    for (s, e), (es, ee) in zip(sil, SILENCES):
        assert s == pytest.approx(es, abs=0.2) and e == pytest.approx(ee, abs=0.2)
    cuts = [e.t for e in a.events_of("scene_change")]
    assert len(cuts) == 1 and cuts[0] == pytest.approx(SCENE_CUT, abs=0.6)
    assert len(a.series["audio_level"].values) == pytest.approx(DURATION * 10, abs=2)


def test_parallel_and_serial_analysis_agree(video):
    a1 = analyze(video, workers=1)
    a3 = analyze(video, workers=3, anchors=None)
    assert [(s.start, s.end) for s in a1.segments_of("silence")] == [(s.start, s.end) for s in a3.segments_of("silence")]
    assert [e.t for e in a1.events_of("scene_change")] == [e.t for e in a3.events_of("scene_change")]


def test_only_selected_stats(video):
    a = analyze(video, stats=["silence"])
    assert "audio_level" not in a.series and "motion" not in a.series
    assert a.segments_of("silence")
