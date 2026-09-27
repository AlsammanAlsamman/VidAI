import pytest

from vidai import AnchorFile, Chapter, Cut, EditPlan, Event, Segment, Shape, Text
from vidai.analyze import analyze


def test_keep_ranges_and_time_mapping():
    p = EditPlan(source="x.mp4", duration=20.0)
    p.add(Cut(start=5, end=8), Cut(start=7, end=9), {"op": "cut", "start": 15, "end": 25})
    assert p.keep_ranges() == [(0.0, 5.0), (9.0, 15.0)]
    assert p.output_duration() == 11.0
    assert p.map_time(4.0) == 4.0 and p.map_time(10.0) == 6.0
    assert p.map_time(6.0) is None and p.map_time_after(6.0) == 5.0


def test_plan_json_roundtrip(tmp_path):
    p = EditPlan(source="x.mp4", duration=10.0)
    p.add(Text(start=0, end=1, text="hi", position=(0.2, 0.3)), Shape(start=1, end=2, shape="circle"),
          Chapter(t=0, title="Intro"), {"op": "zoom", "start": 1, "end": 2}, {"op": "audio", "filter": "denoise"},
          {"op": "model", "name": "m"}, {"op": "subtitle", "start": 0, "end": 1, "text": "x"})
    q = EditPlan.load(p.save(tmp_path / "p.json"))
    assert q.model_dump() == p.model_dump()


def test_remove_gaps_uses_anchors(video):
    a = analyze(video)
    p = EditPlan(source=str(video), duration=a.duration)
    assert p.remove_gaps(a, min_gap=0.8, keep=0.3) == 3
    # 3 gaps of 3, 2, 3 s, each keeps 0.3 s
    assert p.output_duration() == pytest.approx(30 - (2.7 + 1.7 + 2.7), abs=0.5)


def test_cut_mistakes():
    a = AnchorFile(video="x", duration=30.0)
    a.add_segments([Segment(start=4, end=5, kind="silence"), Segment(start=10, end=12, kind="silence")])
    a.add_events([Event(t=9.8, kind="markers", data={"type": "mistake"})])
    p = EditPlan(source="x", duration=30.0)
    assert p.cut_mistakes(a) == 1
    c = p.of(Cut)[0]
    assert c.start == 5.0 and c.end == 11.0


def test_chapters_from_anchors():
    a = AnchorFile(video="x", duration=100.0)
    a.add_events([Event(t=5, kind="scene_change"), Event(t=30, kind="scene_change"), Event(t=60, kind="scene_change")])
    p = EditPlan(source="x", duration=100.0)
    assert p.chapters_from_anchors(a, ["Intro", "Setup", "Demo"]) == 3
    assert [c.t for c in p.of(Chapter)] == [0.0, 30, 60]
