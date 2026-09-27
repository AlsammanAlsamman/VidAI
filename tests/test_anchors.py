from vidai import AnchorFile, Brief, Event, Segment, select_stats


def test_select_stats_depends_on_brief():
    screen = select_stats(Brief(style="screencast"))
    cam = select_stats(Brief(style="talking_head"))
    assert {"silence", "markers", "scene_change", "input_activity"} <= set(screen)
    assert "motion" in cam and "input_activity" not in cam
    assert all(isinstance(r, str) and r for r in screen.values())


def _anchors():
    a = AnchorFile(video="x.mp4", duration=60.0)
    a.add_segments([Segment(start=18.0, end=21.0, kind="silence"), Segment(start=41.0, end=42.0, kind="silence")])
    a.add_events([Event(t=30.5, kind="scene_change")])
    return a


def test_split_points_snap_to_silence():
    a = _anchors()
    assert a.split_points(3) == [19.5, 41.5]
    chunks = a.chunks(3)
    assert chunks[0][0] == 0 and chunks[-1][1] == 60.0
    assert all(b > a_ for a_, b in chunks)


def test_split_points_fall_back_to_scene_change_then_target():
    a = AnchorFile(video="x.mp4", duration=60.0)
    a.add_events([Event(t=31.0, kind="scene_change")])
    assert a.split_points(2) == [31.0]
    assert AnchorFile(video="x", duration=60.0).split_points(2) == [30.0]
    assert AnchorFile(video="x", duration=6.0).split_points(3) == []


def test_roundtrip_and_queries(tmp_path):
    a = _anchors()
    a.brief = Brief(title="t", language="ar+en")
    p = a.save(tmp_path / "x.mp4.anchors.json")
    b = AnchorFile.load(p)
    assert b.model_dump() == a.model_dump()
    assert b.brief.languages == ["ar", "en"]
    assert len(b.segments_of("silence", 20, 50)) == 2
    at = b.at(30.0)
    assert at["events"][0]["kind"] == "scene_change"
    assert b.summary()["segments"]["silence"]["count"] == 2
