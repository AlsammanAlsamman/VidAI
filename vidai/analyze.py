"""Compute analysis stats from a video file, in parallel chunks."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import ffmpeg, native
from .anchors import STATS, AnchorFile, Event, Segment, Series

AUDIO_RATE_HZ = 10.0
VIDEO_RATE_HZ = 2.0
_SR = 16000


def _time_chunks(duration: float, workers: int, min_chunk: float = 10.0) -> list[tuple[float, float]]:
    n = max(1, min(workers, int(duration // min_chunk) or 1))
    edges = np.linspace(0.0, duration, n + 1)
    return list(zip(edges[:-1], edges[1:]))


def _fit(a: np.ndarray, n: int, fill: float) -> np.ndarray:
    if a.size >= n:
        return a[:n]
    return np.concatenate([a, np.full(n - a.size, fill, dtype=np.float32)])


def audio_level(path: str, duration: float, workers: int = 3) -> Series:
    hop = int(_SR / AUDIO_RATE_HZ)

    def work(c: tuple[float, float]) -> np.ndarray:
        s, e = c
        n = int(round((e - s) * AUDIO_RATE_HZ))
        pcm = ffmpeg.read_audio(path, _SR, start=s, duration=e - s)
        return _fit(native.rms_db(pcm, hop), n, -90.0)

    with ThreadPoolExecutor(workers) as ex:
        parts = list(ex.map(work, _time_chunks(duration, workers)))
    vals = np.clip(np.concatenate(parts), -90, 0)
    return Series(rate_hz=AUDIO_RATE_HZ, values=[round(float(v), 1) for v in vals])


def silence_from_level(level: Series, threshold_db: float | None = None, min_duration: float = 0.5) -> list[Segment]:
    """Silent ranges. With threshold_db=None the threshold adapts to the recording's noise floor."""
    a = level.array()
    if a.size == 0:
        return []
    if threshold_db is None:
        floor = float(np.percentile(a, 10))
        loud = float(np.percentile(a, 90))
        threshold_db = min(floor + 0.35 * (loud - floor), -30.0) if loud - floor > 10 else -45.0
    dt = 1.0 / level.rate_hz
    runs = native.find_runs(a, threshold_db, below=True, min_len=int(np.ceil(min_duration / dt - 1e-9)))
    return [Segment(start=round(i * dt, 3), end=round(j * dt, 3), kind="silence",
                    data={"threshold_db": round(threshold_db, 1)}) for i, j in runs]


def motion(path: str, duration: float, workers: int = 3, fast: bool = False) -> Series:
    step = 1.0 / VIDEO_RATE_HZ

    def work(c: tuple[float, float]) -> np.ndarray:
        s, e = c
        n = int(round((e - s) * VIDEO_RATE_HZ))
        pre = step if s > 0 else 0.0  # one frame overlap so cuts at chunk edges are seen
        fr = ffmpeg.read_frames(path, VIDEO_RATE_HZ, start=s - pre, duration=e - s + pre, keyframes_only=fast)
        d = native.frame_mad(fr)  # d[0] = 0, d[i] = |frame i - frame i-1|
        if pre:
            d = d[1:]
        return _fit(d, n, 0.0)

    with ThreadPoolExecutor(workers) as ex:
        parts = list(ex.map(work, _time_chunks(duration, workers)))
    return Series(rate_hz=VIDEO_RATE_HZ, values=[round(float(v), 4) for v in np.concatenate(parts)])


def scene_changes(mot: Series, threshold: float = 0.12) -> list[Event]:
    a = mot.array()
    evs = []
    for i in range(a.size):
        if a[i] >= threshold and a[i] >= a[max(0, i - 1)] and (i + 1 >= a.size or a[i] >= a[i + 1]):
            evs.append(Event(t=round(i / mot.rate_hz, 3), kind="scene_change", data={"score": round(float(a[i]), 3)}))
    return evs


def analyze(video: str | Path, stats: list[str] | dict[str, str] | None = None, workers: int = 3,
            anchors: AnchorFile | None = None, silence_db: float | None = None,
            min_silence: float = 0.5, fast: bool = False) -> AnchorFile:
    """Compute analysis stats and merge them into the video's anchor file (created if missing).

    fast=True decodes only keyframes for the visual stats (~10x faster on long videos)."""
    video = str(video)
    info = ffmpeg.probe(video)
    if anchors is None:
        p = AnchorFile.path_for(video)
        anchors = AnchorFile.load(p) if p.exists() else AnchorFile(video=video)
    anchors.video = video
    anchors.duration = info.duration
    if stats is None:
        stats = anchors.selected_stats or {"audio_level": "", "silence": "", "motion": "", "scene_change": ""}
    wanted = list(stats)
    for s in wanted:
        if s in STATS and s not in anchors.selected_stats:
            anchors.selected_stats[s] = stats[s] if isinstance(stats, dict) else ""

    need_audio = info.has_audio and ({"audio_level", "silence"} & set(wanted))
    need_video = info.has_video and ({"motion", "scene_change"} & set(wanted))

    with ThreadPoolExecutor(2) as ex:  # audio and video analysis side by side
        fa = ex.submit(audio_level, video, info.duration, workers) if need_audio else None
        fv = ex.submit(motion, video, info.duration, workers, fast) if need_video else None
        level = fa.result() if fa else None
        mot = fv.result() if fv else None

    if level is not None:
        if "audio_level" in wanted:
            anchors.series["audio_level"] = level
        if "silence" in wanted:
            anchors.add_segments(silence_from_level(level, silence_db, min_silence))
    if mot is not None:
        if "motion" in wanted:
            anchors.series["motion"] = mot
        if "scene_change" in wanted:
            anchors.add_events(scene_changes(mot))
    return anchors
