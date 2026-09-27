"""Anchors: small stats recorded with a video so Claude can jump straight to the moments that matter.

An anchor file (`<video>.anchors.json`) holds only the stats that Claude selected for this
recording (see `select_stats`). It stores three kinds of data:

- series:   regularly sampled values (e.g. audio level at 10 Hz)
- segments: time ranges (e.g. silence, speech)
- events:   instants (e.g. scene cut, user marker, window change)

All times are seconds in the *source* video timeline.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from .brief import Brief

FORMAT_VERSION = 1


@dataclass(frozen=True)
class StatInfo:
    name: str
    description: str
    source: str  # "live" (captured while recording) or "analysis" (computed from the file)
    kind: str  # "series" | "segments" | "events"
    available: bool = True  # False = planned, needs extra dependencies


STATS: dict[str, StatInfo] = {s.name: s for s in [
    StatInfo("audio_level", "Audio RMS level in dBFS at 10 Hz", "analysis", "series"),
    StatInfo("silence", "Silent ranges (gaps, pauses)", "analysis", "segments"),
    StatInfo("motion", "Visual change between frames at 2 Hz (0-1)", "analysis", "series"),
    StatInfo("scene_change", "Hard visual cuts (slide change, window switch)", "analysis", "events"),
    StatInfo("markers", "Hotkey markers pressed by the user (marker/mistake/section)", "live", "events"),
    StatInfo("input_activity", "Keyboard and mouse events per second", "live", "series"),
    StatInfo("window_focus", "Focused window/app changes", "live", "events"),
    StatInfo("obs_scene", "OBS scene switches", "live", "events"),
    StatInfo("speech", "Speech segments with transcript (Whisper)", "analysis", "segments", available=False),
    StatInfo("face", "Face presence and position", "analysis", "series", available=False),
    StatInfo("eye_state", "Eyes open/closed (blinks)", "analysis", "series", available=False),
]}


def list_stats(include_unavailable: bool = False) -> list[StatInfo]:
    return [s for s in STATS.values() if s.available or include_unavailable]


def select_stats(brief: Brief) -> dict[str, str]:
    """Default stat selection from the brief: {stat: reason}.

    This is a starting point. Claude reads the brief and may add or drop stats,
    then passes its own choice to the recorder/analyzer.
    """
    chosen: dict[str, str] = {
        "audio_level": "base signal for pauses, loudness and safe split points",
        "silence": "find gaps to remove and safe points to split for parallel processing",
        "markers": "the user's own marks are the strongest editing hints",
    }
    if brief.uses_screen:
        chosen["scene_change"] = "slide/window changes are natural section and chapter boundaries"
        chosen["input_activity"] = "typing/clicking shows where the action is (zoom candidates)"
        chosen["window_focus"] = "app switches help label what is on screen"
    if brief.uses_camera:
        chosen["motion"] = "detect camera movement and still parts"
        if STATS["face"].available:
            chosen["face"] = "keep the speaker framed"
        if STATS["eye_state"].available:
            chosen["eye_state"] = "find blinks/closed eyes"
    if brief.style in ("slides",):
        chosen["scene_change"] = "each slide change is a chapter candidate"
    if "chapters" in brief.extras and "scene_change" not in chosen:
        chosen["scene_change"] = "chapter candidates"
    if "subtitles" in brief.extras and STATS["speech"].available:
        chosen["speech"] = "subtitles"
    return chosen


class Series(BaseModel):
    rate_hz: float
    values: list[float]

    def array(self) -> np.ndarray:
        return np.asarray(self.values, dtype=np.float32)

    def value_at(self, t: float) -> float | None:
        i = int(t * self.rate_hz)
        return self.values[i] if 0 <= i < len(self.values) else None

    def window(self, t0: float, t1: float) -> np.ndarray:
        a = self.array()
        return a[max(0, int(t0 * self.rate_hz)): max(0, int(np.ceil(t1 * self.rate_hz)))]


class Segment(BaseModel):
    start: float
    end: float
    kind: str
    data: dict[str, Any] = Field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end - self.start


class Event(BaseModel):
    t: float
    kind: str
    data: dict[str, Any] = Field(default_factory=dict)


class AnchorFile(BaseModel):
    version: int = FORMAT_VERSION
    video: str = ""
    duration: float = 0.0
    recorded_at: str = ""
    brief: Brief | None = None
    selected_stats: dict[str, str] = Field(default_factory=dict)
    series: dict[str, Series] = Field(default_factory=dict)
    segments: list[Segment] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)

    # ---------- io ----------
    @staticmethod
    def path_for(video: str | Path) -> Path:
        video = Path(video)
        return video.with_name(video.name + ".anchors.json")

    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path) if path else self.path_for(self.video)
        path.write_text(self.model_dump_json(indent=1, exclude_none=True), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "AnchorFile":
        path = Path(path)
        if not path.name.endswith(".anchors.json"):
            path = cls.path_for(path)
        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))

    # ---------- queries ----------
    def segments_of(self, kind: str, t0: float = 0.0, t1: float | None = None) -> list[Segment]:
        t1 = self.duration if t1 is None else t1
        return [s for s in self.segments if s.kind == kind and s.end > t0 and s.start < t1]

    def events_of(self, kind: str | None = None, t0: float = 0.0, t1: float | None = None) -> list[Event]:
        t1 = self.duration if t1 is None else t1
        return [e for e in self.events if (kind is None or e.kind == kind) and t0 <= e.t <= t1]

    def add_segments(self, segs: list[Segment]) -> None:
        kinds = {s.kind for s in segs}
        self.segments = [s for s in self.segments if s.kind not in kinds] + segs
        self.segments.sort(key=lambda s: s.start)

    def add_events(self, evs: list[Event], replace_kinds: bool = True) -> None:
        if replace_kinds:
            kinds = {e.kind for e in evs}
            self.events = [e for e in self.events if e.kind not in kinds]
        self.events = sorted(self.events + evs, key=lambda e: e.t)

    def at(self, t: float, radius: float = 2.0) -> dict[str, Any]:
        """Everything known around time t (what Claude asks when inspecting a moment)."""
        out: dict[str, Any] = {"t": t}
        for name, s in self.series.items():
            out[name] = s.value_at(t)
        out["segments"] = [s.model_dump() for s in self.segments if s.start <= t + radius and s.end >= t - radius]
        out["events"] = [e.model_dump() for e in self.events if abs(e.t - t) <= radius]
        return out

    def summary(self) -> dict[str, Any]:
        """Compact overview for Claude: counts, totals and the most useful anchors."""
        out: dict[str, Any] = {
            "video": self.video,
            "duration": round(self.duration, 2),
            "selected_stats": list(self.selected_stats),
            "series": {k: {"rate_hz": v.rate_hz, "n": len(v.values),
                           "mean": round(float(np.mean(v.values)), 3) if v.values else None}
                       for k, v in self.series.items()},
        }
        kinds = sorted({s.kind for s in self.segments})
        out["segments"] = {k: {"count": len(ss := self.segments_of(k)),
                               "total_s": round(sum(s.duration for s in ss), 2)} for k in kinds}
        ekinds = sorted({e.kind for e in self.events})
        out["events"] = {k: [round(e.t, 2) for e in self.events_of(k)][:50] for k in ekinds}
        if self.brief:
            out["brief"] = {"title": self.brief.title, "language": self.brief.language,
                            "style": self.brief.style, "outline": self.brief.outline}
        return out

    # ---------- parallel processing ----------
    def split_points(self, n: int, min_chunk: float = 5.0, search: float | None = None) -> list[float]:
        """Choose n-1 split times near equal fractions of the video, snapped to safe anchors.

        Safe = middle of a silence (never cuts speech), else a scene change, else the exact target.
        """
        if n <= 1 or self.duration < 2 * min_chunk:
            return []
        n = min(n, int(self.duration // min_chunk))
        search = search if search is not None else self.duration / n / 2
        silences = self.segments_of("silence")
        cuts = [e.t for e in self.events_of("scene_change")]
        points: list[float] = []
        for k in range(1, n):
            target = self.duration * k / n
            best = None
            cands = [((s.start + s.end) / 2, abs((s.start + s.end) / 2 - target) - s.duration)
                     for s in silences if abs((s.start + s.end) / 2 - target) <= search]
            if cands:
                best = min(cands, key=lambda c: c[1])[0]
            else:
                near = [c for c in cuts if abs(c - target) <= search]
                best = min(near, key=lambda c: abs(c - target)) if near else target
            lo = (points[-1] if points else 0.0) + min_chunk
            if lo <= best <= self.duration - min_chunk:
                points.append(round(best, 3))
        return points

    def chunks(self, n: int, min_chunk: float = 5.0) -> list[tuple[float, float]]:
        pts = [0.0] + self.split_points(n, min_chunk) + [self.duration]
        return list(zip(pts[:-1], pts[1:]))
