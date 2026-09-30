"""Non-destructive edit plan. All times are source-video seconds (same timeline as anchors).

Claude builds a plan (JSON), inspects it, and renders it at the end. The source video is never modified.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

from .anchors import AnchorFile

Position = Union[str, tuple[float, float]]  # named ("top-left", "center", ...) or (x, y) fractions 0-1


class Cut(BaseModel):
    op: Literal["cut"] = "cut"
    start: float
    end: float
    reason: str = ""


class Text(BaseModel):
    op: Literal["text"] = "text"
    start: float
    end: float
    text: str
    position: Position = "bottom-center"
    size: float = 0.05  # font height as a fraction of video height
    color: str = "#FFFFFF"
    box: str | None = "#000000A0"  # background box color (RGBA hex) or None
    font: str | None = None


class Image(BaseModel):
    op: Literal["image"] = "image"
    start: float
    end: float
    path: str
    position: Position = "top-right"
    width: float = 0.12  # fraction of video width


class Shape(BaseModel):
    """Directions and highlights: arrow, circle, box drawn over the video."""
    op: Literal["shape"] = "shape"
    start: float
    end: float
    shape: Literal["arrow", "circle", "box"] = "arrow"
    x: float = 0.5  # target point / center (fractions)
    y: float = 0.5
    w: float = 0.15  # size (fractions)
    h: float = 0.15
    angle: float = 225.0  # arrow: direction it comes from, degrees (225 = from bottom-left... pointing up-right)
    color: str = "#FF3B30"
    thickness: float = 0.008


class Zoom(BaseModel):
    op: Literal["zoom"] = "zoom"
    start: float
    end: float
    x: float = 0.0  # region top-left (fractions)
    y: float = 0.0
    w: float = 0.5
    h: float = 0.5


class Audio(BaseModel):
    op: Literal["audio"] = "audio"
    filter: Literal["denoise", "loudnorm", "volume", "highpass"] = "loudnorm"
    value: float | None = None  # volume gain (x), highpass Hz, denoise strength dB


class Subtitle(BaseModel):
    op: Literal["subtitle"] = "subtitle"
    start: float
    end: float
    text: str


class Chapter(BaseModel):
    op: Literal["chapter"] = "chapter"
    t: float
    title: str
    auto: bool = False  # made by chapters_from_anchors (replaced when it runs again)


class ApplyModel(BaseModel):
    """Apply a custom lab model (frame transform) to a time range."""
    op: Literal["model"] = "model"
    name: str
    start: float = 0.0
    end: float | None = None
    params: dict = Field(default_factory=dict)


Op = Annotated[Union[Cut, Text, Image, Shape, Zoom, Audio, Subtitle, Chapter, ApplyModel], Field(discriminator="op")]


class EditPlan(BaseModel):
    source: str
    duration: float = 0.0
    ops: list[Op] = Field(default_factory=list)
    burn_subtitles: bool = False
    notes: list[str] = Field(default_factory=list)  # Claude's reasoning log

    # ---------- io ----------
    @staticmethod
    def path_for(video: str | Path) -> Path:
        video = Path(video)
        return video.with_name(video.name + ".edit.json")

    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path) if path else self.path_for(self.source)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(self.model_dump_json(indent=1), encoding="utf-8")
        os.replace(tmp, path)  # readers never see a half-written plan
        return path

    @classmethod
    def load(cls, path: str | Path) -> "EditPlan":
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))

    # ---------- building ----------
    def add(self, *ops: Op | dict) -> "EditPlan":
        from pydantic import TypeAdapter

        ta = TypeAdapter(Op)
        for o in ops:
            self.ops.append(ta.validate_python(o) if isinstance(o, dict) else o)
        return self

    def of(self, op_type: type) -> list:
        return [o for o in self.ops if isinstance(o, op_type)]

    def remove_gaps(self, anchors: AnchorFile, min_gap: float = 0.8, keep: float = 0.3) -> int:
        """Cut silent gaps longer than min_gap, leaving `keep` seconds of pause (split around the gap)."""
        n = 0
        for s in anchors.segments_of("silence"):
            if s.duration < min_gap:
                continue
            a = s.start + keep / 2 if s.start > 0 else 0.0
            b = s.end - keep / 2 if s.end < anchors.duration - 0.05 else anchors.duration
            if b - a > 0.05:
                self.ops.append(Cut(start=round(a, 3), end=round(b, 3), reason=f"gap {s.duration:.1f}s"))
                n += 1
        return n

    def cut_mistakes(self, anchors: AnchorFile, max_take: float = 30.0) -> int:
        """For each 'mistake' marker, cut the failed take: from the end of the last pause before it
        up to the marker (the user then pauses and says it again)."""
        n = 0
        for m in anchors.events_of("markers"):
            if m.data.get("type") != "mistake":
                continue
            prev = [s.end for s in anchors.segments_of("silence", 0, m.t) if s.end <= m.t - 0.3]
            start = max(prev[-1] if prev else 0.0, m.t - max_take)
            nxt = [s for s in anchors.segments_of("silence", m.t) if s.start >= m.t - 0.3]
            end = (nxt[0].start + nxt[0].end) / 2 if nxt else m.t
            self.ops.append(Cut(start=round(start, 3), end=round(end, 3), reason="mistake marker"))
            n += 1
        return n

    def chapters_from_anchors(self, anchors: AnchorFile, titles: list[str] | None = None, min_len: float = 10.0) -> int:
        """Chapter candidates from user section markers, then scene changes. Replaces the chapters a previous
        call made; the min_len spacing is measured on the output timeline (after the cuts made so far)."""
        ts = [e.t for e in anchors.events_of("markers") if e.data.get("type") == "section"]
        if not ts:
            ts = [e.t for e in anchors.events_of("scene_change")]
        self.ops = [o for o in self.ops if not (isinstance(o, Chapter) and o.auto)]
        dur = self.duration or anchors.duration
        timed = bool(self.duration)
        out = self.map_time_after if timed else (lambda t: t)
        total = self.output_duration() if timed else dur
        picked = [0.0]
        for t in sorted(ts):
            if 0 < t < dur and out(t) - out(picked[-1]) >= min_len and total - out(t) >= min_len:
                picked.append(t)
        for i, t in enumerate(picked):
            title = titles[i] if titles and i < len(titles) else ("Intro" if i == 0 else f"Part {i}")
            self.ops.append(Chapter(t=round(t, 3), title=title, auto=True))
        return len(picked)

    # ---------- timeline ----------
    def keep_ranges(self) -> list[tuple[float, float]]:
        d = self.duration
        cuts = sorted(r for c in self.of(Cut) if (r := (min(max(0.0, c.start), d), min(max(0.0, c.end), d)))[1] > r[0])
        merged: list[list[float]] = []
        for a, b in cuts:
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        keeps, t = [], 0.0
        for a, b in merged:
            if a > t:
                keeps.append((t, a))
            t = max(t, b)
        if t < self.duration:
            keeps.append((t, self.duration))
        return [(round(a, 3), min(round(b, 3), d)) for a, b in keeps if b - a > 1e-3]

    def output_duration(self) -> float:
        return sum(b - a for a, b in self.keep_ranges())

    def map_time(self, t: float) -> float | None:
        """Source time -> output time (None if t falls inside a cut)."""
        out = 0.0
        for a, b in self.keep_ranges():
            if t < a:
                return None
            if t <= b:
                return out + (t - a)
            out += b - a
        return None

    def map_time_after(self, t: float) -> float:
        """Like map_time, but a time inside a cut snaps to the next kept moment."""
        out = 0.0
        for a, b in self.keep_ranges():
            if t <= b:
                return out + max(0.0, t - a)
            out += b - a
        return out

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for o in self.ops:
            counts[o.op] = counts.get(o.op, 0) + 1
        return {"source": self.source, "source_duration": round(self.duration, 2),
                "output_duration": round(self.output_duration(), 2), "ops": counts,
                "keep_ranges": len(self.keep_ranges())}


def new_plan(video: str | Path) -> EditPlan:
    from .ffmpeg import probe

    return EditPlan(source=str(video), duration=probe(str(video)).duration)
