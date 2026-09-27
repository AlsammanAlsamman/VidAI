"""YouTube export helpers: chapters text, description draft, upload checklist."""
from __future__ import annotations

from pathlib import Path

from .brief import Brief
from .render import RenderResult


def fmt_ts(t: float) -> str:
    t = int(t)
    h, m, s = t // 3600, t // 60 % 60, t % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def youtube_chapters(chapters: list[tuple[float, str]]) -> tuple[str, list[str]]:
    """Chapters block for the description, plus warnings for YouTube's rules
    (first at 0:00, at least 3, each at least 10 s)."""
    warnings = []
    if not chapters:
        return "", ["no chapters"]
    if chapters[0][0] > 0.5:
        warnings.append("first chapter must start at 0:00 (fixed)")
        chapters = [(0.0, chapters[0][1])] + chapters[1:]
    if len(chapters) < 3:
        warnings.append("YouTube needs at least 3 chapters to show them")
    for (a, _), (b, t) in zip(chapters, chapters[1:]):
        if b - a < 10:
            warnings.append(f"chapter '{t}' starts less than 10 s after the previous one")
    return "\n".join(f"{fmt_ts(t)} {title}" for t, title in chapters), warnings


def youtube_package(result: RenderResult, brief: Brief | None = None) -> dict:
    """Write <output>.youtube.txt with title, description and chapters; return the info."""
    chapters_txt, warnings = youtube_chapters(result.chapters)
    title = (brief.title if brief and brief.title else Path(result.output).stem)
    desc = []
    if brief and brief.topic:
        desc.append(brief.topic)
    if chapters_txt:
        desc += ["", "Chapters:", chapters_txt]
    text = f"TITLE:\n{title}\n\nDESCRIPTION:\n" + "\n".join(desc) + "\n"
    if result.srt:
        text += f"\nSUBTITLES FILE: {result.srt}\n"
    path = Path(result.output).with_suffix(".youtube.txt")
    path.write_text(text, encoding="utf-8")
    return {"video": result.output, "duration": round(result.duration, 2), "package": str(path),
            "srt": result.srt, "chapters": chapters_txt, "warnings": warnings}
