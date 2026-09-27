"""Pre-recording brief: what Claude asks the user before a recording starts."""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

VIDEO_STYLES = ("screencast", "talking_head", "screencast_with_camera", "slides", "vlog", "other")

# The questions Claude asks. Keys match Brief fields.
QUESTIONS: dict[str, str] = {
    "title": "What is the working title of the video?",
    "topic": "What is the video about?",
    "language": "Which language(s) will you speak? (e.g. ar, en, ar+en)",
    "audience": "Who is the audience and what is their level?",
    "style": f"Which style? One of: {', '.join(VIDEO_STYLES)}",
    "expected_minutes": "How long do you expect the recording to be (minutes)?",
    "outline": "What are the planned sections, in order?",
    "extras": "Which extras do you want? (subtitles, intro, outro, logo, music, chapters, zooms, arrows)",
    "notes": "Anything else Claude should know? (things to hide, names, terms)",
}


class Brief(BaseModel):
    title: str = ""
    topic: str = ""
    language: str = "en"
    audience: str = ""
    style: str = "screencast"
    expected_minutes: float = 10.0
    outline: list[str] = Field(default_factory=list)
    extras: list[str] = Field(default_factory=list)
    notes: str = ""

    @property
    def languages(self) -> list[str]:
        return [x.strip() for x in self.language.replace(",", "+").split("+") if x.strip()]

    @property
    def uses_camera(self) -> bool:
        return self.style in ("talking_head", "screencast_with_camera", "vlog")

    @property
    def uses_screen(self) -> bool:
        return self.style in ("screencast", "screencast_with_camera", "slides")

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Brief":
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def ask_interactive(input_fn=input) -> Brief:
    """Terminal questionnaire (Claude normally fills the brief itself from the conversation)."""
    data: dict = {}
    for key, q in QUESTIONS.items():
        ans = input_fn(f"{q}\n> ").strip()
        if not ans:
            continue
        if key == "expected_minutes":
            try:
                data[key] = float(ans)
            except ValueError:
                continue
        elif key in ("outline", "extras"):
            sep = ";" if ";" in ans else ","
            data[key] = [x.strip() for x in ans.split(sep) if x.strip()]
        else:
            data[key] = ans
    return Brief(**data)
