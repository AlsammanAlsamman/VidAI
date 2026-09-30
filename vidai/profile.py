"""VidAI's memory of its user — it learns from every video and does not forget.

~/.vidai/profile/
    vocabulary.json   misheard word/phrase -> what the user meant (fixed before a request is understood)
    words.json        the user's own words (topics, names) -> better speech recognition
    macros.json       request phrase -> commands that worked (Claude solved it once; next time it is instant)
    prefs.json        learned defaults (sticker sizes, hand sides, capture/live settings, brief answers)
    lessons.jsonl     mistakes and what to do instead (read by Claude at the start of every session)
    history.jsonl     one line per recording (what was asked, what went wrong)
"""
from __future__ import annotations

import difflib
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()


def _norm(s: str) -> str:
    s = re.sub(r"[^\w\s؀-ۿ]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


class Profile:
    def __init__(self, root: str | Path | None = None) -> None:
        base = Path(root) if root else Path(os.environ.get("VIDAI_HOME", Path.home() / ".vidai"))
        self.dir = base / "profile"
        self.dir.mkdir(parents=True, exist_ok=True)

    # ---------- storage ----------
    def _load(self, name: str, default):
        p = self.dir / name
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return default

    def _save(self, name: str, data) -> None:
        with _LOCK:
            p = self.dir / name
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, p)

    def _append(self, name: str, row: dict) -> None:
        with _LOCK, open(self.dir / name, "a", encoding="utf-8") as f:
            f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), **row}, ensure_ascii=False) + "\n")

    def _lines(self, name: str, last: int = 50) -> list[dict]:
        p = self.dir / name
        if not p.exists():
            return []
        out = []
        for line in p.read_text(encoding="utf-8").splitlines()[-last:]:
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
        return out

    # ---------- vocabulary: fix what speech-to-text gets wrong for this user ----------
    def vocabulary(self) -> dict[str, str]:
        return self._load("vocabulary.json", {})

    def learn_correction(self, heard: str, meant: str) -> bool:
        h, m = _norm(heard), _norm(meant)
        if not h or not m or h == m:
            return False
        voc = self.vocabulary()
        voc[h] = m
        self._save("vocabulary.json", voc)
        return True

    def correct(self, text: str) -> str:
        """Apply learned corrections (whole words/phrases, longest first)."""
        voc = self.vocabulary()
        if not voc:
            return text
        t = " " + _norm(text) + " "
        changed = False
        for heard in sorted(voc, key=len, reverse=True):
            if f" {heard} " in t:
                t = t.replace(f" {heard} ", f" {voc[heard]} ")
                changed = True
        return t.strip() if changed else text

    # ---------- the user's own words: better recognition ----------
    def add_words(self, words: list[str]) -> None:
        w = self._load("words.json", {})
        for x in words:
            x = x.strip()
            if 2 < len(x) < 40:
                w[x] = w.get(x, 0) + 1
        self._save("words.json", w)

    def hotwords(self, limit: int = 25) -> str:
        w = self._load("words.json", {})
        top = [x for x in sorted(w, key=w.get, reverse=True) if x.lower() != "vidai"][:limit]
        return " ".join(["VidAI", *top])

    # ---------- macros: requests Claude solved once, instant next time ----------
    def macros(self) -> list[dict]:
        return self._load("macros.json", [])

    def clean_macros(self) -> list[str]:
        """Forget automatic shortcuts that were learned from garbled speech (see plausible_shortcut)."""
        ms = self.macros()
        keep = [m for m in ms if m.get("source") == "user" or plausible_shortcut(m["phrase"])]
        if len(keep) != len(ms):
            self._save("macros.json", keep)
        return [m["phrase"] for m in ms if m not in keep]

    def add_macro(self, phrase: str, commands: list[dict], source: str = "claude") -> None:
        p = _norm(phrase)
        if not p or not commands:
            return
        ms = [m for m in self.macros() if m["phrase"] != p]
        ms.append({"phrase": p, "commands": commands, "uses": 0, "source": source,
                   "learned": time.strftime("%Y-%m-%d %H:%M:%S")})
        self._save("macros.json", ms)

    def find_macro(self, text: str, threshold: float = 0.86) -> dict | None:
        t = _norm(text)
        best, score = None, 0.0
        for m in self.macros():
            r = difflib.SequenceMatcher(None, t, m["phrase"]).ratio()
            if r > score:
                best, score = m, r
        return best if best and score >= threshold else None

    def used_macro(self, phrase: str) -> None:
        ms = self.macros()
        for m in ms:
            if m["phrase"] == phrase:
                m["uses"] = m.get("uses", 0) + 1
        self._save("macros.json", ms)

    def forget_macro(self, phrase: str) -> bool:
        p = _norm(phrase)
        ms = self.macros()
        keep = [m for m in ms if m["phrase"] != p]
        self._save("macros.json", keep)
        return len(keep) != len(ms)

    # ---------- preferences ----------
    def prefs(self) -> dict[str, Any]:
        return self._load("prefs.json", {})

    def pref(self, key: str, default=None):
        return self.prefs().get(key, default)

    def set_pref(self, key: str, value) -> None:
        p = self.prefs()
        p[key] = value
        self._save("prefs.json", p)

    def nudge_pref(self, key: str, value: float, weight: float = 0.6) -> float:
        """Move a numeric preference toward a new observation (so one odd choice does not dominate)."""
        old = self.pref(key)
        new = value if old is None else old + weight * (value - old)
        self.set_pref(key, round(new, 3))
        return new

    # ---------- lessons and history ----------
    def add_lesson(self, lesson: str, tags: list[str] | None = None, source: str = "claude") -> None:
        for old in self.lessons(200):
            if difflib.SequenceMatcher(None, _norm(old["lesson"]), _norm(lesson)).ratio() > 0.9:
                return  # already known
        self._append("lessons.jsonl", {"lesson": lesson, "tags": tags or [], "source": source})

    def lessons(self, last: int = 30) -> list[dict]:
        return self._lines("lessons.jsonl", last)

    def add_history(self, row: dict) -> None:
        self._append("history.jsonl", row)

    def history(self, last: int = 10) -> list[dict]:
        return self._lines("history.jsonl", last)

    def summary(self) -> dict:
        """What Claude reads at the start of a session."""
        return {"lessons": [l["lesson"] for l in self.lessons(30)],
                "vocabulary": self.vocabulary(),
                "macros": [{"phrase": m["phrase"], "uses": m.get("uses", 0)} for m in self.macros()],
                "preferences": self.prefs(),
                "frequent_words": self.hotwords(),
                "recent_sessions": self.history(5)}


_VERBS = {"add", "at", "put", "make", "remove", "show", "change", "give", "turn", "set", "move", "write", "draw",
          "zoom", "blur", "place", "bring", "use", "apply", "hide", "delete", "replace", "create", "display",
          "take", "switch", "swap"}  # "at" = misheard "add"
_FILLER = {"and", "please", "vidai", "can", "could", "you", "now", "then", "ok", "okay", "so", "just", "also"}


def plausible_shortcut(phrase: str) -> bool:
    """Is this a clear instruction worth replaying instantly next time? Garbled speech ("to image to making",
    "and add add add rub it"), rambles and sentences with the wake word inside are not."""
    w = _norm(phrase).split()
    while w and w[0] in _FILLER:
        w = w[1:]
    if not 2 <= len(w) <= 12 or w[0] not in _VERBS or "vidai" in w:
        return False
    return not any(w.count(x) >= 2 for x in set(w) if len(x) > 2 and not x.isdigit())  # stuttered words

