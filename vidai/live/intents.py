"""Fast path: understand common requests locally (no Claude round trip, < 1 s).

"add an apple in my hand", "put an orange on my other hand", "make my eyes pop", "horns on my head",
"put sunglasses on me", "remove the apple", "remove everything", "bigger apple", "swap hands".
Anything else returns None and goes to Claude.
"""
from __future__ import annotations

import re

from .stickers import WORDS, find_word

HEADWEAR = {"crown", "hat", "cap", "party hat", "horns", "devil horns", "halo", "angel", "bow", "ribbon"}
EYEWEAR = {"glasses", "sunglasses"}
ADD = r"\b(add|at|and|put|give|place|stick|attach|show|draw|want|make|wear|hold|set)\b"  # "at" = misheard "add"
ON_PART = r"\b(on|in|to|onto|into|above|over|at)\b (my |the |your )?(\w+ )?(hand|hands|head|face|eyes|nose|mouth|finger|palm)\b"


def _part(t: str) -> str | None:
    rules = [
        (r"\bother hand\b", "other_hand"), (r"\bright hand\b", "right_hand"), (r"\bleft hand\b", "left_hand"),
        (r"\bfinger\b", "finger"), (r"\bhands?\b|\bpalm\b", "hand"),
        (r"\b(above|over) (my |the )?head\b", "above_head"), (r"\bhead\b", "head"),
        (r"\beyes?\b", "eyes"), (r"\bnose\b", "nose"), (r"\bmouth\b|\blips\b", "mouth"), (r"\bface\b", "face"),
        (r"\bcorner\b|\bscreen\b", "screen"),
    ]
    for pat, part in rules:
        if re.search(pat, t):
            return part
    return None


def _fx(chain) -> list:
    return [p for p in chain.items if p.name.startswith("fx_")]


def _find(chain, t: str):
    """Effects referred to in the text (by sticker word, or 'eyes')."""
    fx = _fx(chain)
    w = find_word(t)
    hits = [p for p in fx if w and p.params.get("what") in (w[0], w[1])]
    if not hits and re.search(r"\beyes?\b", t):
        hits = [p for p in fx if type(p).__name__ == "BigEyes"]
    if not hits:  # any word of the effect's name or text: "remove these numbers", "remove introduction"
        words = {x for x in re.findall(r"[a-z]{3,}", t)} - {"remove", "removed", "the", "these", "that", "this",
                                                           "delete", "take", "off", "and", "all", "please", "from"}
        for p in fx:
            label = (p.name + " " + str(p.params.get("what", "")) + " " + str(p.params.get("text", ""))).lower()
            if any(x in label or x.rstrip("s") in label for x in words):
                hits.append(p)
    return hits


def match(text: str, chain) -> list[dict] | None:
    t = " " + re.sub(r"[^\w\s']", " ", text.lower()).strip() + " "
    t = re.sub(r"\s+", " ", t)
    fx = _fx(chain)

    # remove
    if re.search(r"\b(remove[ds]?|removing|delete[ds]?|deleting|take off|take away|get rid of|hide|clear|"
                 r"stop showing|reminds)\b", t):
        if re.search(r"\b(all|everything|every effect|effects)\b", t):
            return [{"cmd": "remove", "name": p.name} for p in fx] or None
        hits = _find(chain, t)
        return [{"cmd": "remove", "name": p.name} for p in hits] or None

    # bigger / smaller
    m = re.search(r"\b(bigger|larger|huge|increase|enlarge|grow|smaller|tiny|tinier|less|decrease|reduce|shrink)\b", t)
    if m and fx:
        up = m.group(1) in ("bigger", "larger", "huge", "increase", "enlarge", "grow")
        hits = _find(chain, t) or ([fx[-1]] if len(t.split()) <= 3 else [])
        out = []
        for p in hits:
            if type(p).__name__ == "BigEyes":
                out.append({"cmd": "set", "name": p.name,
                            "params": {"zoom": max(1.1, p.params["zoom"] + (0.4 if up else -0.4))}})
            else:
                out.append({"cmd": "set", "name": p.name,
                            "params": {"scale": p.params.get("scale", 1.0) * (1.4 if up else 1 / 1.4)}})
        return out or None

    # swap hands
    if re.search(r"\bswap\b|\bswitch hands\b|\bother way\b", t):
        hand_fx = [p for p in fx if p.params.get("to") in ("right_hand", "left_hand")]
        if len(hand_fx) >= 2:
            flip = {"right_hand": "left_hand", "left_hand": "right_hand"}
            return [{"cmd": "set", "name": p.name, "params": {"to": flip[p.params["to"]]}} for p in hand_fx]
        return None

    # pop-out eyes
    if re.search(r"\beyes?\b", t) and re.search(r"\b(pop|popping|big|bigger|huge|cartoon|bulg\w*|googly)\b", t):
        return [{"cmd": "add", "name": "fx_big_eyes", "type": "big_eyes", "params": {}}]

    # text: "add a text above my head saying an award", "put a title that says Hello"
    m = re.search(r"\b(text|title|label|sign|words?|caption)\b.*?\b(saying|says|that says|reading|with)\b\s+(.+)$", t)
    if m:
        words = m.group(3).strip(" '")
        part = _part(t[: m.start(2)]) or "above_head"
        if part in ("screen",) or not re.search(r"\bhead\b|\bhand\b|\bface\b", t[: m.start(2)]):
            return [{"cmd": "text", "text": _title(words), "position": "top-center", "size": 0.06, "for": 6}]
        return [{"cmd": "add", "name": f"fx_text_{part}", "type": "attach",
                 "params": {"what": "text:" + _title(words), "to": part, "scale": 1.0}}]

    # attach a sticker
    w = find_word(t)
    if w and (re.search(ADD, " " + t) or re.search(ON_PART, t)):
        word, emoji = w
        part = _part(t) or ("head" if word in HEADWEAR else "eyes" if word in EYEWEAR else None)
        if part is None:
            return None  # where? let Claude decide
        cmds = []
        if part == "other_hand":  # the first object goes to one hand, this one to the other
            prev = [p for p in fx if p.params.get("to") in ("hand", "right_hand", "left_hand")]
            if prev:
                cmds.append({"cmd": "set", "name": prev[-1].name, "params": {"to": "right_hand"}})
            part = "left_hand"
        what = WORDS.get(word, emoji) if word not in ("horns", "devil horns") else "horns"
        cmds.append({"cmd": "add", "name": f"fx_{word.replace(' ', '_')}_{part}", "type": "attach",
                     "params": {"what": what, "to": part}})
        return cmds
    return None


def _title(s: str) -> str:
    return " ".join(w if w.isupper() else w.capitalize() for w in s.split())
