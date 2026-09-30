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


COLORS = {"purple": [80, 50, 150], "blue": [30, 70, 170], "green": [30, 140, 70], "black": [10, 10, 12],
          "white": [235, 235, 240], "red": [170, 30, 40], "pink": [230, 120, 170], "orange": [230, 120, 30],
          "yellow": [230, 200, 50], "gray": [110, 110, 115], "grey": [110, 110, 115], "dark": [20, 18, 30]}


def _adjust(chain, **delta) -> list[dict]:
    """Change (or create) the one picture-adjustment effect. delta: param -> ("+", x) | ("=", x)."""
    cur = chain.get("fx_adjust")
    base = dict(cur.params) if cur is not None else {}
    new = {}
    for k, (op, v) in delta.items():
        default = {"brightness": 0.0, "contrast": 1.0, "saturation": 1.0, "warmth": 0.0, "gamma": 1.0,
                   "sharpen": 0.0}.get(k, 0.0)
        new[k] = v if op == "=" else round(min(max(base.get(k, default) + v, -0.5 if k in ("brightness", "warmth")
                                                   else 0.0), 2.5), 3)
    if cur is None:
        return [{"cmd": "add", "name": "fx_adjust", "type": "adjust", "params": new}]
    return [{"cmd": "set", "name": "fx_adjust", "params": new}]


def _images() -> dict[str, str]:
    """Photos VidAI already has (downloaded earlier), by the words in their file names."""
    import os
    from pathlib import Path

    d = Path(os.environ.get("VIDAI_HOME", Path.home() / ".vidai")) / "work" / "downloads"
    out = {}
    if d.exists():
        for f in d.iterdir():
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                for w in re.findall(r"[a-z]{3,}", f.stem.lower()):
                    out.setdefault(w, str(f))
    return out


def _picture(t: str, chain) -> list[dict] | None:
    """Light, contrast, colours, warmth, sharpness."""
    if re.search(r"\b(fix|improve|better|auto|correct)\w* (the |my )?(light|lighting|exposure)\b|\btoo dark\b", t):
        return _adjust(chain, auto=("=", True))
    if re.search(r"\b(normal|original|reset|natural)\b (the )?(colou?rs?|light|picture|image|filters?)\b", t):
        return [{"cmd": "remove", "name": "fx_adjust"}] if chain.get("fx_adjust") else None
    if re.search(r"\bblack and white\b|\bgr[ae]yscale\b|\bno colou?rs?\b", t):
        return _adjust(chain, saturation=("=", 0.0))
    if re.search(r"\b(brighter|brighten|more light|lighten up|light up|increase (the )?(light|brightness))\b", t):
        return _adjust(chain, brightness=("+", 0.07), gamma=("+", -0.1))
    if re.search(r"\b(darker|less light|dim|decrease (the )?(light|brightness))\b", t):
        return _adjust(chain, brightness=("+", -0.07), gamma=("+", 0.1))
    if re.search(r"\bmore contrast\b|\bincrease (the )?contrast\b", t):
        return _adjust(chain, contrast=("+", 0.15))
    if re.search(r"\bless contrast\b|\bdecrease (the )?contrast\b|\bsofter\b", t):
        return _adjust(chain, contrast=("+", -0.15))
    if re.search(r"\b(more colou?rs?|more colou?rful|colou?rful|vivid|saturate|more saturation)\b", t):
        return _adjust(chain, saturation=("+", 0.3))
    if re.search(r"\b(less colou?rs?|less colou?rful|muted|desaturate|less saturation)\b", t):
        return _adjust(chain, saturation=("+", -0.3))
    if re.search(r"\bwarmer\b|\bmore warm\b", t):
        return _adjust(chain, warmth=("+", 0.35))
    if re.search(r"\b(cooler|colder|more cool|bluer)\b", t):
        return _adjust(chain, warmth=("+", -0.35))
    if re.search(r"\b(sharper|sharpen|more sharp|more detail)\b", t):
        return _adjust(chain, sharpen=("+", 0.3))
    return None


def _background(t: str, chain) -> list[dict] | None:
    """'blur the background', 'beach behind me', 'purple background', 'moving background', 'remove the background'."""
    if not re.search(r"\bbackground\b|\bbehind me\b", t):
        return None
    params = None
    if re.search(r"\bblur\w*\b", t):
        params = {"mode": "blur"}
    elif re.search(r"\b(moving|animated|animation|live)\b", t):
        params = {"mode": "animated"}
    else:
        imgs = _images()
        hit = next((w for w in re.findall(r"[a-z]{3,}", t) if w in imgs and w not in ("background", "behind")), None)
        if hit:
            params = {"mode": "image", "image": imgs[hit]}
        else:
            col = next((c for c in COLORS if re.search(rf"\b{c}\b", t)), None)
            if col:
                params = {"mode": "color", "color": COLORS[col]}
            elif re.search(r"\b(remove|delete|hide|clear|replace|change)\b", t):
                if chain.get("fx_background") is not None and re.search(r"\b(remove|delete|clear)\b", t):
                    return [{"cmd": "remove", "name": "fx_background"}]  # it's on: take it off
                params = {"mode": "blur"}
    if params is None:
        return None  # e.g. "a forest behind me" with no forest photo yet -> Claude finds one
    if chain.get("fx_background") is not None:
        return [{"cmd": "set", "name": "fx_background", "params": params}]
    return [{"cmd": "add", "name": "fx_background", "type": "background", "params": params}]


def _hub(t: str, chain) -> list[dict] | None:
    """Requests a hub model can do: emotion, gestures, anime/cartoon look, painting styles."""
    if re.search(r"\b(remove|stop|no|without|normal)\b", t):
        return None  # "remove the style" etc. is handled by the remove rule
    if re.search(r"\b(emotions?|mood|feelings?|expressions?)\b", t):
        return [{"cmd": "model", "model": "emotion"}]
    if re.search(r"\b(gestures?|thumbs? up|hand signs?|signs with my hand)\b", t):
        return [{"cmd": "model", "model": "gestures"}]
    if "eye" not in t and re.search(r"\b(anime|ghibli|manga|cartoon (me|style|look|version))\b|\bme (a )?cartoon\b", t):
        return [{"cmd": "model", "model": "anime"}]
    styles = {"mosaic": "style_mosaic", "candy": "style_candy", "udnie": "style_udnie", "cubist": "style_udnie",
              "rain": "style_rain_princess", "impressionist": "style_rain_princess", "pointil": "style_pointilism",
              "dots": "style_pointilism"}
    for w, mid in styles.items():
        if re.search(rf"\b{w}", t) and re.search(r"\b(style|look|painting|filter|effect|like)\b", t):
            return [{"cmd": "model", "model": mid}]
    if re.search(r"\b(painting|painted|artistic|work of art)\b", t):
        return [{"cmd": "model", "model": "style_mosaic"}]
    return None


def _realistic(t: str, chain) -> list[dict] | None:
    """"make it realistic" / "blend it in": effects on the head go behind the hair and take the room's light
    (feed models trained on this video, see vidai.live.feed)."""
    if not re.search(r"\b(more )?(realistic|real looking|look(s)? real|blend (it |them )?in|natural looking)\b", t):
        return None
    worn = [p for p in _fx(chain) if p.params.get("to") in ("head", "above_head", "face", "eyes")
            or "hair" in p.wants_feeds() or p.name == "fx_rabbit_ears"]
    stuck = [p for p in _fx(chain) if "realistic" in p.params and p.params.get("to") != "screen"]
    cmds = [{"cmd": "set", "name": p.name, "params": {"realistic": True, "behind_hair": True}}
            for p in worn if "behind_hair" in p.params]
    cmds += [{"cmd": "set", "name": p.name, "params": {"realistic": True}} for p in stuck
             if p not in worn]
    return cmds or None


def match(text: str, chain) -> list[dict] | None:
    t = " " + re.sub(r"[^\w\s']", " ", text.lower()).strip() + " "
    t = re.sub(r"\s+", " ", t)
    fx = _fx(chain)

    if re.search(r"\b(back to normal|reset everything|clear everything|normal video|no effects|remove all( the)? effects)\b", t):
        return [{"cmd": "remove", "name": p.name} for p in fx] or None
    real = _realistic(t, chain)
    if real:
        return real
    for special in (_background, _picture, _hub):  # checked first: "remove the background" means apply it
        out = special(t, chain)
        if out:
            return out

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
