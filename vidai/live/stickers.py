"""Stickers: any emoji (rendered from Noto Color Emoji), a word that maps to an emoji, or an image file."""
from __future__ import annotations

import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np

# word -> emoji (the fast path understands these without Claude)
WORDS = {
    "apple": "🍎", "green apple": "🍏", "orange": "🍊", "banana": "🍌", "lemon": "🍋", "grapes": "🍇",
    "strawberry": "🍓", "watermelon": "🍉", "cherry": "🍒", "pear": "🍐", "pineapple": "🍍", "peach": "🍑",
    "carrot": "🥕", "pizza": "🍕", "burger": "🍔", "donut": "🍩", "cake": "🍰", "ice cream": "🍦", "coffee": "☕",
    "tea": "🍵", "cookie": "🍪", "popcorn": "🍿", "egg": "🥚", "bread": "🍞", "cheese": "🧀",
    "crown": "👑", "hat": "🎩", "cap": "🧢", "party hat": "🥳", "glasses": "👓", "sunglasses": "🕶",
    "horns": "horns", "devil horns": "horns", "devil": "😈", "halo": "😇", "angel": "😇", "mustache": "👨", "bow": "🎀", "ribbon": "🎀",
    "heart": "❤", "hearts": "💕", "star": "⭐", "stars": "✨", "sparkles": "✨", "fire": "🔥", "flame": "🔥",
    "lightning": "⚡", "rainbow": "🌈", "sun": "☀", "moon": "🌙", "cloud": "☁", "snowflake": "❄",
    "flower": "🌸", "rose": "🌹", "tree": "🌳", "leaf": "🍃", "cactus": "🌵",
    "trophy": "🏆", "award": "🏆", "medal": "🏅", "gift": "🎁", "balloon": "🎈", "balloons": "🎈",
    "rocket": "🚀", "light bulb": "💡", "bulb": "💡", "idea": "💡", "book": "📖", "pencil": "✏", "pen": "🖊",
    "microscope": "🔬", "dna": "🧬", "test tube": "🧪", "magnet": "🧲", "computer": "💻", "phone": "📱",
    "money": "💰", "diamond": "💎", "key": "🔑", "lock": "🔒", "bell": "🔔", "clock": "⏰", "camera": "📷",
    "cat": "🐱", "dog": "🐶", "bird": "🐦", "fish": "🐟", "butterfly": "🦋", "bee": "🐝", "unicorn": "🦄",
    "robot": "🤖", "alien": "👽", "ghost": "👻", "skull": "💀", "poop": "💩", "clown": "🤡",
    "thumbs up": "👍", "like": "👍", "ok": "👌", "clap": "👏", "wave": "👋", "muscle": "💪",
    "question": "❓", "exclamation": "❗", "check": "✅", "cross": "❌", "warning": "⚠", "100": "💯",
    "ball": "⚽", "football": "⚽", "basketball": "🏀", "guitar": "🎸", "music": "🎵", "microphone": "🎤",
}
_EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF☀-➿⬀-⯿]")


def find_word(text: str) -> tuple[str, str] | None:
    """Longest known sticker word inside text -> (word, emoji)."""
    t = " " + re.sub(r"[^\w\s]", " ", text.lower()) + " "
    for w in sorted(WORDS, key=len, reverse=True):
        for form in (w, w + "s", w + "es"):
            if f" {form} " in t:
                return w, WORDS[w]
    m = _EMOJI_RE.search(text)
    return (m.group(0), m.group(0)) if m else None


@lru_cache(maxsize=1)
def _emoji_font_path() -> str | None:
    if shutil.which("fc-match"):
        p = subprocess.run(["fc-match", "-f", "%{file}", "Noto Color Emoji"], capture_output=True, text=True).stdout
        if p and "emoji" in p.lower() and Path(p).exists():
            return p
    for p in ("/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf", "/usr/share/fonts/noto/NotoColorEmoji.ttf"):
        if Path(p).exists():
            return p
    return None


@lru_cache(maxsize=256)
def sticker(what: str) -> np.ndarray:
    """RGBA uint8 sticker for an emoji, a known word, or an image path (tight-cropped)."""
    from PIL import Image, ImageDraw, ImageFont

    if Path(what).exists():
        img = Image.open(what).convert("RGBA")
    elif what.lower().strip() in ("horns", "devil horns"):
        img = _draw_horns()
    else:
        if what.startswith("text:"):  # always a text label, never an emoji
            what = emo = what[5:]
        else:
            emo = WORDS.get(what.lower().strip(), what)
        font_path = _emoji_font_path()
        img = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        if font_path and _EMOJI_RE.search(emo):
            d.text((10, 10), emo, font=ImageFont.truetype(font_path, 109), embedded_color=True)
        else:  # no emoji: a word label sticker
            from ..overlays import find_font

            f = ImageFont.truetype(find_font(), 40) if find_font() else ImageFont.load_default(40)
            w = int(d.textlength(what, font=f)) + 30
            img = Image.new("RGBA", (w, 70), (0, 0, 0, 0))
            d = ImageDraw.Draw(img)
            d.rounded_rectangle([0, 0, w - 1, 69], 18, fill=(255, 214, 10, 255))
            d.text((15, 10), what, font=f, fill=(20, 20, 30, 255))
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)
    return np.ascontiguousarray(np.asarray(img))


def scaled(what: str, width: int) -> np.ndarray:
    return _scaled(what, max(8, int(width) // 4 * 4))


@lru_cache(maxsize=256)
def _scaled(what: str, width: int) -> np.ndarray:
    from PIL import Image

    base = Image.fromarray(sticker(what))
    h = max(1, int(base.height * width / base.width))
    return np.ascontiguousarray(np.asarray(base.resize((width, h), Image.LANCZOS)))


def _draw_horns():
    from PIL import Image, ImageDraw

    W, H = 800, 300
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for cx, flip in ((170, -1), (630, 1)):
        d.polygon([(cx - 70, H - 10), (cx + 70, H - 10), (cx + flip * 30, 90), (cx + flip * 90, 10),
                   (cx - flip * 20, 110)], fill=(210, 30, 40, 255), outline=(90, 0, 10, 255))
        d.polygon([(cx - 40, H - 10), (cx - 5, H - 10), (cx + flip * 25, 120), (cx + flip * 70, 40)],
                  fill=(255, 90, 90, 160))
    return img
