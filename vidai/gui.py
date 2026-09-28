"""VidAI Recorder: the window Claude opens after the brief.

    python -m vidai.gui <session_dir> [--autostart] [--auto-stop SECONDS]

Preview -> Record (3 s countdown) -> markers while recording -> Stop -> anchors saved -> back to Claude.
Global hotkeys: ctrl+alt+m marker, ctrl+alt+x mistake, ctrl+alt+n section, ctrl+alt+i important,
ctrl+alt+s stop. In the window: m / x / n / i, space = record/stop.
"""
from __future__ import annotations

import argparse
import subprocess
import threading
import time
from pathlib import Path

import customtkinter as ctk
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .live.pipeline import LiveConfig, LivePipeline
from .overlays import find_font
from .session import Session, SessionRecorder

# ---------- theme ----------
BG = "#0e0e15"
CARD = "#181824"
CARD_2 = "#222232"
LINE = "#2c2c40"
TEXT = "#ececf4"
MUTED = "#8b8ba3"
ACCENT = "#8b6cff"
ACCENT_2 = "#5b8cff"
REC = "#ff3b5c"
OK = "#34d399"
WARN = "#fbbf24"

MODES = {"Camera": "camera", "Screen": "screen", "Screen + Camera": "screen+camera"}
MARKS = [("◆  Marker", "marker", "m", ACCENT_2), ("✂  Mistake", "mistake", "x", WARN),
         ("§  Section", "section", "n", ACCENT), ("★  Important", "important", "i", REC)]
PREVIEW_W, PREVIEW_H = 464, 261
FONT = "Ubuntu Sans"
ASSETS = Path(__file__).resolve().parent / "assets"  # icons ship inside the package


def _pil_font(size: int, bold: bool = True):
    path = find_font(bold=bold)
    return ImageFont.truetype(path, size) if path else ImageFont.load_default(size=size)


def _rounded(img: Image.Image, radius: int) -> Image.Image:
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, img.width - 1, img.height - 1], radius, fill=255)
    out = Image.new("RGBA", img.size, (0, 0, 0, 0))
    out.paste(img.convert("RGBA"), (0, 0), mask)
    return out


def _wrap(draw, text: str, font, width: int) -> list[str]:
    """Split text into lines that fit `width` pixels (word wrap; very long words are cut)."""
    lines, cur = [], ""
    for word in text.split():
        cand = f"{cur} {word}".strip()
        if draw.textlength(cand, font=font) <= width:
            cur = cand
            continue
        if cur:
            lines.append(cur)
        while draw.textlength(word, font=font) > width and len(word) > 1:
            word = word[:-1]
        cur = word
    if cur:
        lines.append(cur)
    return lines or [""]


def _draw_lines(draw, lines: list[str], font, x: int, y: int, width: int, max_y: int, first_color, color,
                gap: int = 3) -> int:
    """Draw wrapped lines until max_y; returns the next y."""
    step = int(font.size * 1.25) + gap if hasattr(font, "size") else 18
    for i, line in enumerate(lines):
        for part in _wrap(draw, line, font, width):
            if y + step > max_y:
                return y
            draw.text((x, y), part, font=font, fill=first_color if i == 0 else color)
            y += step
    return y


class RecorderApp:
    def __init__(self, root: ctk.CTk, session: Session, autostart: bool = False) -> None:
        self.root, self.session = root, session
        self.latest: np.ndarray | None = None
        self.preview: LivePipeline | None = None
        self.live_text = ""
        self.live_color = MUTED
        self.ask: tuple[str, str] | None = None  # (request id, text) waiting for Confirm / Deny
        self.question: tuple[str, str, list] | None = None  # Claude's question: (id, text, options)
        self._bar_for = None  # what the answer bar currently shows
        self.toasts: list[tuple[str, str, float]] = []  # (text, color, until) drawn on the preview only
        self.help_lines: list[str] = []
        self.help_until = 0.0
        self._ptt = threading.Event()  # push-to-talk hotkey (ctrl+alt+space)
        self._ask_shown = False
        from . import actions

        self.full_access = actions.get_mode(session.dir) == "full"
        self.rec: SessionRecorder | None = None
        self.state = "idle"  # idle | countdown | recording | saving | done
        self.count = 0
        self.flash: tuple[str, str, int] | None = None  # (text, color, ticks left) shown on the preview
        self._hotkeys = None
        self._stop_requested = threading.Event()  # set from the hotkey thread, handled in _tick
        self._start_requested = threading.Event()  # "VidAI record" heard in preview
        self._save_thread: threading.Thread | None = None
        self._tickn = 0
        self._f_badge, self._f_count, self._f_small = _pil_font(20), _pil_font(120), _pil_font(15, bold=False)
        self._f_help = _pil_font(12, bold=False)
        icon_path = ASSETS / "icon.png"  # wide eye-shaped logo
        self.icon = Image.open(icon_path).convert("RGBA") if icon_path.exists() else None
        sq = ASSETS / "icon_square.png"  # same logo on a square canvas for the window/taskbar icon
        self.icon_sq = Image.open(sq).convert("RGBA") if sq.exists() else self.icon

        root.title("VidAI Recorder")
        root.configure(fg_color=BG)
        root.resizable(False, False)
        if self.icon:
            from PIL import ImageTk

            self._wm_icon = ImageTk.PhotoImage(self.icon_sq.resize((128, 128), Image.LANCZOS))
            root.iconphoto(True, self._wm_icon)
        self._build()
        try:
            from pynput import keyboard

            self._ptt_keys = keyboard.GlobalHotKeys({"<ctrl>+<alt>+<space>": self._ptt.set})
            self._ptt_keys.start()
        except Exception:
            self._ptt_keys = None
        self._start_preview()
        self._tick()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        if autostart:
            root.after(800, self.toggle)

    # ---------- layout ----------
    def _font(self, size: int, weight: str = "normal") -> ctk.CTkFont:
        return ctk.CTkFont(family=FONT, size=size, weight=weight)

    def _chip(self, parent, text: str, color: str = MUTED) -> ctk.CTkLabel:
        return ctk.CTkLabel(parent, text=text, fg_color=CARD_2, corner_radius=10, text_color=color,
                            font=self._font(11), height=22, padx=8)

    def _build(self) -> None:
        r, b, s = self.root, self.session.brief, self.session
        pad = {"padx": 18}

        # header
        head = ctk.CTkFrame(r, fg_color="transparent")
        head.pack(fill="x", pady=(12, 6), **pad)
        if self.icon:
            lw = 74
            logo = ctk.CTkImage(dark_image=self.icon, light_image=self.icon,
                                size=(lw, int(lw * self.icon.height / self.icon.width)))
            ctk.CTkLabel(head, image=logo, text="").pack(side="left")
        names = ctk.CTkFrame(head, fg_color="transparent")
        names.pack(side="left", padx=10)
        title_row = ctk.CTkFrame(names, fg_color="transparent")
        title_row.pack(anchor="w")
        ctk.CTkLabel(title_row, text="Vid", font=self._font(24, "bold"), text_color=TEXT).pack(side="left")
        ctk.CTkLabel(title_row, text="AI", font=self._font(24, "bold"), text_color=ACCENT).pack(side="left")
        ctk.CTkLabel(title_row, text="  Recorder", font=self._font(15), text_color=MUTED).pack(side="left", pady=(6, 0))
        ctk.CTkLabel(names, text="Claude's video plugin", font=self._font(11), text_color=MUTED).pack(anchor="w")
        self.pill = ctk.CTkLabel(head, text="●  READY", fg_color=CARD_2, corner_radius=14, height=30, padx=14,
                                 text_color=OK, font=self._font(12, "bold"))
        self.pill.pack(side="right")

        # brief card
        card = ctk.CTkFrame(r, fg_color=CARD, corner_radius=16, border_width=1, border_color=LINE)
        card.pack(fill="x", pady=6, **pad)
        ctk.CTkLabel(card, text=b.title or "Untitled video", font=self._font(15, "bold"), text_color=TEXT,
                     anchor="w").pack(fill="x", padx=16, pady=(8, 0))
        if b.topic:
            ctk.CTkLabel(card, text=b.topic, font=self._font(12), text_color=MUTED, anchor="w", justify="left",
                         wraplength=PREVIEW_W - 20).pack(fill="x", padx=16)
        chips = ctk.CTkFrame(card, fg_color="transparent")
        chips.pack(fill="x", padx=14, pady=(6, 8))
        for text in (b.language.upper(), b.style.replace("_", " "), f"~{b.expected_minutes:g} min",
                     f"{len(s.anchors.stats)} anchors"):
            self._chip(chips, text).pack(side="left", padx=2)
        if b.outline:
            self._chip(chips, f"{len(b.outline)} sections", ACCENT).pack(side="left", padx=2)

        # preview
        self.preview_label = ctk.CTkLabel(r, text="")
        self.preview_label.pack(pady=(8, 6), **pad)
        self._show(self._placeholder("Starting preview…"))

        self.level = ctk.CTkProgressBar(r, height=6, corner_radius=3, progress_color=OK, fg_color=CARD_2)
        self.level.set(0)
        self.level.pack(fill="x", pady=(0, 8), **pad)

        # settings
        settings = ctk.CTkFrame(r, fg_color="transparent")
        settings.pack(fill="x", **pad)
        mode_name = {v: k for k, v in MODES.items()}.get(s.capture.mode, "Screen + Camera")
        values = list(MODES) + (["Test"] if s.capture.mode == "test" else [])
        if s.capture.mode == "test":
            mode_name = "Test"
        self.mode = ctk.CTkSegmentedButton(settings, values=values, command=lambda v: self._mode_changed(),
                                           selected_color=ACCENT, selected_hover_color="#7a58ff",
                                           unselected_color=CARD_2, unselected_hover_color=LINE,
                                           fg_color=CARD_2, text_color=TEXT, font=self._font(12), height=32,
                                           corner_radius=10)
        self.mode.set(mode_name)
        self.mode.pack(side="left")
        self.mic = ctk.CTkSwitch(settings, text="Mic", command=self._mode_changed, progress_color=ACCENT,
                                 font=self._font(12), text_color=TEXT, switch_width=38)
        self.mic.select() if s.capture.mic else self.mic.deselect()
        self.mic.pack(side="right")
        self.hide = ctk.CTkSwitch(settings, text="Auto-hide", progress_color=ACCENT, font=self._font(12),
                                  text_color=TEXT, switch_width=38)
        self.hide.select()
        self.hide.pack(side="right", padx=(0, 14))

        # controls: two markers | record | two markers
        ctl = ctk.CTkFrame(r, fg_color="transparent")
        ctl.pack(fill="x", pady=(6, 0), **pad)
        self.mark_btns = []
        cols = [0, 1, 3, 4]
        for col, (label, kind, key, color) in zip(cols, MARKS):
            btn = ctk.CTkButton(ctl, text=f"{label}\n{key}", command=lambda k=kind: self.mark(k), height=50,
                                width=96, corner_radius=12, fg_color=CARD, hover_color=CARD_2, border_width=1,
                                border_color=LINE, text_color=TEXT, text_color_disabled="#55556a",
                                font=self._font(11), state="disabled")
            btn.grid(row=0, column=col, padx=3, sticky="ew")
            ctl.grid_columnconfigure(col, weight=1)
            self.mark_btns.append((btn, color))
            self.root.bind(key, self._keyguard(lambda k=kind: self.mark(k)))
        self.rec_btn = ctk.CTkButton(ctl, text="●", width=72, height=72, corner_radius=36, fg_color=REC,
                                     hover_color="#ff5c77", text_color="white", font=ctk.CTkFont(size=28),
                                     command=self.toggle, border_width=4, border_color="#3a1320")
        self.rec_btn.grid(row=0, column=2, padx=10)
        self.rec_caption = ctk.CTkLabel(ctl, text="Record  ·  space", font=self._font(11), text_color=MUTED)
        self.rec_caption.grid(row=1, column=0, columnspan=5, pady=(4, 0))
        self.root.bind("<space>", self._keyguard(self.toggle))

        # talk to VidAI: type a request, or push-to-talk
        self._pad = pad
        self.input_row = ctk.CTkFrame(r, fg_color="transparent")
        self.input_row.pack(fill="x", pady=(6, 12), **pad)
        self.entry = ctk.CTkEntry(self.input_row, placeholder_text="Tell VidAI…  (or say “VidAI …”, or hold ctrl+alt+space)",
                                  height=32, corner_radius=10, fg_color=CARD, border_color=LINE, text_color=TEXT,
                                  font=self._font(12))
        self.entry.pack(side="left", fill="x", expand=True)
        self.entry.bind("<Return>", lambda e: self._send_typed())
        ctk.CTkButton(self.input_row, text="🎙", width=36, height=32, corner_radius=10, fg_color=CARD_2,
                      hover_color=LINE, command=self._push_to_talk).pack(side="left", padx=(6, 0))
        ctk.CTkButton(self.input_row, text="➤", width=36, height=32, corner_radius=10, fg_color=ACCENT,
                      hover_color="#7a58ff", command=self._send_typed).pack(side="left", padx=(6, 0))
        # answer bar: permission (Confirm / Deny) or Claude's question (option buttons)
        self.ask_bar = ctk.CTkFrame(r, fg_color=CARD_2, corner_radius=12)
        self.ask_label = ctk.CTkLabel(self.ask_bar, text="", font=self._font(12, "bold"), text_color=WARN,
                                      anchor="w", width=10)
        self.ask_label.pack(side="left", padx=(10, 4), pady=3)
        self.ask_buttons = ctk.CTkFrame(self.ask_bar, fg_color="transparent")
        self.ask_buttons.pack(side="left", fill="x", expand=True, padx=(0, 4), pady=3)
        for key, what in (("y", "confirm"), ("Y", "confirm"), ("n", "deny"), ("N", "deny")):
            self.root.bind(key, self._keyguard(lambda w=what: self._answer(w)))
        self.marks_log: list[str] = []

    def _keyguard(self, fn):
        """Window shortcuts (m, x, n, i, y, space...) must not fire while the user types in the box."""
        def handler(event=None):
            if str(self.root.focus_get()).startswith(str(self.entry)):
                return None
            return fn()
        return handler

    # ---------- talking to VidAI ----------
    def _pipe(self):
        return self.rec.pipe if (self.rec and self.state == "recording") else self.preview

    def _send_typed(self) -> None:
        text = self.entry.get().strip()
        self.entry.delete(0, "end")
        self.root.focus_set()
        pipe = self._pipe()
        if not text or pipe is None:
            return
        if self.question:  # typing answers Claude's question
            pipe.command({"cmd": "answer", "question": self.question[0], "text": text, "by": "typed"}, source="gui")
            return
        cmd = __import__("vidai.live.sensors", fromlist=["parse_command"]).parse_command("vidai " + text)
        if cmd and cmd["command"] != "claude":  # typed built-in command ("zoom in", "undo", "help")
            pipe.bus.publish("voice_command", {**cmd, "text": text, "typed": True})
        else:
            self.toast(f"→ {text}", ACCENT_2)
            threading.Thread(target=pipe.request, args=(text, "typed"), daemon=True).start()

    def _push_to_talk(self) -> None:
        pipe = self._pipe()
        if pipe:
            pipe.command({"cmd": "listen", "seconds": 6}, source="gui")

    def toast(self, text: str, color: str = TEXT, seconds: float = 4.0) -> None:
        self.toasts.append((text, color, time.monotonic() + seconds))
        del self.toasts[:-3]

    # ---------- preview drawing ----------
    def _placeholder(self, text: str) -> Image.Image:
        img = Image.new("RGB", (PREVIEW_W, PREVIEW_H), CARD)
        d = ImageDraw.Draw(img)
        for y in range(PREVIEW_H):  # soft vertical gradient
            c = int(24 + 10 * y / PREVIEW_H)
            d.line([(0, y), (PREVIEW_W, y)], fill=(c, c, c + 12))
        if self.icon:
            iw = 150
            ic = self.icon.resize((iw, int(iw * self.icon.height / self.icon.width)), Image.LANCZOS)
            img.paste(ic, ((PREVIEW_W - iw) // 2, PREVIEW_H // 2 - ic.height - 6), ic)
        w = d.textlength(text, font=self._f_small)
        d.text(((PREVIEW_W - w) / 2, PREVIEW_H // 2 + 34), text, font=self._f_small, fill=MUTED)
        return img

    def _decorate(self, img: Image.Image) -> Image.Image:
        d = ImageDraw.Draw(img, "RGBA")
        if self.state == "recording" and self.rec:
            t = int(self.rec.elapsed)
            label = f"REC  {t // 60:02d}:{t % 60:02d}"
            w = d.textlength(label, font=self._f_badge)
            d.rounded_rectangle([14, 14, 14 + w + 44, 50], 18, fill=(0, 0, 0, 150))
            on = (self._tickn // 5) % 2 == 0
            d.ellipse([26, 24, 42, 40], fill=(255, 59, 92, 255 if on else 90))
            d.text((50, 18), label, font=self._f_badge, fill=(255, 255, 255))
        if self.state == "countdown":
            d.rectangle([0, 0, PREVIEW_W, PREVIEW_H], fill=(0, 0, 0, 110))
            s = str(self.count)
            w = d.textlength(s, font=self._f_count)
            d.text(((PREVIEW_W - w) / 2, PREVIEW_H / 2 - 80), s, font=self._f_count, fill=(255, 255, 255))
        now = time.monotonic()
        if self.help_lines and now < self.help_until:
            d.rectangle([0, 0, PREVIEW_W, PREVIEW_H], fill=(8, 8, 16, 215))
            d.text((16, 12), "What you can say", font=self._f_badge, fill=ACCENT)
            _draw_lines(d, self.help_lines, self._f_help, 16, 44, PREVIEW_W - 32, PREVIEW_H - 6, MUTED, TEXT)
        self.toasts = [t for t in self.toasts if t[2] > now]
        y = PREVIEW_H - 14
        for text, color, _ in reversed(self.toasts[-2:]):
            parts = _wrap(d, text, self._f_small, PREVIEW_W - 60)
            if len(parts) > 2:
                parts = [parts[0], parts[1][:-1] + "…"]
            w = max(d.textlength(p, font=self._f_small) for p in parts)
            h = 10 + 20 * len(parts)
            y -= h + 6
            d.rounded_rectangle([12, y, 12 + w + 24, y + h], 12, fill=(10, 10, 20, 200))
            for i, part in enumerate(parts):
                d.text((24, y + 4 + 20 * i), part, font=self._f_small, fill=color)
        pending = (f"{self.session.live.address}, I need to {self.ask[1]}", WARN) if self.ask else (
            ((self.question[1] if self.question[1].startswith("💡") else f"Claude asks: {self.question[1]}"),
             ACCENT_2) if self.question else None)
        if pending:  # the question stays on the preview until answered (never recorded)
            parts = _wrap(d, pending[0], self._f_small, PREVIEW_W - 48)[:3]
            h = 12 + 20 * len(parts)
            d.rounded_rectangle([12, 58, PREVIEW_W - 12, 58 + h], 12, fill=(10, 10, 20, 215),
                                outline=pending[1], width=2)
            for i, part in enumerate(parts):
                d.text((24, 64 + 20 * i), part, font=self._f_small, fill=pending[1])
        if self.flash:
            text, color, _ = self.flash
            w = d.textlength(text, font=self._f_badge)
            x = PREVIEW_W - w - 44
            d.rounded_rectangle([x, 14, PREVIEW_W - 14, 50], 18, fill=(0, 0, 0, 160))
            d.text((x + 15, 18), text, font=self._f_badge, fill=color)
        return img

    def _show(self, img: Image.Image) -> None:
        img = _rounded(img, 18)
        self._ctk_img = ctk.CTkImage(dark_image=img, light_image=img, size=(PREVIEW_W, PREVIEW_H))
        self.preview_label.configure(image=self._ctk_img)

    # ---------- capture ----------
    def _on_frame(self, frame: np.ndarray) -> None:
        self.latest = frame  # capture thread; the UI reads it in _tick

    def _mode_value(self) -> str:
        return "test" if self.mode.get() == "Test" else MODES[self.mode.get()]

    def _start_preview(self) -> None:
        cfg = self.session.capture.model_copy(update={"mode": self._mode_value(), "mic": bool(self.mic.get())})
        self.latest = None
        try:
            live = self.session.live.model_copy(update={"ocr": False})
            self.preview = LivePipeline(cfg, live, None, session_dir=self.session.dir, on_frame=self._on_frame,
                                        on_start_request=self._start_requested.set,
                                        on_stop_request=lambda: None)
            self.preview.bus.subscribe(self._on_live, {"transcript", "voice_command", "claude", "error", "action",
                                                       "permission", "permission_mode", "notify", "warning", "help", "question", "answer"})
            self.preview.start()
        except Exception as e:
            self.preview = None
            self.set_status(f"Preview failed: {e}", REC)

    def _stop_preview(self) -> None:
        if self.preview:
            self.preview.stop(timeout=5)
            self.preview = None

    def _mode_changed(self) -> None:
        if self.state != "idle":
            return
        self.session.capture.mode = self._mode_value()  # type: ignore[assignment]
        self.session.capture.mic = bool(self.mic.get())
        self.session.save()
        self._stop_preview()
        self._show(self._placeholder("Switching source…"))
        self._start_preview()

    # ---------- actions ----------
    def toggle(self) -> None:
        if self.state == "idle":
            self.mode.configure(state="disabled")
            self.mic.configure(state="disabled")
            self.rec_btn.configure(state="disabled")
            self._countdown(3)
        elif self.state == "recording":
            self.stop()

    def _countdown(self, n: int) -> None:
        self.state, self.count = "countdown", n
        self.rec_caption.configure(text=f"Starting in {n}…")
        self._set_pill(f"●  {n}", WARN)
        if n > 0:
            self.root.after(1000, self._countdown, n - 1)
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        carry = self.preview.carry_specs() if self.preview else []
        self._stop_preview()
        self.session.capture.mode = self._mode_value()  # type: ignore[assignment]
        self.session.capture.mic = bool(self.mic.get())
        self.rec = SessionRecorder(self.session, on_frame=self._on_frame, on_stop_request=self._stop_requested.set,
                                   carry=carry)
        try:
            self.rec.start()
        except Exception as e:
            self.state = "idle"
            self.set_status(f"Could not start: {e}", REC)
            self._set_pill("●  READY", OK)
            for w in (self.mode, self.mic, self.rec_btn):
                w.configure(state="normal")
            self.rec_caption.configure(text="Record  ·  space")
            self._start_preview()
            return
        self.state = "recording"
        self.rec.pipe.bus.subscribe(self._on_live, {"transcript", "voice_command", "claude", "error", "learner",
                                                    "screen_text", "action", "permission", "permission_mode", "notify", "warning", "help", "question", "answer"})
        self.rec_btn.configure(state="normal", text="■", fg_color=CARD_2, hover_color=LINE, border_color=REC)
        self.rec_caption.configure(text="Stop  ·  space  ·  ctrl+alt+s")
        for btn, color in self.mark_btns:
            btn.configure(state="normal", border_color=color)
        self._start_hotkeys()
        self.set_status("Recording  ·  ctrl+alt + m / x / n / i to mark, s to stop")
        if self.hide.get() and "screen" in self.session.capture.mode:
            self.root.after(400, self.root.iconify)

    def _on_live(self, ev: dict) -> None:
        """Bus thread: only store text; the Tk loop displays it."""
        d, k = ev["data"], ev["kind"]
        if k == "notify":
            color = {"ok": OK, "warn": WARN, "claude": ACCENT_2, "listen": ACCENT}.get(d.get("kind"), TEXT)
            self.toast(("Claude: " if d.get("kind") == "claude" else "") + d.get("text", ""), color,
                       float(d.get("seconds", 4)))
            return
        if k == "warning":
            self.toast("⚠ " + d.get("text", ""), WARN, 7)
            return
        if k == "help":
            self.help_lines, self.help_until = d.get("lines", []), time.monotonic() + float(d.get("seconds", 14))
            return
        if k == "question":
            self.question = (d["question"], d["text"], d.get("options") or [])
            self.help_until = 0.0
            return
        if k == "answer":
            if self.question and (self.question[0] == d.get("question") or
                                  (d.get("cancelled") and not d.get("question"))):
                self.question = None
            return
        if k == "action" and d.get("what") == "performance":
            self.toast(d.get("text", ""), OK, 4)
            return
        if k == "action" and d.get("what") == "asking":
            self.ask = (d["request"], d["text"])
            self.help_until = 0.0
            return
        if k == "permission":
            if self.ask and self.ask[0] == d.get("request"):
                self.ask = None
            ok = d.get("state") == "approved"
            self.live_text, self.live_color = (f"✓ allowed: {d.get('text', '')}"[:90] if ok
                                               else f"✕ denied: {d.get('text', '')}"[:90]), (OK if ok else REC)
            return
        if k == "permission_mode":
            self.full_access = d.get("mode") == "full"
            if self.full_access:
                self.ask = None
            self.live_text, self.live_color = ("FULL ACCESS: VidAI takes all actions" if self.full_access
                                               else "VidAI will ask you first"), WARN
            return
        if k == "transcript" and not d.get("is_command"):
            self.live_text, self.live_color = f"› {d['text'][:90]}", MUTED
        elif k == "voice_command":
            self.live_text, self.live_color = f"VidAI  {d['command']} {d.get('args', '')}".strip(), ACCENT
        elif k == "claude":
            self.live_text, self.live_color = f"→ Claude: {d['message'][:80]}", ACCENT_2
        elif k == "learner":
            self.live_text, self.live_color = f"model {d['name']}: {d['label']} ({d['confidence']:.0%})", OK
        elif k == "error":
            self.live_text, self.live_color = f"! {d.get('where', d.get('processor', ''))}: {str(d.get('error'))[:80]}", REC
        elif k == "action" and d.get("what") in ("processor_added", "text", "zoom", "shape", "image", "blur"):
            self.live_text, self.live_color = f"live: {d['what']} {d.get('name', '')}", OK
        elif k == "action" and d.get("what") == "learned":
            txt = {"vocabulary": f"learned: “{d.get('heard')}” means “{d.get('meant')}”",
                   "macro": f"learned: “{d.get('request')}” is instant next time",
                   "mistake": f"noted: “{d.get('request')}” went wrong",
                   "preference": "learned your hand preference"}.get(d.get("kind"), "learned something")
            self.live_text, self.live_color = txt[:95], ACCENT
        elif k == "action" and d.get("what") == "corrected":
            self.live_text, self.live_color = f"heard “{d['heard']}” → “{d['meant']}”"[:95], ACCENT_2
        elif k == "action" and d.get("what") == "listening_request":
            self.live_text, self.live_color = f"→ Claude (keep talking…): {d.get('so_far', '')[:70]}", ACCENT_2
        elif k == "action" and d.get("what") == "listening":
            self.live_text, self.live_color = "VidAI is listening… say the command", ACCENT
        elif k == "action" and d.get("what") == "stt_ready":
            self.live_text, self.live_color = ("voice ready: say “VidAI record” to start" if self.state == "idle"
                                               else "voice ready: say “VidAI …” (mark, new section, zoom in, stop)"), OK

    def _answer(self, what: str) -> None:
        pipe = self._pipe()
        if not pipe:
            return
        if self.ask and what in ("confirm", "deny"):
            pipe.command({"cmd": what, "request": self.ask[0], "by": "button"}, source="gui")
        elif self.question and what not in ("confirm", "deny"):
            pipe.command({"cmd": "answer", "question": self.question[0], "text": what, "by": "button"}, source="gui")

    def _show_bar(self) -> None:
        """Answer bar: permission (Confirm/Deny) or Claude's question (one button per option)."""
        want = ("ask", self.ask) if self.ask else (("q", self.question) if self.question else None)
        if want and want[0] == "q" and not self.question[2]:
            want = None  # free-form question: keep the "Tell VidAI…" box — typing or speaking answers it
        if want == self._bar_for:
            return
        self._bar_for = want
        for w in self.ask_buttons.winfo_children():
            w.destroy()
        if want is None:
            self.ask_bar.pack_forget()
            self.input_row.pack(fill="x", pady=(6, 12), **self._pad)
            return
        self.input_row.pack_forget()
        if want[0] == "ask":
            self.ask_label.configure(text="Allow?", text_color=WARN)
            buttons = [("✓ Confirm (y)", "confirm", OK), ("✕ Deny (n)", "deny", CARD)]
        else:
            _, text, options = self.question
            self.ask_label.configure(text="Answer:", text_color=ACCENT_2)
            buttons = [(o, o, ACCENT) for o in options[:4]]
        for label, what, color in buttons:
            ctk.CTkButton(self.ask_buttons, text=label[:18], width=40, height=26, corner_radius=10, fg_color=color,
                          hover_color=LINE, text_color=BG if color in (OK, ACCENT) else TEXT, font=self._font(11),
                          command=lambda w=what: self._answer(w)).pack(side="left", padx=4, expand=True, fill="x")
        self.ask_bar.pack(fill="x", pady=(6, 12), **self._pad)

    def mark(self, kind: str) -> None:
        if self.state != "recording" or not self.rec:
            return
        t = self.rec.mark(kind)
        label, color = next((l, c) for l, k, _, c in MARKS if k == kind)
        self.flash = (label.split()[-1], color, 12)
        self.marks_log.append(f"{label.split()[-1]} {int(t) // 60:02d}:{t % 60:04.1f}")
        self.set_status(f"{len(self.marks_log)} marks  ·  last: " + "  ·  ".join(self.marks_log[-2:][::-1]), TEXT)

    def _start_hotkeys(self) -> None:
        try:
            from pynput import keyboard

            # never touch Tk from this thread (Tk is not thread-safe): just raise a flag
            self._hotkeys = keyboard.GlobalHotKeys({"<ctrl>+<alt>+s": self._stop_requested.set})
            self._hotkeys.start()
        except Exception:
            self._hotkeys = None

    def stop(self) -> None:
        if self.state != "recording" or not self.rec:
            return
        self.state = "saving"
        if self._hotkeys:
            self._hotkeys.stop()
        self.root.deiconify()
        self._set_pill("●  SAVING", WARN)
        self.rec_btn.configure(state="disabled", text="…")
        self.rec_caption.configure(text="Writing video and anchors")
        for btn, _ in self.mark_btns:
            btn.configure(state="disabled", border_color=LINE)

        def work() -> None:
            try:
                a = self.rec.stop()  # type: ignore[union-attr]
                summary = (f"{a.duration:.0f}s recorded  ·  {len(a.segments_of('silence'))} pauses  ·  "
                           f"{len(a.events_of('markers'))} markers")
                self.root.after(0, self._done, summary, None)
            except Exception as e:
                self.root.after(0, self._done, "", str(e))

        self._save_thread = threading.Thread(target=work, daemon=False)  # never killed mid-save
        self._save_thread.start()

    def _done(self, summary: str, error: str | None) -> None:
        self.state = "done"
        if error:
            self._set_pill("●  ERROR", REC)
            self.set_status(error, REC)
            return
        self._set_pill("✓  DONE", OK)
        self._show(self._summary_card(summary))
        self.toast(Path(self.session.dir).name, MUTED, 8)
        self.rec_btn.configure(state="normal", text="⤴", fg_color=ACCENT, hover_color="#7a58ff",
                               border_color="#2a2050", command=lambda: subprocess.Popen(["xdg-open", self.session.dir]))
        self.rec_caption.configure(text="Open folder")

    def _summary_card(self, headline: str) -> Image.Image:
        img = Image.new("RGB", (PREVIEW_W, PREVIEW_H), (16, 16, 28))
        if self.icon:
            ic = self.icon.resize((70, int(70 * self.icon.height / self.icon.width)), Image.LANCZOS)
            img.paste(ic, (PREVIEW_W - ic.width - 12, PREVIEW_H - ic.height - 10), ic)
        d = ImageDraw.Draw(img, "RGBA")
        d.text((16, 12), "Saved — back to Claude to edit", font=self._f_badge, fill=OK)
        lines = [headline]
        try:
            sm = self.rec.pipe.summary() if self.rec and self.rec.pipe else {}
            lines.append(f"{sm.get('requests', 0)} requests · {sm.get('instant', 0)} instant · "
                         f"{sm.get('by_claude', 0)} by Claude")
            if sm.get("effects"):
                lines.append("Effects: " + ", ".join(e.replace("fx_", "") for e in sm["effects"][:6]))
            for l in sm.get("learned", [])[:3]:
                lines.append("Learned: " + (l.get("request") or l.get("meant") or l.get("kind", "")))
            lines += sm.get("tips", [])[:2]
        except Exception:
            pass
        _draw_lines(d, lines, self._f_help, 16, 48, PREVIEW_W - 32, PREVIEW_H - 6, MUTED, TEXT)
        return img

    def _set_pill(self, text: str, color: str) -> None:
        self.pill.configure(text=text, text_color=color)

    def set_status(self, text: str, color: str = MUTED) -> None:
        self.toast(text, color)

    def on_close(self) -> None:
        if getattr(self, "_ptt_keys", None):
            self._ptt_keys.stop()
        if self.state == "recording":
            self.stop()
        if self.state == "saving":
            self.set_status("Saving… the window will close when the video is safe.", WARN)
            self.root.after(300, self._close_when_saved)
            return
        if self.state in ("idle", "countdown"):
            self.session.set_status(state="cancelled", message="the recorder window was closed without recording")
        self._stop_preview()
        self.root.destroy()

    def _close_when_saved(self) -> None:
        if self.state == "saving":
            self.root.after(300, self._close_when_saved)
        else:
            self._stop_preview()
            self.root.destroy()

    # ---------- UI loop (10 fps) ----------
    def _tick(self) -> None:
        self._tickn += 1
        if self._stop_requested.is_set():
            self._stop_requested.clear()
            self.stop()
        if self._start_requested.is_set():
            self._start_requested.clear()
            if self.state == "idle":
                self.toggle()
        if self.flash:
            text, color, n = self.flash
            self.flash = (text, color, n - 1) if n > 1 else None
        if self.state in ("idle", "countdown", "recording"):
            f = self.latest
            if f is not None:
                self._show(self._decorate(Image.fromarray(f).resize((PREVIEW_W, PREVIEW_H), Image.BILINEAR)))
            elif self.state == "countdown":
                self._show(self._decorate(self._placeholder("")))
        cap = self.rec.cap if self.rec and self.state == "recording" else self.preview
        db = cap.level_db if cap else -90.0
        lvl = max(0.0, min(1.0, (db + 60) / 60))
        self.level.set(lvl)
        self.level.configure(progress_color=REC if db > -6 else (WARN if db > -14 else OK))
        if self.state == "idle" and self.preview and not self.preview.running and self.preview.error:
            self.set_status(f"Preview stopped: {self.preview.error.strip().splitlines()[-1]}", REC)
            self._show(self._placeholder("No preview for this source"))
            self.preview = None
        if self.state == "recording" and self.rec:
            t = int(self.rec.elapsed)
            self._set_pill(f"●  REC {t // 60:02d}:{t % 60:02d}" + ("  ·  FULL" if self.full_access else ""), REC)
            if self.rec.cap and not self.rec.cap.running:
                self.set_status(f"Capture stopped unexpectedly: {self.rec.cap.error or ''}", REC)
                self.stop()
        self._show_bar()
        if self._ptt.is_set():
            self._ptt.clear()
            self._push_to_talk()
        if self.full_access and self.state == "idle":
            self._set_pill("●  READY · FULL ACCESS", WARN)
        if self.live_text and self.state in ("idle", "recording"):
            self.set_status(self.live_text, self.live_color)
            self.live_text = ""
        self.root.after(100, self._tick)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="vidai-gui")
    ap.add_argument("session")
    ap.add_argument("--autostart", action="store_true")
    ap.add_argument("--auto-stop", type=float, default=0.0, help="stop after N seconds (testing)")
    ap.add_argument("--demo-marks", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    ctk.set_appearance_mode("dark")
    session = Session.load(a.session)
    root = ctk.CTk()
    app = RecorderApp(root, session, autostart=a.autostart)
    if a.auto_stop:
        def check() -> None:
            if a.demo_marks and app.state == "recording" and app.rec and not app.marks_log and app.rec.elapsed > 1.5:
                app.mark("section")
                app.mark("important")
            if app.state == "recording" and app.rec and app.rec.elapsed >= a.auto_stop:
                app.stop()
            if app.state == "done":
                root.after(4000, root.destroy)
                return
            root.after(200, check)
        root.after(200, check)
    root.mainloop()


if __name__ == "__main__":
    main()
