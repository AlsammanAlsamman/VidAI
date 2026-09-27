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
PREVIEW_W, PREVIEW_H = 480, 270
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


class RecorderApp:
    def __init__(self, root: ctk.CTk, session: Session, autostart: bool = False) -> None:
        self.root, self.session = root, session
        self.latest: np.ndarray | None = None
        self.preview: LivePipeline | None = None
        self.live_text = ""
        self.live_color = MUTED
        self.rec: SessionRecorder | None = None
        self.state = "idle"  # idle | countdown | recording | saving | done
        self.count = 0
        self.flash: tuple[str, str, int] | None = None  # (text, color, ticks left) shown on the preview
        self._hotkeys = None
        self._stop_requested = threading.Event()  # set from the hotkey thread, handled in _tick
        self._start_requested = threading.Event()  # "VidAI record" heard in preview
        self._save_thread: threading.Thread | None = None
        self._tickn = 0
        self._f_badge, self._f_count, self._f_small = _pil_font(20), _pil_font(120), _pil_font(18, bold=False)
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
            self.root.bind(key, lambda e, k=kind: self.mark(k))
        self.rec_btn = ctk.CTkButton(ctl, text="●", width=72, height=72, corner_radius=36, fg_color=REC,
                                     hover_color="#ff5c77", text_color="white", font=ctk.CTkFont(size=28),
                                     command=self.toggle, border_width=4, border_color="#3a1320")
        self.rec_btn.grid(row=0, column=2, padx=10)
        self.rec_caption = ctk.CTkLabel(ctl, text="Record  ·  space", font=self._font(11), text_color=MUTED)
        self.rec_caption.grid(row=1, column=0, columnspan=5, pady=(4, 0))
        self.root.bind("<space>", lambda e: self.toggle())

        # log / status
        self.status = ctk.CTkLabel(r, text="Ready  ·  press Record or say “VidAI record” (voice loading…)",
                                   font=self._font(11), text_color=MUTED, anchor="w", justify="left",
                                   wraplength=PREVIEW_W)
        self.status.pack(fill="x", pady=(4, 12), **pad)
        self.marks_log: list[str] = []

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
            self.preview.bus.subscribe(self._on_live, {"transcript", "voice_command", "claude", "error", "action"})
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
        self._stop_preview()
        self.session.capture.mode = self._mode_value()  # type: ignore[assignment]
        self.session.capture.mic = bool(self.mic.get())
        self.rec = SessionRecorder(self.session, on_frame=self._on_frame, on_stop_request=self._stop_requested.set)
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
                                                    "screen_text", "action"})
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
        elif k == "action" and d.get("what") == "listening_request":
            self.live_text, self.live_color = f"→ Claude (keep talking…): {d.get('so_far', '')[:70]}", ACCENT_2
        elif k == "action" and d.get("what") == "listening":
            self.live_text, self.live_color = "VidAI is listening… say the command", ACCENT
        elif k == "action" and d.get("what") == "stt_ready":
            self.live_text, self.live_color = ("voice ready: say “VidAI record” to start" if self.state == "idle"
                                               else "voice ready: say “VidAI …” (mark, new section, zoom in, stop)"), OK

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
        self._show(self._placeholder("Saved. Go back to Claude to edit."))
        self.set_status(f"{summary}  ·  {Path(self.session.dir).name}", TEXT)
        self.rec_btn.configure(state="normal", text="⤴", fg_color=ACCENT, hover_color="#7a58ff",
                               border_color="#2a2050", command=lambda: subprocess.Popen(["xdg-open", self.session.dir]))
        self.rec_caption.configure(text="Open folder")

    def _set_pill(self, text: str, color: str) -> None:
        self.pill.configure(text=text, text_color=color)

    def set_status(self, text: str, color: str = MUTED) -> None:
        self.status.configure(text=text, text_color=color)

    def on_close(self) -> None:
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
            self._set_pill(f"●  REC {t // 60:02d}:{t % 60:02d}", REC)
            if self.rec.cap and not self.rec.cap.running:
                self.set_status(f"Capture stopped unexpectedly: {self.rec.cap.error or ''}", REC)
                self.stop()
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
