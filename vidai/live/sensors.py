"""Live sensors: turn the raw audio/video streams into stats events on the bus.

AudioSensor   level (10 Hz), speech_start/speech_end, silence_start/silence_end, loud; keeps utterance audio for STT
MotionSensor  motion (2 Hz), scene_change
SpeechToText  transcript + voice_command (+ claude) events, Whisper running in a worker thread
ScreenText    screen_text events (OCR every few seconds when the picture changed)
"""
from __future__ import annotations

import difflib
import queue
import re
import shutil
import subprocess
import threading
from collections import deque
from typing import Any

import numpy as np

from .. import native

SR = 16000
BLOCK = SR // 10  # 100 ms


class AudioSensor:
    def __init__(self, bus, min_silence: float = 0.5, silence_db: float | None = None, keep_seconds: float = 40.0,
                 on_utterance=None, utterance_gap: float | None = None) -> None:
        self.bus = bus
        self.min_silence = min_silence
        self.utterance_gap = utterance_gap or min_silence  # end of a spoken sentence (faster than anchor pauses)
        self.fixed_db = silence_db
        self.levels: list[float] = []  # full history at 10 Hz (becomes the audio_level anchor series)
        self.ring = np.zeros(int(SR * keep_seconds), np.float32)
        self.written = 0  # total samples seen
        self.speaking = False
        self.run_quiet = 0
        self.run_loud = 0
        self.speech_start_t = 0.0
        self.silence_start_t = 0.0
        self.on_utterance = on_utterance  # fn(samples, start_t, end_t)
        self._last_loud = -1e9
        self.silences: list[tuple[float, float]] = []
        self.speech: list[tuple[float, float]] = []

    @property
    def t(self) -> float:
        return self.written / SR

    def threshold(self) -> float:
        if self.fixed_db is not None:
            return self.fixed_db
        hist = self.levels[-300:]  # last 30 s
        if len(hist) < 10:
            return -45.0
        floor, loud = np.percentile(hist, 10), np.percentile(hist, 95)
        return float(min(floor + 0.35 * (loud - floor), -30.0) if loud - floor > 10 else max(floor + 6, -50.0))

    def feed(self, block: np.ndarray) -> None:
        n = block.size
        pos = self.written % self.ring.size
        first = min(n, self.ring.size - pos)
        self.ring[pos:pos + first] = block[:first]
        if first < n:
            self.ring[: n - first] = block[first:]
        self.written += n
        db = float(native.rms_db(block, n)[0]) if n else -90.0
        self.levels.append(round(db, 1))
        t = self.t
        self.bus.publish("level", {"db": round(db, 1)}, t)
        if db > -3 and t - self._last_loud > 1.0:
            self._last_loud = t
            self.bus.publish("loud", {"db": round(db, 1)}, t)
        thr = self.threshold()
        need_quiet = max(1, int(round(self.utterance_gap * 10)))
        if db >= thr:
            self.run_loud += 1
            self.run_quiet = 0
        else:
            self.run_quiet += 1
            self.run_loud = 0
        if not self.speaking and self.run_loud >= 2:
            self.speaking = True
            start = t - 0.2
            if self.silence_start_t is not None and start - self.silence_start_t >= self.min_silence:
                self.silences.append((self.silence_start_t, start))
                self.bus.publish("silence_end", {"duration": round(start - self.silence_start_t, 2),
                                                 "start": round(self.silence_start_t, 2)}, start)
            self.speech_start_t = start
            self.bus.publish("speech_start", {}, start)
        elif self.speaking and (self.run_quiet >= need_quiet or t - self.speech_start_t > 20):
            self.speaking = False
            end = t - self.run_quiet * 0.1
            self.speech.append((self.speech_start_t, end))
            self.bus.publish("speech_end", {"duration": round(end - self.speech_start_t, 2),
                                            "start": round(self.speech_start_t, 2)}, end)
            self.silence_start_t = end
            self.bus.publish("silence_start", {}, end)
            if self.on_utterance:
                self.on_utterance(self.slice(self.speech_start_t - 0.3, end + 0.2), self.speech_start_t, end)

    def slice(self, t0: float, t1: float) -> np.ndarray:
        s0, s1 = max(0, int(t0 * SR), self.written - self.ring.size), min(self.written, int(t1 * SR))
        if s1 <= s0:
            return np.zeros(0, np.float32)
        idx = np.arange(s0, s1) % self.ring.size
        return self.ring[idx].copy()

    def finish(self) -> None:
        t = self.t
        if self.speaking:
            self.speech.append((self.speech_start_t, t))
        elif t - self.silence_start_t >= self.min_silence:
            self.silences.append((self.silence_start_t, t))


class MotionSensor:
    def __init__(self, bus, rate_hz: float = 2.0, scene_threshold: float = 0.12) -> None:
        self.bus, self.rate, self.thr = bus, rate_hz, scene_threshold
        self.values: list[float] = []
        self.prev: np.ndarray | None = None
        self._hist: deque[float] = deque(maxlen=3)

    def feed(self, small_rgb: np.ndarray, t: float) -> None:
        g = np.ascontiguousarray(small_rgb[::2, ::2].mean(axis=2).astype(np.uint8))
        v = 0.0 if self.prev is None else float(native.frame_mad(np.stack([self.prev, g]))[1])
        self.prev = g
        self.values.append(round(v, 4))
        self.bus.publish("motion", {"value": round(v, 4)}, t)
        self._hist.append(v)
        if len(self._hist) == 3 and self._hist[1] >= self.thr and self._hist[1] >= self._hist[0] and \
                self._hist[1] >= self._hist[2]:
            self.bus.publish("scene_change", {"score": round(self._hist[1], 3)}, t - 1.0 / self.rate)


# ---------------- speech to text + voice commands ----------------
WAKE = ["vidai", "vid ai", "video ai", "vid-ai", "vidia", "vidi", "veedai", "fidai", "فيداي", "في داي", "فيدي", "فيديو اي"]

COMMANDS: list[tuple[str, list[str]]] = [  # (command, trigger phrases) — English + Arabic
    ("record", ["start recording", "starts recording", "start the recording", "begin recording", "recording",
                "record", "start", "ابدأ التسجيل", "ابدأ", "سجل"]),
    ("full_access", ["take all actions", "take all the actions", "full access", "you have my permission",
                     "do everything", "all permissions", "you have full access"]),
    ("ask_first", ["ask me first", "ask first", "ask permission", "ask for permission"]),
    ("talk", ["talk", "talk to me", "let's talk", "lets talk", "i have a question", "can i ask you",
              "question", "answer me", "too", "tok", "torque", "taught", "talked", "tuck"]),
    ("undo", ["undo", "undo that", "go back", "take that back", "revert"]),
    ("redo", ["redo", "redo that", "do it again"]),
    ("help", ["help", "what can i say", "what can you do", "show commands", "commands"]),
    ("lighter", ["lighter", "light mode", "go lighter", "be faster", "faster"]),
    ("confirm", ["confirm", "confirmed", "yes", "yes please", "go ahead", "do it", "approve", "approved", "allow",
                 "okay", "ok"]),
    ("deny", ["deny", "denied", "no", "no thanks", "cancel", "don't", "do not", "reject"]),
    ("stop", ["stop recording", "stop", "إيقاف", "توقف"]),
    ("mistake", ["mistake", "cut that", "again", "redo", "خطأ", "غلط", "إعادة"]),
    ("section", ["new section", "section", "chapter", "قسم جديد", "قسم", "فصل"]),
    ("important", ["important", "highlight", "مهم"]),
    ("zoom_in", ["zoom in", "zoom", "تكبير"]),
    ("zoom_out", ["zoom out", "reset zoom", "unzoom", "تصغير"]),
    ("captions_on", ["captions on", "subtitles on", "ترجمة"]),
    ("captions_off", ["captions off", "subtitles off", "بدون ترجمة"]),
    ("label", ["label", "this is", "هذا"]),
    ("wrong", ["wrong", "no that's wrong", "خطأ النموذج"]),
    ("learn", ["learn", "تعلم"]),
    ("marker", ["mark", "marker", "علامة"]),
    ("claude", ["claude", "كلود"]),
]


def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w\s؀-ۿ-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def parse_command(text: str, wake_words: list[str] | None = None) -> dict | None:
    """'VidAI, new section results' -> {"command": "section", "args": "results"}; None if not addressed to VidAI."""
    words = _norm(text).split()
    wake = [_norm(w) for w in (wake_words or WAKE)]
    n_found = next((n for n in (3, 2, 1) if " ".join(words[:n]) in wake), None)  # exact wake phrase first
    if n_found is None:
        n_found = next((n for n in (1, 2, 3) if any(
            difflib.SequenceMatcher(None, " ".join(words[:n]), w).ratio() >= 0.8 for w in wake)), None)
    if n_found is None:
        return None
    rest = " ".join(words[n_found:])
    # longest phrase first, across all commands ("zoom out" before "zoom", "خطأ النموذج" before "خطأ")
    for cmd, p in sorted(((c, p) for c, ps in COMMANDS for p in ps), key=lambda cp: -len(cp[1])):
        if rest == p or rest.startswith(p + " "):
            return {"command": cmd, "args": rest[len(p):].strip()}
    return {"command": "claude", "args": rest}  # anything else is an instruction for Claude


# Whisper's well-known inventions on silence / noise (whole-utterance matches only)
HALLUCINATIONS = {"thank you for watching", "thanks for watching", "thank you", "thanks", "i m sorry", "sorry",
                  "you", "bye", "bye bye", "subscribe", "please subscribe", "like and subscribe", "okay", "oh",
                  "so", "hmm", "uh", "um", "the end", "music", "applause", "silence"}


def is_hallucination(text: str) -> bool:
    t = _norm(text)
    if not t or t in HALLUCINATIONS:
        return True
    if re.search(r"www\.|https?:|\.com|\.info|\.org", text.lower()):
        return True
    if re.fullmatch(r"[\d\s.,:-]+", t):  # "3. 3. 3. 4."
        return True
    words = t.split()
    if len(words) >= 6 and len(set(words)) <= len(words) / 3:  # heavy repetition
        return True
    if len(t) >= 20 and " " not in t:  # "dhidhadidhaddhad..."
        return True
    return False


_MODELS: dict[str, Any] = {}
_MODEL_LOCK = threading.Lock()


class SpeechToText:
    """Transcribes utterances (cut by AudioSensor at pauses) in a worker thread."""

    def __init__(self, bus, model: str = "base", language: str | None = None, wake_words: list[str] | None = None,
                 threads: int = 4, allowed_languages: list[str] | None = None) -> None:
        self.bus, self.model_name, self.language = bus, model, language
        # auto-detection is unreliable on short commands ("zoom in" -> German); pick only among these
        self.allowed = [l for l in (allowed_languages or []) if l]
        self.wake_words = wake_words
        self.threads = threads
        self.q: queue.Queue = queue.Queue(maxsize=20)
        self.model = None
        self.enabled = True
        self.transcripts: list[dict] = []
        self.profile = None  # vidai.profile.Profile: user's words (recognition) and learned corrections
        self.armed_until = -1.0  # after a bare "VidAI", the next utterance within a few seconds is the command
        self.arm_seconds = 5.0
        self.min_speech = 0.35
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def _load(self) -> None:
        with _MODEL_LOCK:  # one Whisper model per process, shared by preview and recording
            if self.model_name not in _MODELS:
                from faster_whisper import WhisperModel

                _MODELS[self.model_name] = WhisperModel(self.model_name, device="cpu", compute_type="int8",
                                                        cpu_threads=self.threads)
        self.model = _MODELS[self.model_name]
        self.bus.publish("action", {"what": "stt_ready", "model": self.model_name})

    def submit(self, samples: np.ndarray, start: float, end: float) -> None:
        if end - start < self.min_speech:  # clicks, breaths, keyboard: never real words
            return
        if self.enabled and samples.size > SR * 0.3:
            try:
                self.q.put_nowait((samples, start, end))
            except queue.Full:
                self.bus.publish("error", {"where": "stt", "error": "speech-to-text queue full, skipping"})

    def _run(self) -> None:
        try:
            self._load()
        except Exception as e:
            self.enabled = False
            self.bus.publish("error", {"where": "stt", "error": f"cannot load Whisper: {e!r}"[:300]})
            return
        while True:
            item = self.q.get()
            if item is None:
                return
            samples, start, end = item
            try:
                lang = self.language
                english_only = self.model_name.endswith(".en")
                if english_only:
                    lang = None  # .en models are English by design (no language option)
                if lang is None and self.allowed and not english_only:
                    _, _, probs = self.model.detect_language(samples)
                    p = dict(probs)
                    lang = max(self.allowed, key=lambda l: p.get(l, 0.0))
                # bias decoding toward the wake word, otherwise "VidAI" comes out as "VidI" / "We die"
                hot = "VidAI فيداي" if lang in (None, "ar") and not english_only else "VidAI"  # noqa: RUF001
                if self.profile is not None:
                    hot = hot.replace("VidAI", self.profile.hotwords())
                segs, info = self.model.transcribe(samples, language=lang, beam_size=1, vad_filter=True,
                                                   condition_on_previous_text=False, hotwords=hot)
                kept = [x.text.strip() for x in segs
                        if not (x.no_speech_prob > 0.6 and x.avg_logprob < -0.5)
                        and x.avg_logprob > -1.0 and x.compression_ratio < 2.4]
                text = " ".join(kept).strip()
            except Exception as e:
                self.bus.publish("error", {"where": "stt", "error": repr(e)[:300]})
                continue
            self.handle_text(text, start, end, info.language)

    def handle_text(self, text: str, start: float, end: float, lang: str | None = None) -> None:
        """Publish a transcript and, if it is addressed to VidAI, a voice command (also handles
        'VidAI' <pause> 'zoom in' as one command)."""
        if not text or re.fullmatch(r"[\W_]*", text) or is_hallucination(text):
            return
        if getattr(self, "profile", None) is not None:
            text = self.profile.correct(text)
        cmd = parse_command(text, self.wake_words)
        if cmd and cmd["command"] == "claude" and not cmd["args"]:  # just "VidAI": listen for the command
            self.armed_until = end + self.arm_seconds
            ev = {"text": text, "start": round(start, 2), "end": round(end, 2), "lang": lang, "is_command": True}
            self.transcripts.append(ev)
            self.bus.publish("transcript", ev, end)
            self.bus.publish("action", {"what": "listening", "until": round(self.armed_until, 2)}, end)
            return
        if cmd is None and start <= self.armed_until:
            cmd = parse_command("vidai " + text, None)
        self.armed_until = -1.0 if cmd else self.armed_until
        ev = {"text": text, "start": round(start, 2), "end": round(end, 2), "lang": lang,
              "is_command": cmd is not None}
        self.transcripts.append(ev)
        self.bus.publish("transcript", ev, end)
        if cmd:
            self.bus.publish("voice_command", {**cmd, "text": text}, end)

    def flush(self, timeout: float = 30.0) -> None:
        """Wait until queued utterances are transcribed (at the end of a recording)."""
        import time

        t0 = time.time()
        while (not self.q.empty() or self.model is None) and time.time() - t0 < timeout and self.enabled:
            time.sleep(0.1)
        time.sleep(0.2)

    def close(self) -> None:
        self.q.put(None)
        self._th.join(10)


# ---------------- on-screen text ----------------
class ScreenText:
    """OCR of the current frame every `interval` s when the picture changed. Tesseract (Arabic+English)
    is used if installed, else RapidOCR (English/Chinese, bundled)."""

    def __init__(self, bus, interval: float = 2.0, langs: str = "eng", width: int = 1280) -> None:
        self.bus, self.interval, self.langs, self.width = bus, interval, langs, width
        self.last_t = -1e9
        self.last_lines: list[str] = []
        self.pending = False
        self.changed_since = True
        self.enabled = True
        self.engine = "tesseract" if shutil.which("tesseract") else "rapidocr"
        self._ocr = None
        self.q: queue.Queue = queue.Queue(maxsize=1)
        self.results: list[dict] = []
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def want(self, t: float) -> bool:
        return self.enabled and not self.pending and self.changed_since and t - self.last_t >= self.interval

    def notify_motion(self, value: float) -> None:
        if value > 0.004:
            self.changed_since = True

    def submit(self, frame: np.ndarray, t: float) -> None:
        import cv2

        h, w = frame.shape[:2]
        small = cv2.resize(frame, (self.width, int(h * self.width / w))) if w > self.width else frame.copy()
        self.pending, self.last_t, self.changed_since = True, t, False
        try:
            self.q.put_nowait((small, t))
        except queue.Full:
            self.pending = False

    def _read(self, img: np.ndarray) -> list[str]:
        if self.engine == "tesseract":
            import cv2

            ok, png = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            r = subprocess.run(["tesseract", "stdin", "stdout", "-l", self.langs, "--psm", "3"], input=png.tobytes(),
                               capture_output=True)
            return [l.strip() for l in r.stdout.decode(errors="replace").splitlines() if l.strip()]
        if self._ocr is None:
            from rapidocr_onnxruntime import RapidOCR

            self._ocr = RapidOCR()
        res, _ = self._ocr(img)
        return [r[1].strip() for r in (res or []) if r[2] > 0.5 and r[1].strip()]

    def _run(self) -> None:
        while True:
            item = self.q.get()
            if item is None:
                return
            img, t = item
            try:
                lines = self._read(img)
            except Exception as e:
                self.enabled = False
                self.bus.publish("error", {"where": "ocr", "error": repr(e)[:300]})
                continue
            finally:
                self.pending = False
            old = set(self.last_lines)
            new = [l for l in lines if l not in old]
            union = old | set(lines)
            similarity = len(old & set(lines)) / len(union) if union else 1.0
            if similarity < 0.8:
                ev = {"text": "\n".join(lines), "lines": lines, "new_lines": new, "engine": self.engine}
                self.results.append({"t": t, **ev})
                self.bus.publish("screen_text", ev, t)
                self.last_lines = lines

    def close(self) -> None:
        self.enabled = False
        try:
            self.q.get_nowait()  # drop a pending frame
        except queue.Empty:
            pass
        self.q.put(None)
        self._th.join(5)  # let a running OCR call finish (onnxruntime aborts if killed at exit)
