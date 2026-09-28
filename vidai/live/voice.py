"""VidAI's voice: offline neural text-to-speech (Piper), played on the speakers AND mixed cleanly into the video.

- Piper (free, offline, ~16x faster than real time on CPU); the voice model downloads itself (models_dl).
- Falls back to speech-dispatcher (spd-say) when Piper is missing: then the voice reaches the video only
  through the microphone.
- Every clip is kept (start time on the recording clock + wav file) so the final mux mixes it into the audio
  track, and the microphone is turned down under it (no echo, works with headphones too).
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
import wave
from pathlib import Path

VOICE = "en_US-lessac-medium.onnx"


class Voice:
    def __init__(self, folder: str | Path | None = None) -> None:
        self.dir = Path(folder) if folder else Path(tempfile.mkdtemp(prefix="vidai_voice_"))
        self.dir.mkdir(parents=True, exist_ok=True)
        self._piper = None
        self._lock = threading.Lock()
        self.engine = "none"
        self.error: str | None = None
        self.n = 0
        try:
            import piper  # noqa: F401

            self.engine = "piper"
        except ImportError:
            self.engine = "spd-say" if shutil.which("spd-say") else "none"

    def _load(self):
        if self._piper is None:
            from piper import PiperVoice

            from ..models_dl import ensure

            ensure(VOICE + ".json")
            self._piper = PiperVoice.load(str(ensure(VOICE)))
        return self._piper

    def synth(self, text: str) -> tuple[Path, float] | None:
        """Text -> (wav path, seconds). None if Piper is not available."""
        if self.engine != "piper":
            return None
        with self._lock:
            try:
                v = self._load()
            except Exception as e:  # model download failed, ...: fall back to spd-say
                self.error = repr(e)[:200]
                self.engine = "spd-say" if shutil.which("spd-say") else "none"
                return None
            self.n += 1
            path = self.dir / f"say_{self.n:03d}.wav"
            with wave.open(str(path), "wb") as w:
                v.synthesize_wav(text, w)
        with wave.open(str(path)) as r:
            dur = r.getnframes() / r.getframerate()
        return path, dur

    @staticmethod
    def play(path: Path) -> None:
        player = shutil.which("paplay") or shutil.which("aplay")
        if player:
            subprocess.run([player, str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)

    @staticmethod
    def speak_fallback(text: str) -> None:
        if shutil.which("spd-say"):
            subprocess.run(["spd-say", "-w", "-r", "5", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=60)
