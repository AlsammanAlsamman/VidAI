"""Live stats bus: every live statistic, command and action while recording is an event on this bus.

- In process: processors, rules and learners subscribe and react within milliseconds.
- For Claude: events are appended to `<session>/live.jsonl`, which Claude reads with `live_stats`
  (the slow loop), and Claude answers through `<session>/control.jsonl` (see pipeline.py).
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

# Event kinds published by VidAI itself (processors/learners may add their own):
#   level          {db}                        10 Hz, not written to live.jsonl (too chatty); see `latest`
#   silence_start  {}                          silence_end {duration}
#   speech_start   {}                          speech_end {duration}
#   loud           {db}                        clipping / shouting
#   motion         {value}                     2 Hz, latest only
#   scene_change   {score}
#   transcript     {text, start, end, lang}    speech-to-text of one utterance
#   voice_command  {command, args, text}       "vidai <command> ..." heard in the transcript
#   screen_text    {text, lines, changed}      OCR of what is on screen / in frame
#   marker         {type, note, source}        hotkey, button, voice or Claude
#   learner        {name, label, confidence}   an instant model changed its prediction
#   action         {what, ...}                 a rule/processor/command did something to the video
#   claude         {message}                   voice/user instruction addressed to Claude
#   ack / error    {command, ...}              result of a control command
#   perf           {fps, frame_ms, dropped_processors}
QUIET_KINDS = {"level", "motion"}


class LiveBus:
    def __init__(self, log_path: str | Path | None = None, clock: Callable[[], float] | None = None,
                 history: int = 5000) -> None:
        self.clock = clock or (lambda: 0.0)
        self.log_path = Path(log_path) if log_path else None
        self._subs: list[tuple[set[str] | None, Callable[[dict], None]]] = []
        self._lock = threading.Lock()
        self._seq = 0
        self.history: deque[dict] = deque(maxlen=history)
        self.latest: dict[str, dict] = {}
        self._pending: list[str] = []
        if self.log_path and self.log_path.exists():  # continue numbering (preview, then recording, same log)
            last = read_events(self.log_path, 0, None, 1)
            self._seq = last[-1]["seq"] if last else 0
        self._fh = open(self.log_path, "a", encoding="utf-8") if self.log_path else None
        self._last_flush = time.monotonic()

    def subscribe(self, fn: Callable[[dict], None], kinds: set[str] | list[str] | None = None) -> None:
        self._subs.append((set(kinds) if kinds else None, fn))

    def publish(self, kind: str, data: dict[str, Any] | None = None, t: float | None = None) -> dict:
        with self._lock:
            self._seq += 1
            ev = {"seq": self._seq, "t": round(self.clock() if t is None else t, 3), "kind": kind, "data": data or {}}
            self.latest[kind] = ev
            if kind not in QUIET_KINDS:
                self.history.append(ev)
                if self._fh:
                    self._pending.append(json.dumps(ev, ensure_ascii=False))
                    self._maybe_flush()
        for kinds, fn in list(self._subs):
            if kinds is None or kind in kinds:
                try:
                    fn(ev)
                except Exception as e:  # a bad subscriber must never break the recording
                    if kind != "error":
                        self.publish("error", {"where": getattr(fn, "__qualname__", str(fn)), "error": repr(e)})
        return ev

    def _maybe_flush(self, force: bool = False) -> None:
        if self._fh and self._pending and (force or time.monotonic() - self._last_flush > 0.3):
            self._fh.write("\n".join(self._pending) + "\n")
            self._fh.flush()
            self._pending.clear()
            self._last_flush = time.monotonic()

    def flush(self) -> None:
        with self._lock:
            self._maybe_flush(force=True)

    def close(self) -> None:
        self.flush()
        if self._fh:
            self._fh.close()
            self._fh = None

    def snapshot(self) -> dict[str, Any]:
        """Latest value of every stat (what processors and Claude see as 'now')."""
        return {k: v["data"] for k, v in self.latest.items()}


def read_events(log_path: str | Path, since: int = 0, kinds: list[str] | None = None,
                limit: int = 200) -> list[dict]:
    p = Path(log_path)
    if not p.exists():
        return []
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev["seq"] > since and (not kinds or ev["kind"] in kinds):
                out.append(ev)
    return out[-limit:]
