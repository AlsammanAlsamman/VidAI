"""Performance governor: keep the recording at full frame rate under load."""
from __future__ import annotations

import time

from .common import PREVIEW_EVERY


class GovernorMixin:
    """Part of LivePipeline (see pipeline.py); uses its state."""

    def _govern(self) -> None:
        """Keep the recording at full frame rate. Pressure = falling behind real time or frames near the budget."""
        if self._t0_wall is None or time.monotonic() - self._t0_wall < 3:
            return
        now = time.monotonic()
        budget = 1000.0 / self.fps
        pressure = self.lag_frames > 12 or self.frame_ms_ema > 0.75 * budget
        relaxed = self.lag_frames < 4 and self.frame_ms_ema < 0.4 * budget
        if pressure:
            self._pressure_since = self._pressure_since or now
            if self.level < 2 and now - self._level_since > 1.5:
                self._set_level(self.level + 1)
            elif self.level == 2 and now - self._pressure_since > 8 and self.lag_frames > 30:
                heavy = [p for p in self.chain.items if p.enabled and p.calls and not p.name.startswith("_")]
                if heavy:
                    worst = max(heavy, key=lambda p: p.total_ms / p.calls)
                    worst.enabled = False
                    self.bus.publish("warning", {"what": "performance", "text":
                                     f"Turned off '{worst.name}' to keep the video smooth "
                                     f"({worst.total_ms / worst.calls:.0f} ms per frame). Say 'VidAI, lighter' or "
                                     f"remove other effects, then turn it on again."})
                    self._pressure_since = now
        else:
            self._pressure_since = None
            if relaxed and self.level > 0 and now - self._level_since > 6:
                self._set_level(self.level - 1)

    def _set_level(self, level: int) -> None:
        self.level = level
        self._level_since = time.monotonic()
        self.preview_every = PREVIEW_EVERY * (1 + level)  # fewer preview frames under load
        if self.ctx.tracks is not None:
            self.ctx.tracks.max_hz = (30, 15, 8)[level]
        if self.ocr:
            self.ocr.interval = (self.live.ocr_interval, self.live.ocr_interval * 2, self.live.ocr_interval * 4)[level]
        self.ctx.quality = level  # effects may read it (e.g. run heavy models less often)
        text = ("Performance: normal", "Performance: light mode (tracking and preview slower) to keep 30 fps",
                "Performance: minimal mode — the computer is busy; fewer effects will keep the video smooth")[level]
        self.bus.publish("warning" if level else "action", {"what": "performance", "level": level, "text": text})
