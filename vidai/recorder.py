"""Record with OBS (via obs-websocket) while live samplers write the selected anchor stats.

OBS setup: Tools -> WebSocket Server Settings -> Enable (default port 4455), set a password.

Hotkeys during recording (global):
  ctrl+alt+m  marker          ctrl+alt+x  mistake (Claude may cut the take before it)
  ctrl+alt+n  new section     ctrl+alt+i  important (zoom/highlight candidate)
"""
from __future__ import annotations

import shutil
import subprocess
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from .anchors import AnchorFile, Event, Series, select_stats
from .brief import Brief

HOTKEYS = {"<ctrl>+<alt>+m": "marker", "<ctrl>+<alt>+x": "mistake",
           "<ctrl>+<alt>+n": "section", "<ctrl>+<alt>+i": "important"}


class Sampler:
    """A live stat collector. Subclasses fill `events` or `counts` using `self.now()`."""

    stat = ""

    def __init__(self, clock: Callable[[], float]) -> None:
        self.now = clock
        self.events: list[Event] = []
        self._stop = threading.Event()
        self.bus = None  # optional live bus: events are also published there

    def _emit(self, ev: Event) -> None:
        self.events.append(ev)
        if self.bus is not None:
            self.bus.publish(ev.kind, dict(ev.data), ev.t)

    def start(self) -> None: ...
    def stop(self) -> None:
        self._stop.set()

    def result(self, duration: float) -> tuple[list[Event], Series | None]:
        return self.events, None


class MarkerSampler(Sampler):
    stat = "markers"

    def start(self) -> None:
        try:
            from pynput import keyboard

            self._hk = keyboard.GlobalHotKeys({k: (lambda t=t: self.mark(t)) for k, t in HOTKEYS.items()})
            self._hk.start()
        except Exception as e:  # no X display, no permission, ...
            self._hk = None
            print(f"[vidai] hotkeys unavailable: {e}")

    def mark(self, kind: str, note: str = "") -> None:
        self.events.append(Event(t=round(self.now(), 3), kind="markers", data={"type": kind, "note": note}))

    def stop(self) -> None:
        super().stop()
        if getattr(self, "_hk", None):
            self._hk.stop()


class InputActivitySampler(Sampler):
    """Counts keyboard + mouse events per second (no key contents are stored)."""

    stat = "input_activity"

    def __init__(self, clock: Callable[[], float]) -> None:
        super().__init__(clock)
        self.counts: Counter[int] = Counter()

    def hit(self, *_: Any) -> None:
        self.counts[int(self.now())] += 1

    def start(self) -> None:
        try:
            from pynput import keyboard, mouse

            self._k = keyboard.Listener(on_press=self.hit)
            self._m = mouse.Listener(on_click=lambda *a: self.hit(), on_scroll=lambda *a: self.hit())
            self._k.start()
            self._m.start()
        except Exception as e:
            self._k = self._m = None
            print(f"[vidai] input activity unavailable: {e}")

    def stop(self) -> None:
        super().stop()
        for l in (getattr(self, "_k", None), getattr(self, "_m", None)):
            if l:
                l.stop()

    def result(self, duration: float) -> tuple[list[Event], Series | None]:
        n = int(duration) + 1
        return [], Series(rate_hz=1.0, values=[float(self.counts.get(i, 0)) for i in range(n)])


def active_window_title() -> str | None:
    if shutil.which("xdotool"):
        r = subprocess.run(["xdotool", "getactivewindow", "getwindowname"], capture_output=True, text=True)
        return r.stdout.strip() or None
    if shutil.which("xprop"):
        r = subprocess.run(["xprop", "-root", "_NET_ACTIVE_WINDOW"], capture_output=True, text=True)
        wid = r.stdout.strip().split()[-1] if r.stdout.strip() else ""
        if wid and wid != "0x0":
            r = subprocess.run(["xprop", "-id", wid, "WM_NAME"], capture_output=True, text=True)
            if "=" in r.stdout:
                return r.stdout.split("=", 1)[1].strip().strip('"')
    return None


class WindowFocusSampler(Sampler):
    stat = "window_focus"

    def __init__(self, clock: Callable[[], float], interval: float = 0.5,
                 probe: Callable[[], str | None] = active_window_title) -> None:
        super().__init__(clock)
        self.interval, self.probe = interval, probe

    def start(self) -> None:
        def loop() -> None:
            last = None
            while not self._stop.is_set():
                title = self.probe()
                if title and title != last:
                    self._emit(Event(t=round(self.now(), 3), kind="window_focus", data={"title": title}))
                    last = title
                self._stop.wait(self.interval)

        self._th = threading.Thread(target=loop, daemon=True)
        self._th.start()


class OBSSceneSampler(Sampler):
    stat = "obs_scene"

    def __init__(self, clock: Callable[[], float], event_client_factory: Callable[[], Any] | None) -> None:
        super().__init__(clock)
        self.factory = event_client_factory

    def start(self) -> None:
        if not self.factory:
            return
        try:
            self._ec = self.factory()

            def on_current_program_scene_changed(data: Any) -> None:
                self.events.append(Event(t=round(self.now(), 3), kind="obs_scene",
                                         data={"scene": getattr(data, "scene_name", str(data))}))

            self._ec.callback.register(on_current_program_scene_changed)
        except Exception as e:
            self._ec = None
            print(f"[vidai] OBS scene events unavailable: {e}")

    def stop(self) -> None:
        super().stop()
        if getattr(self, "_ec", None):
            try:
                self._ec.disconnect()
            except Exception:
                pass


class Recorder:
    """Controls OBS and the samplers. `client` may be injected (tests use a fake)."""

    def __init__(self, brief: Brief | None = None, stats: dict[str, str] | list[str] | None = None,
                 host: str = "localhost", port: int = 4455, password: str = "",
                 client: Any = None, event_client_factory: Callable[[], Any] | None = None,
                 window_probe: Callable[[], str | None] = active_window_title) -> None:
        self.brief = brief or Brief()
        if stats is None:
            stats = select_stats(self.brief)
        self.stats = stats if isinstance(stats, dict) else {s: "" for s in stats}
        self.host, self.port, self.password = host, port, password
        self._client = client
        self._ec_factory = event_client_factory
        self._window_probe = window_probe
        self.samplers: list[Sampler] = []
        self._t0 = 0.0
        self.recording = False

    @property
    def client(self) -> Any:
        if self._client is None:
            import obsws_python as obs

            self._client = obs.ReqClient(host=self.host, port=self.port, password=self.password, timeout=5)
            if self._ec_factory is None:
                self._ec_factory = lambda: obs.EventClient(host=self.host, port=self.port, password=self.password)
        return self._client

    def clock(self) -> float:
        return time.monotonic() - self._t0

    def _make_samplers(self) -> list[Sampler]:
        s: list[Sampler] = [MarkerSampler(self.clock)]  # markers are always on
        if "input_activity" in self.stats:
            s.append(InputActivitySampler(self.clock))
        if "window_focus" in self.stats:
            s.append(WindowFocusSampler(self.clock, probe=self._window_probe))
        if "obs_scene" in self.stats:
            s.append(OBSSceneSampler(self.clock, self._ec_factory))
        return s

    def start(self) -> None:
        client = self.client
        client.start_record()
        self._t0 = time.monotonic()
        self.recording = True
        self.samplers = self._make_samplers()
        for s in self.samplers:
            s.start()

    def mark(self, kind: str = "marker", note: str = "") -> None:
        for s in self.samplers:
            if isinstance(s, MarkerSampler):
                s.mark(kind, note)

    def stop(self, analyze_after: bool = True, workers: int = 3) -> AnchorFile:
        duration = self.clock()
        res = self.client.stop_record()
        self.recording = False
        for s in self.samplers:
            s.stop()
        video = getattr(res, "output_path", None) or ""
        anchors = AnchorFile(video=video, duration=round(duration, 3), brief=self.brief,
                             selected_stats=dict(self.stats),
                             recorded_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        for s in self.samplers:
            evs, series = s.result(duration)
            if evs:
                anchors.add_events(evs)
            if series is not None:
                anchors.series[s.stat] = series
        if video and Path(video).exists():
            time.sleep(0.5)  # let OBS finish the file
            if analyze_after:
                from .analyze import analyze

                anchors = analyze(video, self.stats, workers=workers, anchors=anchors)
            anchors.save()
        return anchors
