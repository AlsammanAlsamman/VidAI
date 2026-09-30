"""Talking with the user: voice commands, requests (fast path or Claude), questions, "VidAI talk",
"VidAI suggest", the thinking icon and VidAI's own voice. Messages here are never recorded."""
from __future__ import annotations

import threading
import time

import numpy as np

from .common import PREVIEW_W
from .processors import REGISTRY


class DialogMixin:
    """Part of LivePipeline (see pipeline.py); uses its state."""

    def _on_claude_request(self, ev: dict) -> None:
        if ev["data"].get("source") not in ("voice", "user", "gui", "talk", "typed"):
            return  # rules' notify_claude does not show the icon
        self.pending[ev["seq"]] = {"t": ev["t"], "message": ev["data"].get("message", "")}
        self._show_thinking(True)

    def _show_thinking(self, on: bool) -> None:
        mode = self.live.thinking
        if mode == "off":
            return
        if self._thinking is None:
            self._thinking = REGISTRY["thinking"]("_thinking", {"position": self.live.thinking_position}, False)
            for w in (self.W, PREVIEW_W):  # build the animation here (bus thread), not in the frame loop
                self._thinking._sprites[w] = self._thinking._build(w)
            if mode == "video":
                self.chain.add(self._thinking)
        if on and not self._thinking.enabled:
            self._thinking.shown_at = None
        self._thinking.enabled = on if mode == "video" else False
        self.bus.publish("action", {"what": "thinking_on" if on else "thinking_off", "pending": len(self.pending)})

    def _on_utterance(self, samples: np.ndarray, start: float, end: float) -> None:
        if start < self.speaking_until:  # that was VidAI's own voice from the speakers
            self.bus.publish("action", {"what": "ignored_own_voice", "start": round(start, 2)})
            return
        if self.stt:
            self.stt.submit(samples, start, end)

    def _on_voice(self, ev: dict) -> None:
        d = ev["data"]
        cmd, args = d["command"], d.get("args", "")
        if cmd == "claude" and self.questions and args.strip():  # maybe an answer to Claude's question
            q = self.questions[list(self.questions)[-1]]
            if not q["options"] or _match_option(args, q["options"]) in q["options"]:
                self.command({"cmd": "answer", "text": args}, source="voice")
                return
            # not one of the options: it's a new request, the question stays open
        if cmd in ("undo", "redo", "help", "lighter"):
            self.command({"cmd": cmd}, source="voice")
            return
        if cmd == "talk":
            self.command({"cmd": "talk"}, source="voice")
            return
        if cmd == "claude":
            if args.strip():  # an empty request is never sent to Claude
                self._req = {"parts": [args.strip()], "due": time.monotonic() + self.request_gap,
                             "waiting": False, "cap": time.monotonic() + 20.0, "seq": ev["seq"],
                             "source": "talk" if self._talk else "voice"}
                self._talk = False
                self.bus.publish("action", {"what": "listening_request", "so_far": args.strip()})
                self._flush_if_obvious()
            return
        if not self.live.voice_actions:
            return
        if cmd in ("marker", "mistake", "section", "important"):
            self.command({"cmd": "mark", "type": cmd, "note": args}, source="voice")
            if cmd == "important":
                self.command({"cmd": "shape", "shape": "box", "x": 0.5, "y": 0.5, "w": 0.96, "h": 0.94,
                              "color": "#FFCC00", "thickness": 0.006, "for": 1.5}, source="voice")
        elif cmd == "zoom_in":
            self.command({"cmd": "zoom", "x": 0.25, "y": 0.25, "w": 0.5, "h": 0.5, "for": 8}, source="voice")
        elif cmd == "zoom_out":
            self.command({"cmd": "remove", "name": "zoom"}, source="voice")
        elif cmd == "captions_on":
            self.command({"cmd": "add", "name": "captions", "type": "captions"}, source="voice")
        elif cmd == "captions_off":
            self.command({"cmd": "remove", "name": "captions"}, source="voice")
        elif cmd == "stop":
            self.command({"cmd": "stop"}, source="voice")
        elif cmd in ("confirm", "deny", "next") and self.questions and not self.suggesting and not self.asks:
            # a question from Claude is open: "something else" / "no" / "yes" are answers to it
            self.command({"cmd": "answer", "text": d.get("text", "")}, source="voice")
        elif cmd in ("confirm", "deny"):
            if self.asks:
                self.command({"cmd": cmd}, source="voice")
            elif self.suggesting:
                self.command({"cmd": "answer", "question": self.suggesting["qid"],
                              "text": "Confirm" if cmd == "confirm" else "Cancel"}, source="voice")
        elif cmd == "next":
            if self.suggesting:
                self.command({"cmd": "answer", "question": self.suggesting["qid"], "text": "Next"}, source="voice")
        elif cmd == "suggest":
            self.command({"cmd": "suggest"}, source="voice")
        elif cmd == "full_access":
            if d.get("wake"):  # never from an armed window, a correction or background audio without "VidAI"
                self.command({"cmd": "mode", "mode": "full"}, source="voice")
            else:
                self.notify("Say “VidAI, take all actions” to give full access", "warn", 5)
        elif cmd == "ask_first":
            self.command({"cmd": "mode", "mode": "ask"}, source="voice")
        elif cmd == "record":
            if self.on_start_request and not self.output:
                self.on_start_request()
                self.bus.publish("action", {"what": "record_requested", "source": "voice"})
        elif cmd == "learn":
            name = args.split()[0] if args else f"learner{len(self.learners) + 1}"
            self.command({"cmd": "learn", "name": name}, source="voice")
        elif cmd == "label":
            parts = args.split()
            if len(parts) >= 2 and parts[0] in self.learners:
                self.command({"cmd": "label", "name": parts[0], "value": " ".join(parts[1:])}, source="voice")
            elif parts and self.learners:
                self.command({"cmd": "label", "name": list(self.learners)[-1], "value": " ".join(parts)},
                             source="voice")
        elif cmd == "wrong":
            last = self.requests_log[-1] if self.requests_log else None
            if last and last["names"] and self.clock() - last["t"] < 20:  # undo what VidAI just did
                for n in last["names"]:
                    self.command({"cmd": "remove", "name": n}, source="voice")
                self.say("Sorry. What did you mean?")
            elif self.learners:
                self.command({"cmd": "wrong", "name": list(self.learners)[-1]}, source="voice")

    def notify(self, text: str, kind: str = "info", seconds: float = 3.5) -> None:
        """A short message for the user, shown on the preview/window only (not burned into the video)."""
        self.bus.publish("notify", {"text": text, "kind": kind, "seconds": seconds})

    def help_card(self) -> dict:
        from ..profile import Profile
        from .stickers import WORDS

        prof = Profile()
        lines = ["Say “VidAI …” then:",
                 "record · stop · mark · mistake · new section <title> · important",
                 "zoom in / out · captions on / off · undo · redo · lighter",
                 "add <thing> in my hand / on my head / on my eyes  (apple, crown, sunglasses …)",
                 "make my eyes pop · text above my head saying … · remove <thing> / everything",
                 "fix the light · brighter / darker · more contrast · more colourful · black and white · warmer / cooler",
                 "blur the background · beach behind me · purple background · moving background · normal colours",
                 "suggest → ideas previewed live: confirm · next · cancel",
                 "take all actions · ask me first · confirm / deny",
                 "anything else → Claude (the eye icon shows while Claude works)"]
        macros = [m["phrase"] for m in prof.macros()][-6:]
        if macros:
            lines.append("Your shortcuts: " + " · ".join(macros))
        lib = sorted(f.stem for f in (prof.dir.parent / "effects").glob("*.py")) if (prof.dir.parent / "effects").exists() else []
        if lib:
            lines.append("Saved effects: " + " · ".join(lib[:8]))
        lines.append(f"{len(WORDS)} stickers · hold ctrl+alt+space to talk without “VidAI” · or type below")
        self.bus.publish("help", {"lines": lines, "seconds": 14})
        return {"lines": len(lines)}

    def listen(self, seconds: float = 6.0) -> None:
        """Push-to-talk: the next thing the user says is a command (no wake word needed)."""
        if self.stt:
            self.stt.armed_until = self._audio_now() + seconds
        self.notify("🎙 Listening… say the command", "listen", seconds)

    def _on_speech_for_request(self, ev: dict) -> None:
        r = self._req
        if r is None:
            return
        if ev["kind"] == "speech_start":  # the user keeps talking: wait for that transcript
            r["waiting"] = True
            r["talking"] = True
        elif ev["kind"] == "speech_end":  # they stopped: the transcript follows within ~1-2 s
            r["talking"] = False
            r["transcript_due"] = time.monotonic() + 2.5
        elif ev["kind"] == "transcript" and not ev["data"].get("is_command") and ev["seq"] > r["seq"]:
            r["parts"].append(ev["data"]["text"].strip())
            r["waiting"] = False
            r["due"] = time.monotonic() + self.request_gap
            self._flush_if_obvious()

    def _flush_if_obvious(self) -> None:
        """Act at once when the fast path understands a finished sentence (no need to wait for the pause)."""
        import re as _re

        from .intents import match

        r = self._req
        if r is None or r.get("source") == "talk":  # a question is never a shortcut: wait for all of it
            return
        msg = " ".join(" ".join(p.rstrip(".…") for p in r["parts"]).split())
        unfinished = _re.search(r"\b(saying|says|say|with|and|the|a|an|to|on|in|my|of|that|above|over)\s*$",
                                msg.lower())
        if not unfinished and match(msg, self.chain):
            self._flush_request(force=True)

    def _flush_request(self, force: bool = False) -> None:
        with self._cmd_lock:
            r = self._req
            if r is None:
                return
            now = time.monotonic()
            if not (force or now >= r["cap"] or (not r["waiting"] and now >= r["due"]) or
                    (r["waiting"] and not r.get("talking") and now >= r.get("transcript_due", r["due"] + 3.0))):
                return
            # (a long correction like "add 1, 2, 3, 4, 5" is waited for until its transcript arrives)
            self._req = None
            msg = " ".join(" ".join(p.rstrip(".…") for p in r["parts"]).replace("...", " ").split())
            self.request(msg, source=r.get("source", "voice"))

    def request(self, msg: str, source: str = "voice") -> dict:
        """A request in plain words: handled locally when VidAI understands it (fast path, < 1 s),
        otherwise sent to Claude (thinking icon until Claude answers)."""
        from .intents import match

        self._txn += 1  # everything this request does can be undone as one step
        self.redo_stack.clear()
        fixed = self.profile.correct(msg)
        if fixed != msg:
            self.bus.publish("action", {"what": "corrected", "heard": msg, "meant": fixed})
            msg = fixed
        via, cmds = "memory", None
        macro = None if source == "talk" else self.profile.find_macro(msg)
        if macro:
            cmds = [dict(c) for c in macro["commands"]]
            self.profile.used_macro(macro["phrase"])
        else:
            via = "fast"
            try:
                cmds = None if source == "talk" else match(msg, self.chain)
            except Exception as e:
                self.bus.publish("error", {"where": "fast_path", "error": repr(e)[:200]})
            cmds = [self._apply_prefs(c) for c in cmds] if cmds else None
        rec = {"msg": msg, "t": self.clock(), "via": via if cmds else "claude", "names": [], "cmds": [],
               "removed": False}
        self._finish_request_log(rec)
        if cmds:
            for c in cmds:
                self.command(c, source=via)
                if c.get("cmd") in ("add", "text", "zoom", "shape", "image", "blur"):
                    rec["names"].append(c.get("name") or c.get("cmd"))
            self.bus.publish("action", {"what": "fast_request", "message": msg, "commands": cmds, "via": via})
            import re as _re

            if _re.search(r"\bswap\b|\bswitch hands\b|\bother way\b", msg.lower()):  # learn once per swap
                self.profile.set_pref("hands_swapped", not self.profile.pref("hands_swapped", False))
                self._learned("preference", hands_swapped=self.profile.pref("hands_swapped"))
            return {"handled": via, "commands": cmds}
        self._claude_req = rec
        self.bus.publish("claude", {"message": msg, "source": source,
                                    **({"reply": "voice"} if source == "talk" else {})})
        return {"handled": "claude"}

    def say(self, text: str, record: bool = True) -> None:
        """VidAI speaks: offline voice (Piper) on the speakers, and — while recording — the same clip is mixed
        cleanly into the video at stop. While it talks (+0.6 s) the mic is ignored, so VidAI never answers itself."""
        self.bus.publish("action", {"what": "vidai_said", "text": text})
        if not self.live.speak or self.voice.engine == "none":
            return
        est = 0.8 + 0.42 * len(text.split())
        self.speaking_until = max(self.speaking_until, self._audio_now() + est)

        def run() -> None:
            try:
                clip = self.voice.synth(text)
                if clip:
                    path, dur = clip
                    t0 = self.clock()  # video time: where the clip goes in the final mix
                    self.speaking_until = max(self.speaking_until, self._audio_now() + dur + 0.8)
                    if self.output and record:
                        self.voice_clips.append((round(t0, 3), str(path), round(dur, 3)))
                        self.bus.publish("action", {"what": "vidai_voice", "t": round(t0, 3), "path": str(path),
                                                    "seconds": round(dur, 3), "text": text})
                    self.voice.play(path)
                else:
                    self.voice.speak_fallback(text)
            except Exception as e:
                self.bus.publish("error", {"where": "voice", "error": repr(e)[:200]})
            self.speaking_until = max(self.speaking_until, self._audio_now() + 0.8)  # speaker + mic latency

        threading.Thread(target=run, daemon=True).start()

    def _audio_now(self) -> float:
        """Current time on the audio clock (what utterance start/end times use); video clock if no mic."""
        return self.audio.t if self.audio.written else self.clock()

    SUGGESTIONS = [  # (title shown/spoken, phrase the fast path understands, needs)
        ("Fix the light", "fix the light", None),
        ("Blur the background", "blur the background", None),
        ("A beach behind you", "beach behind me", "image:beach"),
        ("A moving background", "moving background", None),
        ("More colourful picture", "more colourful", None),
        ("Warmer colours", "warmer", None),
        ("Show your emotion above your head", "show my emotion", "model:emotion"),
        ("React to your hand gestures", "detect my gestures", "model:gestures"),
        ("A crown on your head", "put a crown on my head", None),
        ("Sunglasses", "put sunglasses on me", None),
        ("Pop-out cartoon eyes", "make my eyes pop", None),
        ("A painting look", "make it look like a painting", "model:style_mosaic"),
        ("Black and white", "black and white", None),
        ("A little sparkle on your hand", "put sparkles in my hand", None),
    ]

    def _suggestions(self) -> list[dict]:
        """Ideas that fit now: nothing already on, instant to preview (installed models only), dark picture ->
        light first, and what the user confirmed before ranks higher (skipped often -> dropped)."""
        from .. import hub
        from .intents import _images, match

        prefs = self.profile.pref("suggestions", {}) or {}
        imgs = None
        out = []
        for title, phrase, needs in self.SUGGESTIONS:
            if needs and needs.startswith("model:") and not hub.installed(needs[6:]):
                continue
            if needs and needs.startswith("image:"):
                imgs = imgs if imgs is not None else _images()
                if needs[6:] not in imgs:
                    continue
            cmds = match(phrase, self.chain)
            if not cmds or all(c.get("cmd") == "remove" for c in cmds):
                continue
            if any(c.get("cmd") == "add" and self.chain.get(c.get("name", "")) is not None for c in cmds):
                continue  # already on

            def no_change(c: dict) -> bool:
                q = self.chain.get(c.get("name", "")) if c.get("cmd") == "set" else None
                return q is not None and all(q.params.get(k) == v for k, v in (c.get("params") or {}).items())

            if all(no_change(c) for c in cmds):
                continue  # it would change nothing (e.g. auto light is already on)
            st = prefs.get(title, {"yes": 0, "no": 0})
            if st["no"] >= 3 and st["yes"] == 0:
                continue  # the user keeps skipping it
            score = 2 * st["yes"] - st["no"]
            if title == "Fix the light" and self._is_dark():
                score += 10
            out.append({"title": title, "phrase": phrase, "cmds": cmds, "score": score})
        out.sort(key=lambda d: -d["score"])  # stable: equal scores keep the list order
        return out[:10]

    def _is_dark(self) -> bool:
        f = self.preview_frame
        return f is not None and float(f.mean()) < 90

    def _suggest_show(self, i: int) -> None:
        import uuid

        st = self.suggesting
        if st is None:
            return
        if i >= len(st["items"]):
            self._suggest_end(undo=False, note=False)
            self.notify("That's all my ideas for now 💡", "info", 4)
            return
        st["i"] = i
        item = st["items"][i]
        self._txn += 1  # the preview is one undo step
        self.redo_stack.clear()
        for c in item["cmds"]:
            self.command(dict(c), source="suggest")
        qid = "sugg_" + uuid.uuid4().hex[:6]
        st["qid"] = qid
        self.command({"cmd": "question", "id_q": qid, "speak": False, "options": ["Confirm", "Next", "Cancel"],
                      "text": f"💡 {item['title']}  ({i + 1}/{len(st['items'])})"}, source="suggest")
        self.say(f"{item['title']}?")
        if self.stt:  # "next" / "confirm" / "cancel" without the wake word
            self.stt.armed_until = max(self._audio_now(), self.speaking_until) + 15

    def _suggest_answer(self, ans: str) -> None:
        st = self.suggesting
        if st is None:
            return
        item = st["items"][st["i"]]
        prefs = self.profile.pref("suggestions", {}) or {}
        rec = prefs.setdefault(item["title"], {"yes": 0, "no": 0})
        a = (ans or "").lower()
        if a.startswith("confirm"):
            rec["yes"] += 1
            self.profile.set_pref("suggestions", prefs)
            self._suggest_end(undo=False, note=False)
            self.notify(f"✓ Kept: {item['title']} — say “VidAI suggest” for more ideas", "ok", 5)
        elif a.startswith("next"):
            rec["no"] += 1
            self.profile.set_pref("suggestions", prefs)
            self.undo(quiet=True)
            self._suggest_show(st["i"] + 1)
        else:  # cancel (or anything else)
            self._suggest_end(undo=True, note=True)

    def _suggest_end(self, undo: bool, note: bool) -> None:
        st, self.suggesting = self.suggesting, None
        if st and st.get("qid"):
            self.questions.pop(st["qid"], None)
            self.bus.publish("answer", {"question": st["qid"], "answer": None, "cancelled": True})
        if undo and st and st["i"] >= 0:
            self.undo(quiet=True)
        if note:
            self.notify("Suggestions cancelled", "info", 3)


_NUMBERS = {"one": 0, "first": 0, "1": 0, "two": 1, "second": 1, "2": 1, "three": 2, "third": 2, "3": 2,
            "four": 3, "fourth": 3, "4": 3, "five": 4, "fifth": 4, "5": 4}


def _match_option(text: str, options: list[str]) -> str:
    """Map a spoken/typed answer to one of the options ("the second", "blur", "option two"...)."""
    import difflib
    import re as _re

    t = _re.sub(r"[^\w\s]", " ", text.lower()).strip()
    if not options:
        return text.strip()
    for w in t.split():
        if w in _NUMBERS and _NUMBERS[w] < len(options):
            return options[_NUMBERS[w]]
    for o in options:
        if o.lower() in t or t in o.lower():
            return o
    best = max(options, key=lambda o: difflib.SequenceMatcher(None, t, o.lower()).ratio())
    return best if difflib.SequenceMatcher(None, t, best.lower()).ratio() > 0.45 else text.strip()
