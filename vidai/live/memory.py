"""Learning from the user: preferences, misunderstood requests, Claude's answers that become
instant macros, and the end-of-recording summary (stored in vidai.profile)."""
from __future__ import annotations


class MemoryMixin:
    """Part of LivePipeline (see pipeline.py); uses its state."""

    def _apply_prefs(self, c: dict) -> dict:
        """Use what the user chose before (sizes, which hand) for a new effect."""
        c = dict(c)
        prm = dict(c.get("params") or {})
        if c.get("type") == "attach":
            sc = self.profile.pref(f"scale:{prm.get('what')}")
            if sc and "scale" not in (c.get("params") or {}):
                prm["scale"] = sc
            if self.profile.pref("hands_swapped") and prm.get("to") in ("right_hand", "left_hand"):
                prm["to"] = {"right_hand": "left_hand", "left_hand": "right_hand"}[prm["to"]]
            c["params"] = prm
        return c

    def _finish_request_log(self, new: dict) -> None:
        """A new request right after one whose result was removed = that one was misunderstood."""
        prev = self.requests_log[-1] if self.requests_log else None
        self.requests_log.append(new)
        if prev and prev["removed"] and new["t"] - prev["t"] < 25:
            import difflib

            r = difflib.SequenceMatcher(None, prev["msg"].lower(), new["msg"].lower()).ratio()
            if 0.5 <= r < 1.0 and self.profile.learn_correction(prev["msg"], new["msg"]):
                self._learned("vocabulary", heard=prev["msg"], meant=new["msg"])

    def _learned(self, kind: str, **info) -> None:
        self.learned.append({"kind": kind, **info})
        self.bus.publish("action", {"what": "learned", "kind": kind, **info})

    def _track_learning(self, cmd: str | None, c: dict, source: str) -> None:
        name = c.get("name")
        if c.get("no_learn"):  # e.g. Claude restoring an older effect: not part of this request's answer
            return
        # an effect removed soon after its request -> that request went wrong
        if cmd == "remove" and name:
            for rec in reversed(self.requests_log[-5:]):
                if name in rec["names"] and self.clock() - rec["t"] < 20 and not rec["removed"]:
                    rec["removed"] = True
                    self._probation = [p for p in self._probation if p["msg"] != rec["msg"]]
                    self.profile.add_lesson(f"The request '{rec['msg']}' was answered with {rec['names']} "
                                            f"({rec['via']}) and the user removed it right away.",
                                            ["mistake", rec["via"]], source="auto")
                    if rec["via"] == "memory":
                        self.profile.forget_macro(rec["msg"])
                    self._learned("mistake", request=rec["msg"])
        # size / hand adjustments become defaults
        if cmd == "set" and name:
            p = self.chain.get(name)
            prm = c.get("params") or {}
            if p is not None and "scale" in prm and p.params.get("what"):
                self.profile.nudge_pref(f"scale:{p.params['what']}", float(prm["scale"]))
        # what Claude does for a request -> candidate macro
        if source == "claude" and self._claude_req is not None and cmd in (
                "add", "text", "zoom", "shape", "image", "blur", "set", "remove", "enable", "disable", "rule"):
            self._claude_req["cmds"].append({"cmd": cmd, **{k: v for k, v in c.items() if k != "for"}})
            if cmd in ("add", "text", "zoom", "shape", "image", "blur"):
                self._claude_req["names"].append(name or cmd)
        if cmd == "question" and self._claude_req is not None:
            self._claude_req["asked"] = True  # needed clarification: the words alone don't define the answer
        if cmd == "done" and self._claude_req is not None:
            if self._claude_req["cmds"]:
                self._probation.append({**self._claude_req, "t_done": self.clock()})
            self._claude_req = None

    def _check_probation(self) -> None:
        """Claude's answer kept for 20 s (not removed) -> remember it: next time the request is instant."""
        now = self.clock()
        for p in list(self._probation):
            if now - p["t_done"] >= 20:
                self._probation.remove(p)
                alive = [self.chain.get(n) for n in p["names"]]
                if p.get("asked") or any(c.get("cmd") not in ("add", "text", "zoom", "shape", "image", "blur",
                                                              "rule", "mark") for c in p["cmds"]):
                    continue  # set/remove/enable... depend on what is on screen now: replayed later they'd be wrong
                if (not p["names"]) or any(q is not None and q.enabled for q in alive):
                    from ..profile import plausible_shortcut

                    if not plausible_shortcut(p["msg"]):
                        continue  # garbled or rambling speech: never an instant shortcut
                    self.profile.add_macro(p["msg"], p["cmds"])
                    self._learned("macro", request=p["msg"])

    def summary(self) -> dict:
        """What happened in this recording, for the user (window) and Claude."""
        import collections

        reqs = self.requests_log
        via = collections.Counter(r["via"] for r in reqs)
        effects = sorted({n for r in reqs for n in r["names"]})
        tips = []
        if via.get("claude"):
            tips.append("Requests Claude solved and you kept become instant next time.")
        if any(r["removed"] for r in reqs):
            tips.append("Effects you removed right away were noted as mistakes; say what you meant next time.")
        if self.level:
            tips.append("The computer was busy: fewer or lighter effects keep the video smooth.")
        return {"requests": len(reqs), "instant": via.get("fast", 0) + via.get("memory", 0),
                "by_claude": via.get("claude", 0), "effects": effects, "learned": list(self.learned),
                "undo_used": sum(1 for e in self.bus.history if e["kind"] == "ack"
                                 and e["data"].get("command") == "undo"), "tips": tips}

    def _remember_session(self) -> None:
        import collections
        import re as _re

        for p in list(self._probation):  # keep what was not removed by the end
            p["t_done"] = -1e9
        self._check_probation()
        self.profile.clean_macros()  # anything learned from garbled speech goes
        stop = {"that", "this", "with", "have", "from", "will", "what", "your", "they", "about", "there", "then",
                "vidai", "here", "just", "like", "going", "want", "make", "more", "some", "into", "them"}
        words = collections.Counter(w for tr in (self.stt.transcripts if self.stt else [])
                                    for w in _re.findall(r"[A-Za-z][A-Za-z0-9\-]{3,}", tr["text"])
                                    if w.lower() not in stop)
        self.profile.add_words([w for w, n in words.items() if n >= 2 or w.isupper()])
        reqs = self.requests_log
        self.profile.add_history({
            "session": str(self.dir or ""), "duration": round(self.clock(), 1),
            "requests": [r["msg"] for r in reqs],
            "answered_by": dict(collections.Counter(r["via"] for r in reqs)),
            "removed_quickly": [r["msg"] for r in reqs if r["removed"]],
            "errors": sum(1 for e in self.bus.history if e["kind"] == "error"),
            "learned": self.learned,
        })
