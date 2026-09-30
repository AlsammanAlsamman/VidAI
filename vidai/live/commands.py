"""Commands: one entry point for Claude's control file, rules, voice, the GUI and macros.
Each command is a `_cmd_<name>` method; `_COMMANDS` maps command names to them."""
from __future__ import annotations

import importlib.util
import sys
import threading
import time
from pathlib import Path

from .dialog import _match_option
from .learn import LiveLearner
from .processors import REGISTRY, LiveProcessor
from .sensors import ScreenText


class CommandsMixin:
    """Part of LivePipeline (see pipeline.py); uses its state."""

    _COMMANDS = {
        "add": "_cmd_add",
        "text": "_cmd_text",
        "zoom": "_cmd_text",
        "shape": "_cmd_text",
        "image": "_cmd_text",
        "blur": "_cmd_text",
        "enable": "_cmd_enable",
        "disable": "_cmd_disable",
        "remove": "_cmd_remove",
        "set": "_cmd_set",
        "rule": "_cmd_rule",
        "unrule": "_cmd_unrule",
        "mark": "_cmd_mark",
        "learn": "_cmd_learn",
        "label": "_cmd_label",
        "wrong": "_cmd_wrong",
        "forget": "_cmd_forget",
        "stt": "_cmd_stt",
        "ocr": "_cmd_ocr",
        "stop": "_cmd_stop",
        "done": "_cmd_done",
        "ask": "_cmd_ask",
        "confirm": "_cmd_confirm",
        "deny": "_cmd_confirm",
        "mode": "_cmd_mode",
        "model": "_cmd_model",
        "suggest": "_cmd_suggest",
        "talk": "_cmd_talk",
        "say": "_cmd_say",
        "record": "_cmd_record",
        "undo": "_cmd_undo",
        "redo": "_cmd_redo",
        "help": "_cmd_help",
        "lighter": "_cmd_lighter",
        "listen": "_cmd_listen",
        "notify": "_cmd_notify",
        "question": "_cmd_question",
        "cancel_question": "_cmd_cancel_question",
        "answer": "_cmd_answer",
        "remove_last_added": "_cmd_remove_last_added",
        "thinking": "_cmd_thinking",
        "status": "_cmd_status",
    }

    def _rule_action(self, action: dict, ev: dict, rule: dict) -> None:
        a = dict(action)
        if "show_text" in a:
            self.command({"cmd": "text", "text": a.pop("show_text"), **a}, source=f"rule:{rule['id']}")
        elif "zoom" in a:
            self.command({"cmd": "zoom", **a.pop("zoom"), **a}, source=f"rule:{rule['id']}")
        elif "shape" in a:
            self.command({"cmd": "shape", **a.pop("shape"), **a}, source=f"rule:{rule['id']}")
        elif "enable" in a:
            self.command({"cmd": "enable", "name": a["enable"], "for": a.get("for")}, source=f"rule:{rule['id']}")
        elif "disable" in a:
            self.command({"cmd": "disable", "name": a["disable"]}, source=f"rule:{rule['id']}")
        elif "set" in a:
            self.command({"cmd": "set", "name": a["set"], "params": a.get("params", {})}, source=f"rule:{rule['id']}")
        elif "mark" in a:
            self.command({"cmd": "mark", "type": a["mark"], "note": a.get("note", "")}, source=f"rule:{rule['id']}")
        elif "sticker" in a:  # e.g. a gesture rule: {"sticker": "👍", "to": "hand", "for": 2}
            self.command({"cmd": "add", "name": self._temp_name("sticker"), "type": "attach",
                          "params": {"what": a["sticker"], "to": a.get("to", "screen"),
                                     "position": a.get("position", "top-right")}, "for": a.get("for", 2)},
                         source=f"rule:{rule['id']}")
        elif "notify_claude" in a:
            self.bus.publish("claude", {"message": a["notify_claude"], "source": f"rule:{rule['id']}"})
        elif "label" in a:
            self.command({"cmd": "label", "name": a["label"], "value": a.get("value", "yes")},
                         source=f"rule:{rule['id']}")
        else:
            self.bus.publish("error", {"where": f"rule:{rule['id']}", "error": f"unknown action {action}"})

    def _temp_name(self, base: str) -> str:
        self._n_tmp += 1
        return f"{base}_{self._n_tmp}"

    def _check_code_file(self, path: str) -> None:
        """Effect code runs only from the session's processors folder (after the user allowed Claude's code in
        this session) or from the saved effects library."""
        from .. import actions
        from ..lab import vidai_home

        f = Path(path).expanduser().resolve()
        lib = (vidai_home() / "effects").resolve()
        if lib in f.parents:
            return
        if self.dir and Path(self.dir).resolve() in f.parents:
            if not actions.code_allowed(self.dir):
                raise PermissionError("the user has not allowed effect code written by Claude in this session "
                                      "(use live_processor: VidAI asks them)")
            return
        raise PermissionError(f"effect code must be in the session folder or the effects library, not {path}")

    def _load_file(self, path: str) -> list[str]:
        self._check_code_file(path)
        before = set(REGISTRY.items())
        mod_name = f"vidai_live_user_{Path(path).stem}_{int(time.time() * 1000)}"
        spec = importlib.util.spec_from_file_location(mod_name, path)
        module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        new = [k for k, v in REGISTRY.items() if (k, v) not in before]
        if not new:  # also accept plain subclasses without @register
            for obj in vars(module).values():
                if isinstance(obj, type) and issubclass(obj, LiveProcessor) and obj is not LiveProcessor \
                        and obj.__module__ == mod_name:
                    REGISTRY[obj.__name__.lower()] = obj
                    new.append(obj.__name__.lower())
        return new

    USER_SOURCES = {"voice", "gui", "user"}

    def command(self, c: dict, source: str = "claude") -> dict:
        """Apply one command (from Claude's control file, a rule, a voice command or the GUI)."""
        with self._cmd_lock:
            return self._command(c, source)

    def _command(self, c: dict, source: str) -> dict:
        c = dict(c)
        cmd = c.pop("cmd", None)
        cid = c.pop("id", None)  # echoed back so the sender can match replies exactly
        no_learn = c.pop("no_learn", False)
        tag = {"id": cid} if cid else {}
        t = self.clock()
        try:
            self._track_learning(cmd, {**c, "no_learn": no_learn}, source)
        except Exception as e:
            self.bus.publish("error", {"where": "learning", "error": repr(e)[:200]})
        try:
            if source not in self.USER_SOURCES and (cmd in ("confirm", "deny") or
                                                    (cmd == "mode" and c.get("mode") == "full")):
                raise PermissionError(f"'{cmd}' can only come from the user (voice or the window), not {source}")
            inverse = self._inverse(cmd, c) if source not in ("undo", "carry", "config") else None
            res = self._do(cmd, c, t)
            if inverse == [{"cmd": "remove_last_added"}] and (res or {}).get("name"):
                inverse = [{"cmd": "remove", "name": res["name"]}]  # the exact effect, whatever came after it
            if inverse is not None:
                self._record(inverse, {"cmd": cmd, **c}, source)
            self.bus.publish("ack", {"command": cmd, "source": source, **tag, **(res or {})})
            return res or {}
        except Exception as e:
            self.bus.publish("error", {"where": "command", "command": cmd, "source": source, **tag,
                                       "error": repr(e)[:300]})
            return {"error": repr(e)}

    def _install_and_apply(self, mid: str, params: dict) -> None:
        import uuid

        from .. import actions, hub

        m = hub.CATALOG[mid]
        if not hub.installed(mid):
            if actions.get_mode(self.dir) != "full":
                rid = uuid.uuid4().hex[:8]
                got = threading.Event()
                answer: dict = {}

                def on_perm(ev: dict) -> None:
                    if ev["data"].get("request") == rid:
                        answer.update(ev["data"])
                        got.set()

                self.bus.subscribe(on_perm, {"permission"})
                self.command({"cmd": "ask", "id_ask": rid, "text": actions.describe("model", {"model": mid})},
                             source="hub")
                if not got.wait(90) or answer.get("state") != "approved":
                    self.notify(f"Not installing {m['title']}", "warn")
                    return
            self.notify(f"Downloading {m['title']} ({m['mb']} MB)…", "info", 8)
            try:
                actions.download_model(mid)
            except Exception as e:
                self.bus.publish("error", {"where": "hub", "model": mid, "error": repr(e)[:200]})
                self.notify(f"Could not download {m['title']}", "warn")
                return
            if self.dir:
                with open(Path(self.dir) / "CREDITS.txt", "a", encoding="utf-8") as f:
                    f.write(f"Model: {m['title']} — {m['url']} ({m['license']})\n")
        adapter = m["adapter"]
        name = "fx_style" if adapter == "style" else f"fx_{mid}"
        prm = {**({"model": mid} if adapter == "style" else {}), **params}
        self.command({"cmd": "add", "name": name, "type": adapter, "params": prm}, source="hub")
        slow = "" if m["live"] else " (paints a few times per second live; full quality when editing)"
        self.notify(f"✓ {m['title']}{slow}", "ok", 5)
        self.bus.publish("action", {"what": "model_applied", "model": mid, "name": name})

    def carry_specs(self) -> list[dict]:
        """Lasting effects that are on now (added in preview) -> re-added when the recording starts."""
        out = []
        for p in self.chain.items:
            spec = getattr(p, "spec", None)
            if spec and p.enabled and p.until is None and not p.name.startswith("_"):
                out.append({**spec, "params": dict(p.params)})
        return out

    def status(self) -> dict:
        return {"t": round(self.clock(), 2), "frames": self.frames, "loop_ms": round(self.loop_ms, 2),
                "processors": [p.describe() for p in self.chain.items], "rules": list(self.rules.rules.values()),
                "learners": [l.describe() for l in self.learners.values()],
                "stt": bool(self.stt and self.stt.enabled), "ocr": bool(self.ocr and self.ocr.enabled),
                "pending_requests": [{"request": k, **v} for k, v in self.pending.items()]}

    def _do(self, cmd: str | None, c: dict, t: float) -> dict | None:
        dur = c.pop("for", None)
        until = t + float(dur) if dur else None
        fn = self._COMMANDS.get(cmd or "")
        if fn is None:
            raise ValueError(f"unknown command {cmd!r}")
        return getattr(self, fn)(cmd, c, t, until)

    def _cmd_add(self, cmd: str, c: dict, t: float, until: float | None) -> dict | None:
        if c.get("file"):
            types = self._load_file(c["file"])
            ptype = c.get("type") or types[-1]
        else:
            ptype = c.get("type", c.get("name"))
        cls = REGISTRY.get(str(ptype).lower())
        if cls is None:
            raise KeyError(f"unknown processor type {ptype!r}; known: {sorted(set(REGISTRY))}")
        p = self.chain.add(cls(c.get("name") or ptype, c.get("params", {}), c.get("enabled", True), until))
        p.spec = {"cmd": "add", "name": p.name, "params": p.params,  # to carry it from preview into recording
                  **({"file": c["file"]} if c.get("file") else {"type": ptype})}
        self.bus.publish("action", {"what": "processor_added", "name": p.name, "type": cls.__name__,
                                    "params": p.params, "until": until})
        return {"name": p.name}

    def _cmd_text(self, cmd: str, c: dict, t: float, until: float | None) -> dict | None:
        name = c.pop("name", None) or (cmd if cmd == "zoom" else self._temp_name(cmd))
        p = self.chain.add(REGISTRY[cmd](name, c, True, until))
        self.bus.publish("action", {"what": cmd, "name": name, "params": p.params, "until": until})
        return {"name": name}

    def _cmd_enable(self, cmd: str, c: dict, t: float, until: float | None) -> dict | None:
        p = self.chain.get(c["name"])
        if not p:
            raise KeyError(f"no processor {c['name']!r}")
        p.enabled, p.until, p.slow_frames = True, until, 0
        return {"name": p.name}

    def _cmd_disable(self, cmd: str, c: dict, t: float, until: float | None) -> dict | None:
        p = self.chain.get(c["name"])
        if p:
            p.enabled = False
        return {"name": c["name"]}

    def _cmd_remove(self, cmd: str, c: dict, t: float, until: float | None) -> dict | None:
        return {"removed": self.chain.remove(c["name"])}

    def _cmd_set(self, cmd: str, c: dict, t: float, until: float | None) -> dict | None:
        p = self.chain.get(c["name"])
        if not p:
            raise KeyError(f"no processor {c['name']!r}")
        p.configure(c.get("params", {}))
        self.chain.ensure_feeds(p)  # e.g. behind_hair switched on
        self.chain.prune_feeds()
        return {"name": p.name, "params": p.params}

    def _cmd_rule(self, cmd: str, c: dict, t: float, until: float | None) -> dict | None:
        return {"rule": self.rules.add(c["rule"])}

    def _cmd_unrule(self, cmd: str, c: dict, t: float, until: float | None) -> dict | None:
        return {"removed": self.rules.remove(c["id"])}

    def _cmd_mark(self, cmd: str, c: dict, t: float, until: float | None) -> dict | None:
        self.bus.publish("marker", {"type": c.get("type", "marker"), "note": c.get("note", ""),
                                    "source": c.get("source", "")})
        return {"type": c.get("type", "marker")}

    def _cmd_learn(self, cmd: str, c: dict, t: float, until: float | None) -> dict | None:
        name = c.get("name") or f"learner{len(self.learners) + 1}"
        self.learners[name] = LiveLearner(self.bus, name, c.get("labels"), c.get("region"), c.get("k", 5))
        return {"learner": name, "labels": self.learners[name].model.labels}

    def _cmd_label(self, cmd: str, c: dict, t: float, until: float | None) -> dict | None:
        return self.learners[c["name"]].label(c.get("value", "yes"))

    def _cmd_wrong(self, cmd: str, c: dict, t: float, until: float | None) -> dict | None:
        return self.learners[c["name"]].wrong()

    def _cmd_forget(self, cmd: str, c: dict, t: float, until: float | None) -> dict | None:
        return {"removed": self.learners.pop(c["name"], None) is not None}

    def _cmd_stt(self, cmd: str, c: dict, t: float, until: float | None) -> dict | None:
        if self.stt:
            self.stt.enabled = bool(c.get("on", True))
            if c.get("language"):
                self.stt.language = c["language"]
        return {"stt": bool(self.stt and self.stt.enabled)}

    def _cmd_ocr(self, cmd: str, c: dict, t: float, until: float | None) -> dict | None:
        if c.get("on", True) and not self.ocr:
            self.ocr = ScreenText(self.bus, c.get("interval", 2.0), c.get("langs", self.live.ocr_langs))
        elif self.ocr:
            self.ocr.enabled = bool(c.get("on", True))
        return {"ocr": bool(self.ocr and self.ocr.enabled)}

    def _cmd_stop(self, cmd: str, c: dict, t: float, until: float | None) -> dict | None:
        if self.on_stop_request:
            self.on_stop_request()
        return {"stopping": True}

    def _cmd_done(self, cmd: str, c: dict, t: float, until: float | None) -> dict | None:
        """Claude finished a request (or all of them)"""
        req = c.get("request")
        if req is None:
            self.pending.clear()
        else:
            self.pending.pop(int(req), None)
        if not self.pending:
            self._show_thinking(False)
        return {"pending": len(self.pending)}

    def _cmd_ask(self, cmd: str, c: dict, t: float, until: float | None) -> dict | None:
        """VidAI needs permission for an action (install, download, create...)"""
        from .. import actions

        rid, text = str(c["id_ask"]), c["text"]
        if actions.get_mode(self.dir) == "full":
            self.bus.publish("permission", {"request": rid, "state": "approved", "by": "full_access"})
            return {"request": rid, "state": "approved"}
        self.asks[rid] = text
        self.say(f"{self.live.address}, I need to {text}. Say VidAI confirm, or VidAI deny.")
        self.bus.publish("action", {"what": "asking", "request": rid, "text": text})
        if self.stt:  # a bare "yes" / "confirm" right after the question is enough
            self.stt.arm(self._audio_now() + 25, "ask")
        return {"request": rid, "state": "pending"}

    def _cmd_confirm(self, cmd: str, c: dict, t: float, until: float | None) -> dict | None:
        rid = str(c.get("request") or (list(self.asks)[-1] if self.asks else ""))
        if rid not in self.asks:
            return {"state": "nothing to answer"}
        text = self.asks.pop(rid)
        state = "approved" if cmd == "confirm" else "denied"
        self.bus.publish("permission", {"request": rid, "state": state, "by": c.get("by", "user"), "text": text})
        self.say("Done, working on it." if state == "approved" else "Okay, I won't.")
        return {"request": rid, "state": state}

    def _cmd_mode(self, cmd: str, c: dict, t: float, until: float | None) -> dict | None:
        """'VidAI, take all actions' / 'VidAI, ask me first'."""
        from .. import actions

        mode = "full" if c.get("mode") == "full" else "ask"
        actions.set_mode(self.dir, mode)
        self.bus.publish("permission_mode", {"mode": mode})
        self.say("Full access. I will take all actions needed." if mode == "full"
                 else "Okay, I will ask you first.")
        if mode == "full":  # anything already waiting is approved too
            for rid in list(self.asks):
                self.asks.pop(rid)
                self.bus.publish("permission", {"request": rid, "state": "approved", "by": "full_access"})
        return {"mode": mode}

    def _cmd_model(self, cmd: str, c: dict, t: float, until: float | None) -> dict | None:
        """Use a hub model: install it (with permission) in the background, then apply it"""
        from .. import hub

        mid = str(c.get("model") or c.get("model_id"))
        if mid not in hub.CATALOG:
            raise KeyError(f"unknown hub model {mid!r}; see vidai.hub.CATALOG / model_search")
        threading.Thread(target=self._install_and_apply, args=(mid, dict(c.get("params") or {})),
                         daemon=True).start()
        return {"model": mid, "state": "applying" if hub.installed(mid) else "installing"}

    def _cmd_suggest(self, cmd: str, c: dict, t: float, until: float | None) -> dict | None:
        """VidAI proposes ideas one by one, previewed live: Confirm / Next / Cancel"""
        if self.suggesting:
            self._suggest_end(undo=True, note=False)
        items = self._suggestions()
        if not items:
            self.notify("No new ideas right now", "info")
            return {"suggestions": 0}
        self.suggesting = {"items": items, "i": -1, "qid": ""}
        self._suggest_show(0)
        return {"suggestions": len(items)}

    def _cmd_talk(self, cmd: str, c: dict, t: float, until: float | None) -> dict | None:
        """'VidAI talk': VidAI asks, the next sentence is a question for Claude."""
        self._talk = True
        self.say("How can I help you?")
        if self.stt:  # no wake word needed for the question (armed after VidAI stops talking)
            self.stt.arm(max(self._audio_now(), self.speaking_until) + 10, "talk")
        self.notify("🎙 Ask your question…", "listen", 8)
        return {"talk": True}

    def _cmd_say(self, cmd: str, c: dict, t: float, until: float | None) -> dict | None:
        """Claude answers out loud (and optionally as a subtitle in the video)"""
        text = str(c.get("text", "")).strip()
        if text:
            self.say(text, record=c.get("record", True))
            if c.get("subtitle"):
                dur = 1.5 + 0.42 * len(text.split())
                self._do("text", {"name": self._temp_name("answer"), "text": text, "position": "bottom-center",
                                  "size": 0.04, "for": dur}, t)
            elif c.get("notify", True):
                self.notify("VidAI: " + text, "claude", 3 + 0.3 * len(text.split()))
        return {"said": bool(text), "engine": self.voice.engine}

    def _cmd_record(self, cmd: str, c: dict, t: float, until: float | None) -> dict | None:
        """Claude (or a typed/voice request) starts the recording from preview"""
        if self.on_start_request and not self.output:
            self.on_start_request()
            return {"recording": "starting"}
        return {"recording": bool(self.output)}

    def _cmd_undo(self, cmd: str, c: dict, t: float, until: float | None) -> dict | None:
        return self.undo()

    def _cmd_redo(self, cmd: str, c: dict, t: float, until: float | None) -> dict | None:
        return self.redo()

    def _cmd_help(self, cmd: str, c: dict, t: float, until: float | None) -> dict | None:
        return self.help_card()

    def _cmd_lighter(self, cmd: str, c: dict, t: float, until: float | None) -> dict | None:
        self._set_level(min(2, self.level + 1))
        return {"level": self.level}

    def _cmd_listen(self, cmd: str, c: dict, t: float, until: float | None) -> dict | None:
        self.listen(float(c.get("seconds", 6)))
        return {"listening": True}

    def _cmd_notify(self, cmd: str, c: dict, t: float, until: float | None) -> dict | None:
        """Claude -> user message (window only)"""
        self.notify(c.get("text", ""), c.get("kind", "claude"), float(c.get("seconds", 5)))
        return {"notified": True}

    def _cmd_question(self, cmd: str, c: dict, t: float, until: float | None) -> dict | None:
        """Claude asks the user; answer by button, typing or voice"""
        qid = str(c["id_q"])
        self.questions[qid] = {"text": c["text"], "options": list(c.get("options") or []), "t": time.monotonic()}
        self.bus.publish("question", {"question": qid, "text": c["text"], "options": self.questions[qid]["options"]})
        if c.get("speak", True):
            opts = self.questions[qid]["options"]
            self.say(c["text"] + (" Options: " + ", ".join(opts) + "." if opts else ""))
        if self.stt:
            self.stt.arm(self._audio_now() + 30, "question")
        return {"question": qid}

    def _cmd_cancel_question(self, cmd: str, c: dict, t: float, until: float | None) -> dict | None:
        qid = str(c.get("question") or "")
        if self.suggesting and qid == self.suggesting["qid"]:  # nobody answered: take the preview back
            self._suggest_end(undo=True, note=False)
        q = self.questions.pop(qid, None) if qid else None
        if q is None and not qid and self.questions:
            self.questions.clear()
            q = True
        if q is not None:
            self.bus.publish("answer", {"question": qid, "answer": None, "cancelled": True})
        return {"cancelled": q is not None}

    def _cmd_answer(self, cmd: str, c: dict, t: float, until: float | None) -> dict | None:
        if not self.questions:
            return {"state": "no question"}
        qid = str(c.get("question") or list(self.questions)[-1])
        q = self.questions.pop(qid, None)
        if q is None:
            return {"state": "unknown question"}
        ans = _match_option(c.get("text", ""), q["options"])
        self.bus.publish("answer", {"question": qid, "answer": ans, "said": c.get("text", ""), "by": c.get("by", "user")})
        if self.suggesting and qid == self.suggesting["qid"]:
            self._suggest_answer(ans)
        else:
            self.notify(f"✓ {ans}", "ok")
        return {"question": qid, "answer": ans}

    def _cmd_remove_last_added(self, cmd: str, c: dict, t: float, until: float | None) -> dict | None:
        fx = [p for p in self.chain.items if not p.name.startswith("_")]
        return {"removed": self.chain.remove(fx[-1].name) if fx else False}

    def _cmd_thinking(self, cmd: str, c: dict, t: float, until: float | None) -> dict | None:
        """Claude shows the icon itself while working on something longer"""
        self._show_thinking(bool(c.get("on", True)))
        return {"thinking": bool(c.get("on", True))}

    def _cmd_status(self, cmd: str, c: dict, t: float, until: float | None) -> dict | None:
        return self.status()
