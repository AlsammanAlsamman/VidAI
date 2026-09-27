"""Rules: "when <event> (and <conditions>) then <actions>" — Claude's instant reflexes during recording.

A rule (JSON, sent by Claude with live_control {"cmd": "rule", ...}):

    {"id": "pause_title",
     "when": {"kind": "silence_end", "where": {"duration": {">": 2.5}}},
     "do": [{"show_text": "{section}", "for": 3, "position": "top-left"}],
     "cooldown": 20}

where-operators: ">", ">=", "<", "<=", "==", "!=", "contains", "in", "startswith", "matches" (regex)
Actions:
    {"show_text": "...", "for": s, ...text params}      temporary text overlay
    {"zoom": {"x","y","w","h"}, "for": s}                temporary zoom
    {"shape": {...shape params}, "for": s}               temporary arrow/circle/box
    {"enable": "<processor>", "for": s?}  {"disable": "<processor>"}
    {"set": "<processor>", "params": {...}}
    {"mark": "section|important|mistake|marker", "note": "..."}
    {"notify_claude": "..."}                             ask Claude to look (slow loop)
    {"label": "<learner>", "value": "<label>"}           teach an instant model
Templates in strings: {text}, {command}, {args}, {label}, {section}, {t} and any event data field.
"""
from __future__ import annotations

import re
from typing import Any, Callable

OPS: dict[str, Callable[[Any, Any], bool]] = {
    ">": lambda a, b: a is not None and a > b,
    ">=": lambda a, b: a is not None and a >= b,
    "<": lambda a, b: a is not None and a < b,
    "<=": lambda a, b: a is not None and a <= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "contains": lambda a, b: a is not None and str(b).lower() in str(a).lower(),
    "in": lambda a, b: a in b,
    "startswith": lambda a, b: a is not None and str(a).lower().startswith(str(b).lower()),
    "matches": lambda a, b: a is not None and re.search(b, str(a), re.I) is not None,
}


class _SafeDict(dict):
    def __missing__(self, k: str) -> str:
        return ""


def fill(template: Any, values: dict) -> Any:
    if isinstance(template, str):
        try:
            return template.format_map(_SafeDict(values))
        except (ValueError, IndexError):
            return template
    if isinstance(template, dict):
        return {k: fill(v, values) for k, v in template.items()}
    if isinstance(template, list):
        return [fill(v, values) for v in template]
    return template


def matches(rule: dict, ev: dict) -> bool:
    when = rule.get("when", {})
    kinds = when.get("kind")
    if kinds and ev["kind"] not in ([kinds] if isinstance(kinds, str) else kinds):
        return False
    for field, cond in (when.get("where") or {}).items():
        val = ev["data"].get(field)
        conds = cond if isinstance(cond, dict) else {"==": cond}
        for op, ref in conds.items():
            if op not in OPS or not OPS[op](val, ref):
                return False
    return True


class RuleEngine:
    def __init__(self, bus, do_action: Callable[[dict, dict, dict], None]) -> None:
        self.bus = bus
        self.rules: dict[str, dict] = {}
        self.last_fired: dict[str, float] = {}
        self.do_action = do_action
        self.state: dict[str, Any] = {"section": ""}
        bus.subscribe(self._on_event)

    def add(self, rule: dict) -> str:
        rid = rule.get("id") or f"rule{len(self.rules) + 1}"
        rule["id"] = rid
        for a in rule.get("do", []):
            if not isinstance(a, dict):
                raise ValueError(f"bad action {a!r}")
        self.rules[rid] = rule
        return rid

    def remove(self, rid: str) -> bool:
        return self.rules.pop(rid, None) is not None

    def _on_event(self, ev: dict) -> None:
        if ev["kind"] == "marker" and ev["data"].get("type") == "section":
            self.state["section"] = ev["data"].get("note", "")
        if ev["kind"] in ("action", "ack", "error", "level", "motion"):
            return  # never trigger on our own outputs or chatty stats
        for rid, rule in list(self.rules.items()):
            if not rule.get("enabled", True) or not matches(rule, ev):
                continue
            cd = rule.get("cooldown", 0)
            if cd and ev["t"] - self.last_fired.get(rid, -1e9) < cd:
                continue
            self.last_fired[rid] = ev["t"]
            values = {**self.state, **ev["data"], "t": ev["t"], "kind": ev["kind"]}
            for action in rule.get("do", []):
                self.do_action(fill(action, values), ev, rule)
            if rule.get("once"):
                rule["enabled"] = False
