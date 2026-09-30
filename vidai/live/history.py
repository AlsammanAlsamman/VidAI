"""Undo / redo: every request is one reversible group of commands."""
from __future__ import annotations


class UndoMixin:
    """Part of LivePipeline (see pipeline.py); uses its state."""

    def _inverse(self, cmd: str | None, c: dict) -> list[dict] | None:
        """Commands that undo `cmd` (captured before it runs)."""
        if cmd in ("add", "text", "zoom", "shape", "image", "blur"):
            name = c.get("name") or (cmd if cmd == "zoom" else None)
            before = self.chain.get(name) if name else None
            undo = [{"cmd": "remove", "name": name}] if name else [{"cmd": "remove_last_added"}]
            if before is not None and getattr(before, "spec", None):
                undo.append({**before.spec, "params": dict(before.params)})
            return undo
        if cmd == "remove":
            p = self.chain.get(c.get("name"))
            if p is not None and getattr(p, "spec", None):
                return [{**p.spec, "params": dict(p.params)}]
            return []
        if cmd == "set":
            p = self.chain.get(c.get("name"))
            if p is not None:
                return [{"cmd": "set", "name": p.name, "params": {k: p.params.get(k) for k in c.get("params", {})}}]
        if cmd in ("enable", "disable"):
            p = self.chain.get(c.get("name"))
            if p is not None:
                return [{"cmd": "enable" if p.enabled else "disable", "name": p.name}]
        return None

    def _record(self, inverse: list[dict], forward: dict, source: str) -> None:
        if self.history and self.history[-1]["txn"] == self._txn:
            self.history[-1]["undo"] = inverse + self.history[-1]["undo"]
            self.history[-1]["redo"].append(forward)
        else:
            self.history.append({"txn": self._txn, "undo": list(inverse), "redo": [forward]})
        del self.history[:-50]

    def undo(self, quiet: bool = False) -> dict:
        if not self.history:
            if not quiet:
                self.notify("Nothing to undo")
            return {"undone": 0}
        step = self.history.pop()
        for c in step["undo"]:
            self.command(dict(c), source="undo")
        self.redo_stack.append(step)
        if not quiet:
            self.notify("↶ Undone")
        return {"undone": len(step["redo"])}

    def redo(self) -> dict:
        if not self.redo_stack:
            self.notify("Nothing to redo")
            return {"redone": 0}
        step = self.redo_stack.pop()
        for c in step["redo"]:
            self.command(dict(c), source="undo")
        self.history.append(step)
        self.notify("↷ Redone")
        return {"redone": len(step["redo"])}
