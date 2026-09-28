"""VidAI's own privileged actions: install packages, download files, create files in its workspace.

VidAI runs them itself (so Claude Code never needs a shell or a permission prompt), but only with the user's
consent for the session:
  - ask mode (default): VidAI says "Master, I need to ... Say 'VidAI confirm' or 'VidAI deny'" and waits
  - full mode: the user said "VidAI, take all actions" -> everything is allowed for this session
Limits that always apply: installs go only into VidAI's own Python environment (pip, never system), files
only under ~/.vidai/ (workspace) or the session folder, downloads up to MAX_DOWNLOAD bytes. Every action is
appended to ~/.vidai/actions.log.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

MAX_DOWNLOAD = 2 * 1024 ** 3  # 2 GB
_PKG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-\[\],]*([<>=!~]=?[A-Za-z0-9.*+!\-]+)?$")


def home() -> Path:
    d = Path(os.environ.get("VIDAI_HOME", Path.home() / ".vidai"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def workspace() -> Path:
    d = home() / "work"
    d.mkdir(parents=True, exist_ok=True)
    return d


def log(action: str, **info) -> None:
    line = json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "action": action, **info}, ensure_ascii=False)
    with open(home() / "actions.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------- permission state (shared by the recorder process and Claude's tools) ----------------
def _perm_file(session: str | Path | None) -> Path:
    return (Path(session) if session else home()) / "permissions.json"


def get_mode(session: str | Path | None) -> str:
    try:
        return json.loads(_perm_file(session).read_text())["mode"]
    except (OSError, ValueError, KeyError):
        return "ask"


def set_mode(session: str | Path | None, mode: str) -> None:
    p = _perm_file(session)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({"mode": mode, "since": time.strftime("%Y-%m-%d %H:%M:%S")}))
    os.replace(tmp, p)
    log("permission_mode", mode=mode, session=str(session or ""))


# ---------------- the actions themselves ----------------
def install(packages: list[str]) -> dict:
    bad = [p for p in packages if not _PKG.match(p) or p.startswith("-")]
    if bad:
        raise ValueError(f"not a valid package name: {bad}")
    cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-q", *packages]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    ok = r.returncode == 0
    log("install", packages=packages, ok=ok, python=sys.executable, error=None if ok else r.stderr[-500:])
    if not ok:
        raise RuntimeError(r.stderr[-800:])
    return {"installed": packages, "into": sys.prefix}


def _safe_target(name: str, base: Path) -> Path:
    target = (base / name).resolve()
    if base.resolve() not in target.parents and target != base.resolve():
        raise ValueError(f"{name!r} is outside {base}")
    return target


def download(url: str, name: str | None = None, folder: str = "downloads") -> dict:
    u = urllib.parse.urlparse(url)
    if u.scheme not in ("https", "http"):
        raise ValueError("only http(s) downloads")
    base = workspace() / folder
    base.mkdir(parents=True, exist_ok=True)
    name = name or Path(u.path).name or "download.bin"
    target = _safe_target(name, base)
    tmp = target.with_suffix(target.suffix + ".part")
    n = 0
    req = urllib.request.Request(url, headers={"User-Agent": "vidai"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 16):
            n += len(chunk)
            if n > MAX_DOWNLOAD:
                f.close()
                tmp.unlink(missing_ok=True)
                raise ValueError("download too large")
            f.write(chunk)
    os.replace(tmp, target)
    log("download", url=url, path=str(target), bytes=n)
    return {"path": str(target), "bytes": n}


def download_model(model_id: str) -> dict:
    """Install a catalog model (vidai.hub) into ~/.vidai/assets/hub."""
    from . import hub

    m = hub.CATALOG[model_id]
    target = hub.path(model_id)
    tmp = target.with_suffix(target.suffix + ".part")
    n = 0
    req = urllib.request.Request(m["url"], headers={"User-Agent": "vidai"})
    with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 16):
            n += len(chunk)
            if n > MAX_DOWNLOAD:
                f.close()
                tmp.unlink(missing_ok=True)
                raise ValueError("download too large")
            f.write(chunk)
    os.replace(tmp, target)
    log("install_model", model=model_id, url=m["url"], path=str(target), bytes=n, license=m["license"])
    return {"model": model_id, "path": str(target), "bytes": n, "license": m["license"]}


def create_file(relpath: str, content: str, session: str | None = None) -> dict:
    base = Path(session) if session else workspace()
    target = _safe_target(relpath, base)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    log("create_file", path=str(target), bytes=len(content.encode()))
    return {"path": str(target)}


def describe(action: str, args: dict) -> str:
    """What VidAI says when it asks."""
    if action == "install":
        return f"install {', '.join(args['packages'])}"
    if action == "download":
        return f"download {args.get('name') or Path(urllib.parse.urlparse(args['url']).path).name} from " \
               f"{urllib.parse.urlparse(args['url']).netloc}"
    if action == "create_file":
        return f"create the file {args['relpath']}"
    if action == "model":
        from . import hub

        m = hub.CATALOG[args["model"]]
        return f"download the {m['title']} model ({m['mb']} MB, {m['license']})"
    return action


RUN = {"install": lambda a: install(a["packages"]),
       "download": lambda a: download(a["url"], a.get("name"), a.get("folder", "downloads")),
       "create_file": lambda a: create_file(a["relpath"], a["content"], a.get("session")),
       "model": lambda a: download_model(a["model"])}
