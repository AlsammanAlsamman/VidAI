"""VidAI's own privileged actions: install packages, download files, create files in its workspace.

VidAI runs them itself (so Claude Code never needs a shell or a permission prompt), but only with the user's
consent for the session:
  - ask mode (default): VidAI says "Master, I need to ... Say 'VidAI confirm' or 'VidAI deny'" and waits
  - full mode: the user said "VidAI, take all actions" -> everything is allowed for this session
Limits that always apply: installs go only into VidAI's own Python environment (pip, never system), files
only under ~/.vidai/ (workspace) or a real session folder, downloads only over https from public hosts, up to
MAX_DOWNLOAD bytes. Every action is appended to ~/.vidai/actions.log.

Consent is given only by the user: by voice or a button in the recorder, or (outside a recording) through
Claude Code's own permission prompt for the `vidai_confirmed_action` tool. Claude can lower the mode to "ask"
but never raise it to "full"; full access expires after FULL_HOURS.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

MAX_DOWNLOAD = 2 * 1024 ** 3  # 2 GB
FULL_HOURS = 4.0
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,63}$")
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


def _perms(session: str | Path | None) -> dict:
    try:
        d = json.loads(_perm_file(session).read_text())
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_perms(session: str | Path | None, d: dict) -> None:
    p = _perm_file(session)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(d))
    os.replace(tmp, p)


def get_mode(session: str | Path | None) -> str:
    d = _perms(session)
    if d.get("mode") == "full" and time.time() < float(d.get("until", 0)):
        return "full"
    return "ask"


def set_mode(session: str | Path | None, mode: str, hours: float = FULL_HOURS) -> None:
    d = _perms(session)
    d.update({"mode": mode, "since": time.strftime("%Y-%m-%d %H:%M:%S")})
    if mode == "full":
        d["until"] = time.time() + hours * 3600
    else:
        d.pop("until", None)
    _save_perms(session, d)
    log("permission_mode", mode=mode, session=str(session or ""))


def code_allowed(session: str | Path | None) -> bool:
    """May Claude-written effect code run in this session? (the user said yes once, or full access)"""
    return get_mode(session) == "full" or bool(_perms(session).get("code"))


def allow_code(session: str | Path | None) -> None:
    d = _perms(session)
    d["code"] = True
    _save_perms(session, d)
    log("allow_code", session=str(session or ""))


def safe_name(name: str, what: str = "name") -> str:
    """A plain identifier used in a file name (no paths, no '..')."""
    if not isinstance(name, str) or not _NAME.match(name) or ".." in name:
        raise ValueError(f"invalid {what} {name!r}: use letters, digits, '_', '-', '.' (max 64)")
    return name


def session_dir(session: str | Path) -> Path:
    """A real VidAI session folder (has session.json), resolved."""
    d = Path(session).expanduser().resolve()
    if not (d / "session.json").is_file():
        raise ValueError(f"{session!r} is not a VidAI session folder")
    return d


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


def download(url: str, name: str | None = None, folder: str = "downloads", sha256: str | None = None) -> dict:
    from .models_dl import fetch

    base = _safe_target(folder, workspace())
    base.mkdir(parents=True, exist_ok=True)
    name = name or Path(urllib.parse.urlparse(url).path).name or "download.bin"
    target = _safe_target(name, base)
    _, n, digest = fetch(url, target, sha256=sha256, max_bytes=MAX_DOWNLOAD, timeout=120)
    log("download", url=url, path=str(target), bytes=n, sha256=digest)
    return {"path": str(target), "bytes": n, "sha256": digest}


def download_model(model_id: str) -> dict:
    """Install a catalog model (vidai.hub) into ~/.vidai/assets/hub (checksum-verified)."""
    from . import hub
    from .models_dl import fetch

    m = hub.CATALOG[model_id]
    target = hub.path(model_id)
    _, n, digest = fetch(m["url"], target, sha256=m.get("sha256"), pin_key=f"hub/{m['file']}",
                         max_bytes=MAX_DOWNLOAD)
    log("install_model", model=model_id, url=m["url"], path=str(target), bytes=n, license=m["license"],
        sha256=digest)
    return {"model": model_id, "path": str(target), "bytes": n, "license": m["license"]}


def create_file(relpath: str, content: str, session: str | None = None) -> dict:
    base = session_dir(session) if session else workspace()
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
       "download": lambda a: download(a["url"], a.get("name"), sha256=a.get("sha256")),
       "create_file": lambda a: create_file(a["relpath"], a["content"], a.get("session")),
       "model": lambda a: download_model(a["model"])}
