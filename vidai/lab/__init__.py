"""VidAI Lab: small custom models that Claude builds and trains when regular edits are not enough.

Workflow
1. Regular tools (ffmpeg filters, OpenCV, existing models) are tried first.
2. If they are not good enough, Claude writes a model (a `LabModel` subclass in `model.py`),
   prepares data, and calls `train_until_suitable` (or `train_round` one step at a time).
3. Each round is evaluated against a target metric; Claude adjusts hyper-parameters and repeats
   until the model is suitable, then it is saved to the registry.
4. Registered models become new tools: `ApplyModel` ops in an edit plan.

Registry layout:  $VIDAI_HOME/models/<name>/{spec.json, weights.npz, model.py?}
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np


def vidai_home() -> Path:
    return Path(os.environ.get("VIDAI_HOME", Path.home() / ".vidai"))


def models_dir() -> Path:
    d = vidai_home() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


class LabModel:
    """Base class for custom models.

    task = "frame_transform": implement transform_frame (used by ApplyModel in render)
    task = "frame_classifier": implement predict (used to create anchors / find moments)
    """

    task: str = "frame_transform"

    def __init__(self, **hparams: Any) -> None:
        self.hparams = hparams

    def fit(self, X: Any, Y: Any) -> None:
        raise NotImplementedError

    def evaluate(self, X: Any, Y: Any) -> dict[str, float]:
        raise NotImplementedError

    def predict(self, X: Any) -> Any:
        raise NotImplementedError

    def transform_frame(self, frame: np.ndarray, t: float = 0.0, **params: Any) -> np.ndarray:
        """frame: (H, W, 3) uint8 RGB -> same shape."""
        raise NotImplementedError

    def state(self) -> dict[str, np.ndarray]:
        return {}

    def load_state(self, state: dict[str, np.ndarray]) -> None:
        pass


@dataclass
class ModelSpec:
    name: str
    class_path: str  # "package.module:Class", or "model.py:Class" for code stored with the model
    task: str
    description: str = ""
    hparams: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    target: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    created: str = ""


def _import_class(class_path: str, model_dir: Path | None = None) -> type[LabModel]:
    mod_name, _, cls_name = class_path.partition(":")
    if mod_name.endswith(".py"):
        file = (model_dir / mod_name) if model_dir else Path(mod_name)
        spec = importlib.util.spec_from_file_location(f"vidai_lab_{file.parent.name}", file)
        module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        sys.modules[spec.name] = module  # type: ignore[union-attr]
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    else:
        module = importlib.import_module(mod_name)
    return getattr(module, cls_name)


def save_model(model: LabModel, spec: ModelSpec, code_file: str | Path | None = None) -> Path:
    d = models_dir() / spec.name
    d.mkdir(parents=True, exist_ok=True)
    if code_file:
        shutil.copy(code_file, d / "model.py")
        spec.class_path = "model.py:" + spec.class_path.partition(":")[2]
    spec.created = spec.created or time.strftime("%Y-%m-%d %H:%M:%S")
    np.savez(d / "weights.npz", **model.state())
    (d / "spec.json").write_text(json.dumps(asdict(spec), indent=2, default=float))
    return d


def load_model(name: str) -> tuple[LabModel, ModelSpec]:
    d = models_dir() / name
    if not (d / "spec.json").exists():
        raise KeyError(f"model '{name}' not found in {models_dir()}")
    spec = ModelSpec(**json.loads((d / "spec.json").read_text()))
    cls = _import_class(spec.class_path, d)
    model = cls(**spec.hparams)
    w = d / "weights.npz"
    if w.exists():
        with np.load(w) as z:
            model.load_state({k: z[k] for k in z.files})
    return model, spec


def list_models() -> list[dict[str, Any]]:
    out = []
    for d in sorted(models_dir().iterdir()):
        if (d / "spec.json").exists():
            s = json.loads((d / "spec.json").read_text())
            out.append({k: s.get(k) for k in ("name", "task", "description", "metrics", "created")})
    return out


def delete_model(name: str) -> None:
    shutil.rmtree(models_dir() / name)


def _meets(metrics: dict[str, float], metric: str, target: float, higher_is_better: bool) -> bool:
    v = metrics.get(metric)
    return v is not None and (v >= target if higher_is_better else v <= target)


def train_round(model_cls: type[LabModel], hparams: dict[str, Any], train: tuple, val: tuple) -> tuple[LabModel, dict]:
    model = model_cls(**hparams)
    t0 = time.time()
    model.fit(*train)
    metrics = model.evaluate(*val)
    metrics["train_seconds"] = round(time.time() - t0, 3)
    return model, metrics


@dataclass
class TrainReport:
    suitable: bool
    best_metrics: dict[str, float]
    best_hparams: dict[str, Any]
    rounds: list[dict[str, Any]]
    saved_to: str | None = None


def train_until_suitable(
    model_cls: type[LabModel],
    train: tuple,
    val: tuple,
    metric: str,
    target: float,
    higher_is_better: bool = True,
    hparams: dict[str, Any] | None = None,
    adjust: Callable[[dict[str, Any], dict[str, float], int], dict[str, Any] | None] | None = None,
    max_rounds: int = 10,
    save_as: str | None = None,
    description: str = "",
    code_file: str | Path | None = None,
) -> TrainReport:
    """Train, evaluate, adjust, repeat until `metric` reaches `target` (or max_rounds).

    `adjust(hparams, metrics, round) -> next hparams` is where Claude (or a default rule)
    changes the model settings. Returning None stops early.
    """
    hp = dict(hparams or {})
    rounds: list[dict[str, Any]] = []
    best: tuple[LabModel, dict, dict] | None = None
    for r in range(max_rounds):
        model, m = train_round(model_cls, hp, train, val)
        rounds.append({"round": r, "hparams": dict(hp), "metrics": m})
        worst = -np.inf if higher_is_better else np.inf
        v, bv = m.get(metric, worst), (best[1].get(metric, worst) if best else worst)
        if best is None or (v > bv if higher_is_better else v < bv):
            best = (model, m, dict(hp))
        if _meets(m, metric, target, higher_is_better):
            break
        nxt = adjust(hp, m, r) if adjust else None
        if nxt is None:
            break
        hp = nxt
    assert best is not None
    model, m, hp = best
    ok = _meets(m, metric, target, higher_is_better)
    report = TrainReport(ok, m, hp, rounds)
    if save_as and ok:
        cls_path = f"{model_cls.__module__}:{model_cls.__name__}"
        spec = ModelSpec(name=save_as, class_path=cls_path, task=model.task, description=description,
                         hparams=hp, metrics=m, target={"metric": metric, "value": target,
                                                         "higher_is_better": higher_is_better},
                         history=rounds)
        report.saved_to = str(save_model(model, spec, code_file))
    return report
