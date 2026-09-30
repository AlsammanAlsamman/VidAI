import numpy as np
import pytest

from vidai import lab
from vidai.lab.examples import ColorMatch


def _color_data(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 256, (4000, 3)).astype(np.uint8)
    xf = X.astype(np.float32)
    Y = np.clip(0.8 * xf[:, [2, 1, 0]] + 30 + 0.3 * (xf ** 2) / 255, 0, 255).astype(np.uint8)  # swap + curve
    return (X[:3000], Y[:3000]), (X[3000:], Y[3000:])


def test_train_until_suitable_iterates_until_target():
    tr, va = _color_data()

    def adjust(hp, metrics, r):
        return {**hp, "degree": 2}  # linear is not enough -> add curve terms

    rep = lab.train_until_suitable(ColorMatch, tr, va, "mae", 3.0, higher_is_better=False,
                                   hparams={"degree": 1}, adjust=adjust, save_as="color_fix")
    assert rep.suitable and len(rep.rounds) == 2
    assert rep.rounds[0]["metrics"]["mae"] > 3.0 >= rep.rounds[1]["metrics"]["mae"]
    m, spec = lab.load_model("color_fix")
    assert spec.hparams["degree"] == 2
    assert np.abs(m.predict(va[0]).astype(int) - va[1].astype(int)).mean() <= 3.0
    assert [x["name"] for x in lab.list_models()] == ["color_fix"]


def test_not_saved_when_not_suitable():
    tr, va = _color_data()
    rep = lab.train_until_suitable(ColorMatch, tr, va, "mae", 0.01, higher_is_better=False, save_as="nope",
                                   max_rounds=3)
    assert not rep.suitable and rep.saved_to is None and len(rep.rounds) == 1  # no adjust -> stop
    with pytest.raises(KeyError):
        lab.load_model("nope")


def test_classifier_and_code_file_model(tmp_path):
    code = lab.vidai_home() / "work" / "bright.py"  # model code runs only from VidAI's own folders
    code.parent.mkdir(parents=True, exist_ok=True)
    code.write_text(
        "from vidai.lab.examples import LogisticFrameClassifier\n"
        "class Bright(LogisticFrameClassifier):\n    pass\n")
    rng = np.random.default_rng(1)
    X = rng.random((400, 16)).astype(np.float32)
    y = (X.mean(1) > 0.5).astype(int)
    cls = lab._import_class(f"{code}:Bright")
    rep = lab.train_until_suitable(cls, (X[:300], y[:300]), (X[300:], y[300:]), "accuracy", 0.9,
                                   save_as="bright", code_file=code)
    assert rep.suitable
    m, spec = lab.load_model("bright")
    assert spec.class_path == "model.py:Bright"
    assert (m.predict(X[300:]) == y[300:]).mean() >= 0.9
