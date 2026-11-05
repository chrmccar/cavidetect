"""Increment 3 — lightweight ML: features, LOPO split, train/predict, baseline fallback."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cavidetect import ml


def _labelled_scene(T=24, step=10, seed=0):
    """Abrupt tube (lumen, label=1) vs gradual patch (drying, label=0) in one tissue body."""
    rng = np.random.default_rng(seed)
    stack = np.full((T, 40, 80), 20.0, np.float32)
    stack[:, 5:35, 5:75] = 110.0
    tissue = np.zeros((40, 80), bool); tissue[5:35, 5:75] = True
    stack[step:, 12:16, 10:40] = 225.0                 # abrupt tube
    ramp = np.linspace(110.0, 225.0, T)
    stack[:, 22:26, 45:70] = ramp[:, None, None]       # gradual patch
    stack += rng.normal(0, 2, stack.shape).astype(np.float32)
    label = np.zeros((40, 80), bool); label[12:16, 10:40] = True   # only the tube is cavitation
    return stack, tissue, label


def test_feature_stack_shape_and_names():
    stack, tissue, _ = _labelled_scene()
    feats = ml.feature_stack(stack, tissue)
    assert feats.shape == (40, 80, len(ml.FEATURE_NAMES))
    assert feats.dtype == np.float32
    assert np.isfinite(feats).all()


def test_leave_one_plant_out_splits_by_plant():
    datasets = {"49": 1, "53": 2, "60": 3}
    splits = list(ml.leave_one_plant_out(datasets))
    assert len(splits) == 3
    for held, train in splits:
        assert held not in train and len(train) == 2


@pytest.mark.skipif(not ml._sklearn(), reason="scikit-learn not installed")
def test_train_and_predict_separates_abrupt_from_gradual():
    stack, tissue, label = _labelled_scene()
    X, y = ml.samples_from(stack, tissue, label)
    assert X.shape[0] == int(tissue.sum()) and set(np.unique(y)).issubset({0, 1})
    clf = ml.train(X, y, kind="logistic")
    proba = ml.predict_proba_map(clf, stack, tissue)
    # (trained and evaluated on the same toy scene — this checks the plumbing separates the
    #  two classes, not generalisation; real eval is leave-one-plant-out on the Mac)
    assert proba[12:16, 10:40].mean() > proba[22:26, 45:70].mean() + 0.2


@pytest.mark.skipif(not ml._sklearn(), reason="scikit-learn not installed")
def test_save_load_roundtrip(tmp_path):
    stack, tissue, label = _labelled_scene()
    X, y = ml.samples_from(stack, tissue, label)
    clf = ml.train(X, y)
    p = ml.save_model(clf, str(tmp_path / "m.joblib"))
    clf2 = ml.load_model(p)
    a = ml.predict_proba_map(clf, stack, tissue)
    b = ml.predict_proba_map(clf2, stack, tissue)
    assert np.allclose(a, b)


def test_combine_blends_in_range():
    ml_p = np.array([[0.0, 1.0], [0.5, 0.2]], np.float32)
    ev = np.array([[1.0, 0.0], [0.5, 0.8]], np.float32)
    c = ml.combine(ml_p, ev, alpha=0.5)
    assert np.allclose(c, [[0.5, 0.5], [0.5, 0.5]])
    assert (c >= 0).all() and (c <= 1).all()
