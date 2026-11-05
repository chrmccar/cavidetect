"""Lightweight trainable classifier (v0.2.0 brief §3).

Audit (2026-11-05): the repo had **no** ML — the ``annotated/`` masks were single-frame
hand paints never used at inference. This module adds a *lightweight* feature classifier
(logistic regression / random forest), appropriate when only ~3 independent plants are
labelled. It distinguishes lumen cavitation from inter-rhizoid drying, scale whitening and
tape by scoring per-pixel features drawn from the raw final frame, 1-/2-frame positive
differences, the step-vs-gradual score, ridge tubeness and local texture.

Design rules from the brief:
* **Combine, don't replace.** The model yields a probability; callers blend it with the
  ridge/temporal evidence (:func:`combine`). A baseline-only (no-ML) mode always works.
* **Split by plant/dry-down, never by crop/frame** for evaluation (:func:`leave_one_plant_out`).
* **Optional dependency.** scikit-learn is an extra; if it is absent, training/prediction
  raise a clear error and the detector simply runs without ML.

Features are deterministic and cheap; this is intentionally not deep learning.
"""
from __future__ import annotations
import numpy as np

FEATURE_NAMES = ["brightness", "jump1", "jump2", "step_score", "ridge", "local_std",
                 "persistence"]


def _sklearn():
    try:
        import sklearn  # noqa: F401
        return True
    except Exception:
        return False


def feature_stack(stack, tissue, ridge=None, smooth=3):
    """Dense per-pixel feature maps as an (H, W, F) float array, F == len(FEATURE_NAMES).

    Reuses the temporal evidence maps so the ML features and the temporal method agree.
    ``ridge`` (normalised Frangi) is computed if omitted.
    """
    from .temporal import temporal_evidence, _smooth_t
    from scipy.ndimage import uniform_filter
    stack = np.asarray(stack, np.float32)
    final = stack[-max(stack.shape[0] // 8, 1):].mean(0)
    if ridge is None:
        from .colour_shape import ridge_map
        ridge = ridge_map(final, tissue)
    te = temporal_evidence(stack, tissue, smooth=smooth, min_confidence=0.0,
                           min_jump_frac=0.0)
    sm = _smooth_t(stack, smooth)
    d1 = np.maximum(sm[1:] - sm[:-1], 0).max(0) if stack.shape[0] > 1 else np.zeros_like(final)
    d2 = np.maximum(sm[2:] - sm[:-2], 0).max(0) if stack.shape[0] > 2 else d1
    mean = uniform_filter(final, 5)
    sq = uniform_filter(final * final, 5)
    local_std = np.sqrt(np.maximum(sq - mean * mean, 0))

    def _n(a):                                   # robust 0..1 scaling within tissue
        v = a[tissue] if tissue.any() else a
        hi = np.percentile(v, 99) if v.size else 1.0
        return np.clip(a / (hi + 1e-9), 0, 1)

    feats = np.stack([_n(final), _n(d1), _n(d2), te["step_score"], np.clip(ridge, 0, 1),
                      _n(local_std), te["confidence"]], axis=-1).astype(np.float32)
    return feats


def samples_from(stack, tissue, label_mask, ridge=None):
    """(X, y) training rows for one dataset: every tissue pixel, labelled by ``label_mask``
    (True = lumen cavitation). Split by plant happens at the dataset level, not here."""
    feats = feature_stack(stack, tissue, ridge)
    sel = np.asarray(tissue, dtype=bool)
    X = feats[sel]
    y = np.asarray(label_mask, dtype=bool)[sel].astype(np.int8)
    return X, y


def train(X, y, kind="logistic", class_weight="balanced", seed=0):
    """Fit a lightweight classifier. Raises if scikit-learn is unavailable."""
    if not _sklearn():
        raise RuntimeError("scikit-learn not installed; install the 'extra' dependencies "
                           "or run in baseline (no-ML) mode")
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == "forest":
        clf = RandomForestClassifier(n_estimators=120, max_depth=8, random_state=seed,
                                     class_weight=class_weight, n_jobs=-1)
        clf.fit(X, y)
        return clf
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=500, class_weight=class_weight,
                                           random_state=seed))
    clf.fit(X, y)
    return clf


def predict_proba_map(clf, stack, tissue, ridge=None):
    """Per-pixel P(lumen cavitation) map; 0 outside tissue."""
    feats = feature_stack(stack, tissue, ridge)
    H, W, F = feats.shape
    sel = np.asarray(tissue, dtype=bool)
    proba = np.zeros((H, W), np.float32)
    if sel.any():
        p = clf.predict_proba(feats[sel])[:, 1]
        proba[sel] = p.astype(np.float32)
    return proba


def combine(ml_proba, evidence, alpha=0.5):
    """Blend an ML probability map with a (0..1) evidence/confidence map. ``alpha`` weights ML."""
    ml_proba = np.asarray(ml_proba, np.float32)
    evidence = np.asarray(evidence, np.float32)
    return np.clip(alpha * ml_proba + (1 - alpha) * evidence, 0, 1)


def leave_one_plant_out(datasets):
    """Yield (held_out_plant, train_plants) splits. ``datasets`` maps plant_id -> anything.

    Enforces the brief's rule: evaluation splits are by plant/dry-down, never by crop/frame.
    """
    plants = list(datasets)
    for held in plants:
        yield held, [p for p in plants if p != held]


def save_model(clf, path):
    import joblib
    joblib.dump(clf, path)
    return path


def load_model(path):
    import joblib
    return joblib.load(path)
