"""Detection-method registry + parameter schema.

Each method is a ``Method(name, label, detect, schema)``. ``detect(final,
tissue, **params) -> dict`` returns at least ``shape`` and ``colour`` boolean
masks (``ridge`` optional). ``schema`` is a list of ``ParamSpec`` that is the
single source of truth: the GUI builds its settings panel from it and the CLI
validates ``--params`` against it via ``resolve_params``.

Scale convention: ``detect`` reads ``um_per_px`` from its params as the
micrometres per *analysis* pixel. The caller (``colour_shape.measure``) folds
binning in by passing ``um_per_px * bin`` while recording the true optical
value in ``params.json``; a direct call with ``bin=1`` uses the value as-is.
"""
from __future__ import annotations
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class ParamSpec:
    key: str
    label: str
    type: str                       # "float" | "int" | "bool" | "choice"
    default: object
    min: float | None = None
    max: float | None = None
    choices: tuple | None = None
    unit: str | None = None
    group: str = "General"
    help: str = ""


@dataclass
class Method:
    name: str
    label: str
    detect: Callable
    schema: list = field(default_factory=list)
    needs_stack: bool = False           # detect() also wants the full frame stack (temporal)


REGISTRY: "dict[str, Method]" = {}


def register(m: Method) -> None:
    REGISTRY[m.name] = m


def get_method(name: str) -> Method:
    try:
        return REGISTRY[name]
    except KeyError:
        valid = ", ".join(REGISTRY)
        raise KeyError(f"unknown method {name!r}; valid methods: {valid}")


def list_methods() -> "list[tuple[str, str]]":
    """(name, label) in registration order (colour-frangi first)."""
    return [(m.name, m.label) for m in REGISTRY.values()]


def _clamp(v, spec: ParamSpec):
    if spec.min is not None and v < spec.min:
        v = spec.min
    if spec.max is not None and v > spec.max:
        v = spec.max
    return v


def _coerce(spec: ParamSpec, v):
    if v is None:                       # optional param left at "auto"
        return None
    if spec.type == "bool":
        if isinstance(v, str):
            return v.strip().lower() in ("1", "true", "yes", "on")
        return bool(v)
    if spec.type == "int":
        return int(_clamp(int(round(float(v))), spec))
    if spec.type == "float":
        return float(_clamp(float(v), spec))
    if spec.type == "choice":
        return v if (spec.choices and v in spec.choices) else spec.default
    return v


def resolve_params(method_name: str, raw: dict) -> dict:
    """Full param dict: schema defaults overlaid with coerced/clamped ``raw``.
    Unknown keys are dropped with a warning."""
    method = get_method(method_name)
    specs = {s.key: s for s in method.schema}
    out = {s.key: s.default for s in method.schema}
    for k, v in (raw or {}).items():
        if k not in specs:
            warnings.warn(f"unknown param {k!r} for method {method_name!r}; ignored")
            continue
        out[k] = _coerce(specs[k], v)
    return out


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

def _shape_rules(p: dict) -> dict:
    """Build the filter_regions rule dict from resolved params."""
    return {
        "inscribed_um": (p["inscribed_lo_um"], p["inscribed_hi_um"]),
        "thinness": (p["thinness_lo"], p["thinness_hi"]),
        "elongation": (p["elongation_lo"], p["elongation_hi"]),
        "min_area_px": p.get("min_area_px", p.get("min_area")),
    }


# --------------------------------------------------------------------------- #
# Method 1: colour-frangi (current behaviour; default)
# --------------------------------------------------------------------------- #

def _detect_colour_frangi(final, tissue, **p):
    from . import colour_shape as cs
    d = cs.detect_spatial(final, tissue=tissue, air=p.get("air"),
                          bright_pct=p["bright_pct"], vessel_min=p["vessel_min"],
                          ecc_min=p["ecc_min"], sol_max=p["sol_max"], min_area=p["min_area"])
    if p.get("apply_shape_filter"):
        from scipy.ndimage import label
        from .shape_filters import filter_regions
        lbl, _ = label(d["shape"])
        d["shape"] = filter_regions(lbl, _shape_rules(p), p["um_per_px"])
    return d


_SHAPE_RULE_SPECS = [
    ParamSpec("inscribed_lo_um", "Inscribed circle min", "float", 7.0, 0.0, unit="µm",
              group="Shape rules", help="smallest tube width to keep (max inscribed circle)"),
    ParamSpec("inscribed_hi_um", "Inscribed circle max", "float", 25.0, 0.0, unit="µm",
              group="Shape rules", help="largest tube width to keep"),
    ParamSpec("thinness_lo", "Thinness min", "float", 0.0, 0.0, 1.0,
              group="Shape rules", help="4*pi*A/P^2; 1=round, 0=line"),
    ParamSpec("thinness_hi", "Thinness max", "float", 0.6, 0.0, 1.0,
              group="Shape rules", help="keep below this (reject round)"),
    ParamSpec("elongation_lo", "Elongation min", "float", 0.6, 0.0, 1.0,
              group="Shape rules", help="1-minor/major; 0=round, 1=line"),
    ParamSpec("elongation_hi", "Elongation max", "float", 1.0, 0.0, 1.0,
              group="Shape rules"),
]

register(Method(
    "colour-frangi", "Colour + frangi tube-ness (current, default)",
    _detect_colour_frangi,
    schema=[
        ParamSpec("um_per_px", "Pixel size", "float", 2.5, 0.001, unit="µm/px",
                  group="General", help="true optical scale; shape rules use µm"),
        ParamSpec("bright_pct", "Air-colour percentile", "int", 65, 0, 100,
                  group="Threshold", help="brightness cut within tissue"),
        ParamSpec("air", "Manual air threshold", "float", None, group="Threshold",
                  help="absolute grey-level cut; leave blank for the percentile above"),
        ParamSpec("vessel_min", "Min tubeness", "float", 0.04, 0.0, 1.0, group="Shape rules"),
        ParamSpec("ecc_min", "Min eccentricity", "float", 0.9, 0.0, 1.0, group="Shape rules"),
        ParamSpec("sol_max", "Max solidity", "float", 0.6, 0.0, 1.0, group="Shape rules"),
        ParamSpec("min_area", "Min area", "int", 8, 1, unit="px", group="Shape rules"),
        ParamSpec("apply_shape_filter", "Also apply BioVoxxel shape rules", "bool", False,
                  group="Shape rules",
                  help="extra gate: inscribed circle / thinness / elongation"),
    ] + _SHAPE_RULE_SPECS,
))


# --------------------------------------------------------------------------- #
# Method 2: shape-filter (expert's ImageJ flow, pure Python)
# --------------------------------------------------------------------------- #

def _detect_shape_filter(final, tissue, **p):
    from scipy.ndimage import label
    from skimage.filters import threshold_otsu
    from .preprocess_final import preprocess_final
    from .shape_filters import filter_regions
    scale = p["um_per_px"]
    pre = preprocess_final(final, contrast=p["contrast"], gamma=p["gamma"],
                           clahe_clip=p["clahe_clip"], bandpass=p["bandpass"],
                           bp_small_um=p["bp_small_um"], bp_large_um=p["bp_large_um"],
                           um_per_analysis_px=scale)
    vals = pre[tissue]
    if p["threshold"] == "otsu":
        thr = float(threshold_otsu(vals))
    else:
        thr = float(np.percentile(vals, p["bright_pct"]))
    colour = tissue & (pre >= thr)
    lbl, _ = label(colour)
    shape = filter_regions(lbl, _shape_rules(p), scale)
    return dict(tissue=tissue, air=thr, colour=colour, shape=shape,
                ridge=np.zeros(final.shape, np.float32))


register(Method(
    "shape-filter", "Shape filter (inscribed circle / thinness / elongation)",
    _detect_shape_filter,
    schema=[
        ParamSpec("um_per_px", "Pixel size", "float", 2.5, 0.001, unit="µm/px",
                  group="General", help="true optical scale; shape rules use µm"),
        ParamSpec("contrast", "Contrast", "choice", "none",
                  choices=("none", "gamma", "clahe"), group="Preprocess"),
        ParamSpec("gamma", "Gamma", "float", 1.0, 0.1, 5.0, group="Preprocess",
                  help="<1 brightens faint tubes (contrast=gamma)"),
        ParamSpec("clahe_clip", "CLAHE clip", "float", 0.01, 0.001, 0.1, group="Preprocess"),
        ParamSpec("bandpass", "Band-pass (DoG)", "bool", False, group="Preprocess",
                  help="keep structures between the two sizes below"),
        ParamSpec("bp_small_um", "Band-pass small", "float", 2.5, 0.0, unit="µm",
                  group="Preprocess"),
        ParamSpec("bp_large_um", "Band-pass large", "float", 40.0, 0.0, unit="µm",
                  group="Preprocess"),
        ParamSpec("threshold", "Threshold", "choice", "percentile",
                  choices=("percentile", "otsu"), group="Threshold"),
        ParamSpec("bright_pct", "Percentile", "int", 65, 0, 100, group="Threshold",
                  help="used when threshold = percentile"),
        ParamSpec("min_area_px", "Min area", "int", 8, 1, unit="px", group="Shape rules"),
    ] + _SHAPE_RULE_SPECS,
))


# --------------------------------------------------------------------------- #
# Method 4: temporal-evidence (re-engages the frame stack; §2 of the v0.2.0 brief)
# --------------------------------------------------------------------------- #

def _detect_temporal(final, tissue, stack=None, **p):
    from .temporal import temporal_evidence
    if stack is None:
        raise ValueError("temporal-evidence needs the frame stack")
    return temporal_evidence(stack, tissue, smooth=p["smooth"],
                             max_transition=p["max_transition"], rise_frac=p["rise_frac"],
                             min_jump_frac=p["min_jump_frac"],
                             min_confidence=p["min_confidence"],
                             baseline_pct=p["baseline_pct"])


register(Method(
    "temporal-evidence", "Temporal evidence (abrupt 1-2 frame fill; confidence)",
    _detect_temporal,
    needs_stack=True,
    schema=[
        ParamSpec("um_per_px", "Pixel size", "float", 2.5, 0.001, unit="µm/px", group="General"),
        ParamSpec("smooth", "Temporal smoothing", "int", 3, 1, 15, unit="frames",
                  group="General", help="rolling mean before differencing"),
        ParamSpec("max_transition", "Max transition", "int", 2, 1, 5, unit="frames",
                  group="Temporal", help="a fill may span at most this many frames"),
        ParamSpec("rise_frac", "Onset rise fraction", "float", 0.5, 0.0, 1.0,
                  group="Temporal", help="pixel onset = first crossing of this fraction of its rise"),
        ParamSpec("min_jump_frac", "Min jump fraction", "float", 0.25, 0.0, 1.0,
                  group="Temporal", help="step must be >= this fraction of the pixel's full rise"),
        ParamSpec("min_confidence", "Min confidence", "float", 0.15, 0.0, 1.0,
                  group="Threshold", help="keep candidates at/above this (permissive = high recall)"),
        ParamSpec("baseline_pct", "Baseline percentile", "int", 20, 0, 100, group="Temporal"),
    ],
))


# --------------------------------------------------------------------------- #
# Method 3: ridge-hysteresis (experimental; exposes detect_spatial_v2)
# --------------------------------------------------------------------------- #

def _detect_ridge_hysteresis(final, tissue, **p):
    from . import colour_shape as cs
    return cs.detect_spatial_v2(final, tissue=tissue,
                                bright_hi_pct=p["bright_hi_pct"],
                                bright_lo_pct=p["bright_lo_pct"],
                                seed_ridge=p["seed_ridge"], grow_ridge=p["grow_ridge"],
                                min_area=p["min_area"], scale_area=p["scale_area"],
                                scale_ecc=p["scale_ecc"], border_px=p["border_px"])


register(Method(
    "ridge-hysteresis", "Ridge hysteresis (experimental)",
    _detect_ridge_hysteresis,
    schema=[
        ParamSpec("um_per_px", "Pixel size", "float", 2.5, 0.001, unit="µm/px", group="General"),
        ParamSpec("bright_hi_pct", "Seed percentile", "int", 70, 0, 100, group="Threshold"),
        ParamSpec("bright_lo_pct", "Grow percentile", "int", 45, 0, 100, group="Threshold"),
        ParamSpec("seed_ridge", "Seed tubeness", "float", 0.08, 0.0, 1.0, group="Shape rules"),
        ParamSpec("grow_ridge", "Grow tubeness", "float", 0.02, 0.0, 1.0, group="Shape rules"),
        ParamSpec("min_area", "Min area", "int", 8, 1, unit="px", group="Shape rules"),
        ParamSpec("scale_area", "Scale reject area", "int", 600, 1, unit="px", group="Shape rules"),
        ParamSpec("scale_ecc", "Scale reject ecc", "float", 0.85, 0.0, 1.0, group="Shape rules"),
        ParamSpec("border_px", "Border distance", "int", 10, 0, unit="px", group="Shape rules"),
    ],
))
