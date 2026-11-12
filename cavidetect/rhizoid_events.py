"""Per-rhizoid cavitation detection (method ``rhizoid-events``).

Works at the level of *individual pegged rhizoids* rather than total embolised
area. A rhizoid is a tube; cavitation is a **whole-strand brightness jump in one
frame (rarely two)**. A *partial* brightening along a strand is **drying**, not
cavitation, and is not counted.

Pipeline (Approach A, design 2026-11-12):
  CLAHE every (windowed) frame -> build a late-tail reference where cavitated
  strands are brightest -> ridge-trace individual rhizoids (split bundles at
  junctions, merge over-split collinear fragments) -> read each strand's
  brightness profile backwards through the stack -> classify whole-strand jump
  (cavitation) vs partial/gradual rise (drying).

The module returns the registry-standard ``shape``/``colour``/``onset`` masks so
every existing output keeps working, plus ``tubes`` (per-rhizoid instance map)
and ``events`` (per-rhizoid timing records) consumed by the new output writers
and the Inspector correction loop.
"""
from __future__ import annotations
import numpy as np


# --------------------------------------------------------------------------- #
# Preprocessing + reference
# --------------------------------------------------------------------------- #

def reference_frame(cstack, ref_tail_n=8):
    """Reference = per-pixel max over the last ``ref_tail_n`` frames.

    Strands are brightest *after* cavitation, so the late tail is where
    individual rhizoids are most separable. Max (not mean) keeps a one-frame
    bright event from being diluted by neighbouring dark frames.
    """
    cstack = np.asarray(cstack, np.float32)
    n = max(1, min(int(ref_tail_n), cstack.shape[0]))
    return cstack[-n:].max(0)


# --------------------------------------------------------------------------- #
# Instance tracing
# --------------------------------------------------------------------------- #

def _segment_geometry(seg, lab):
    """(centroid, unit principal direction, two endpoints) for one segment label."""
    ys, xs = np.where(seg == lab)
    pts = np.column_stack([ys, xs]).astype(np.float64)
    c = pts.mean(0)
    d = pts - c
    if len(pts) >= 2:
        _, _, vt = np.linalg.svd(d, full_matrices=False)
        axis = vt[0]
    else:
        axis = np.array([1.0, 0.0])
    proj = d @ axis
    e0 = pts[proj.argmin()]
    e1 = pts[proj.argmax()]
    norm = np.linalg.norm(axis)
    axis = axis / norm if norm else np.array([1.0, 0.0])
    return c, axis, e0, e1


def _merge_collinear(seg, gap_max=8.0, angle_tol_deg=35.0):
    """Merge over-split collinear fragments of one rhizoid into a single id.

    Segments are joined when a free end of one continues almost straight into a
    free end of another (small gap AND both segment axes aligned with the join
    direction). At a *fork* — one segment whose end sits near two others — only
    the straightest continuation is joined, so genuinely diverging rhizoids that
    share an upper trunk stay separate individuals.

    ``seg`` is an int32 label image (0 = background). Returns a relabelled int32
    image. Pure geometry; deterministic; no ML.
    """
    seg = np.asarray(seg, np.int32)
    labels = [int(v) for v in np.unique(seg) if v != 0]
    if len(labels) <= 1:
        return _relabel(seg)

    geom = {lab: _segment_geometry(seg, lab) for lab in labels}
    # every free endpoint: (label, endpoint-xy, outward unit direction)
    ends = []
    for lab in labels:
        c, axis, e0, e1 = geom[lab]
        ends.append((lab, e0, _unit(e0 - c)))
        ends.append((lab, e1, _unit(e1 - c)))

    cos_tol = np.cos(np.deg2rad(angle_tol_deg))
    # candidate joins between endpoints of *different* segments, scored by straightness
    cand = []
    for i in range(len(ends)):
        la, pa, da = ends[i]
        for j in range(i + 1, len(ends)):
            lb, pb, db = ends[j]
            if la == lb:
                continue
            join = pb - pa
            gap = np.linalg.norm(join)
            if gap > gap_max:
                continue
            u = _unit(join)
            # pa's outward dir should point toward pb, pb's toward pa, and the two
            # segment axes should be collinear with the join (a straight continuation)
            s_a = float(da @ u)
            s_b = float(db @ (-u))
            if gap > 1e-6 and (s_a < cos_tol or s_b < cos_tol):
                continue
            score = (s_a + s_b) - 0.05 * gap       # straighter + closer = better
            cand.append((score, la, lb, i, j))

    cand.sort(reverse=True)
    parent = {lab: lab for lab in labels}
    used_end = set()
    for score, la, lb, i, j in cand:
        if i in used_end or j in used_end:         # each endpoint joins at most once
            continue
        if _find(parent, la) == _find(parent, lb):
            continue
        used_end.add(i); used_end.add(j)
        parent[_find(parent, la)] = _find(parent, lb)

    out = np.zeros_like(seg)
    for lab in labels:
        out[seg == lab] = _find(parent, lab)
    return _relabel(out)


def trace_rhizoids(raw_reference, tissue, divergence_merge=True,
                   clahe_reference=True, gauss_sigma=1.2, ridge_thr=0.04,
                   close_iter=2, min_strand_px=20, min_seg_px=6,
                   gap_max=14.0, angle_tol_deg=45.0,
                   clahe_blocksize=127, clahe_bins=256, clahe_maxslope=1.5):
    """Trace individual rhizoids as tubular ridges in the reference frame.

    CLAHE the reference (single frame — honours the expert's "CLAHE first" for
    faint-rhizoid visibility without touching the temporal DN that timing uses)
    -> smooth -> tubeness ridge -> bridge small gaps -> drop short fragments ->
    skeletonise + split at junctions (``label_tubes``) so crossings/forks
    separate -> re-merge over-split collinear fragments of one strand. Returns
    an int32 instance map (0 = background, 1..K = rhizoids) plus the ridge map.

    The whole tube (faint base *and* bright fill) is traced from its ridge
    structure, not its brightness, so the filled *fraction* can be measured
    later to tell a whole-strand cavitation from a partial drying event.
    """
    from . import colour_shape as cs
    from . import inspector as insp
    from . import preprocess_final as pp
    from scipy.ndimage import gaussian_filter, binary_closing, label as nlabel

    ref = pp.apply_contrast(raw_reference, mode="clahe", clahe_blocksize=clahe_blocksize,
                            clahe_bins=clahe_bins, clahe_maxslope=clahe_maxslope) \
        if clahe_reference else np.asarray(raw_reference, np.float32)
    sm = gaussian_filter(ref, gauss_sigma)
    ridge = cs.ridge_map(sm, tissue)
    cand = tissue & (ridge >= ridge_thr)
    if close_iter > 0:
        cand = binary_closing(cand, iterations=int(close_iter))
    cand = cand & tissue
    lbl, _ = nlabel(cand)                        # drop fragments shorter than a rhizoid
    sizes = np.bincount(lbl.ravel())
    small = np.where(sizes < int(min_strand_px))[0]
    cand = cand & ~np.isin(lbl, small)
    tubes = insp.label_tubes(cand, min_seg_px=min_seg_px)
    if divergence_merge and tubes.max() > 1:
        tubes = _merge_collinear(tubes, gap_max=gap_max, angle_tol_deg=angle_tol_deg)
    return tubes.astype(np.int32), ridge.astype(np.float32)


# --------------------------------------------------------------------------- #
# Per-strand history (read backwards) + classification
# --------------------------------------------------------------------------- #

def strand_signals(stack, strand_mask, corridor_dilate_px=3,
                   min_contrast=15.0, cross_frac=0.5):
    """Two per-frame signals for one strand, read from the raw stack.

    * ``bright_frac`` (length T, 0..1) — fraction of *whole-strand* pixels that
      have crossed into air-colour. Denominator is every strand pixel, so a
      partial fill reads ~0.5 and a whole-strand fill ~1.0 (the spatial
      cavitation-vs-drying signal). A pixel is air-colour once it crosses
      baseline + ``cross_frac`` of its own dark->bright span, and only if that
      span clears ``min_contrast``.
    * ``norm_mean`` (length T, 0..1) — the strand's mean brightness normalised to
      its own dark->bright range (the *temporal* signal: a cavitation rise
      completes in 1-2 frames, a drying rise is gradual).

    Small drift is absorbed by sampling each pixel as the max over a
    ``corridor_dilate_px`` neighbourhood (keeps the denominator fixed to the
    strand rather than inflating it). Raw DN only — never CLAHE, whose per-frame
    non-linearity would corrupt the absolute jumps the temporal signal keys on.
    """
    stack = np.asarray(stack, np.float32)
    T = stack.shape[0]
    ys, xs = np.where(strand_mask)
    P = len(ys)
    if P == 0:
        return np.zeros(T, np.float32), np.zeros(T, np.float32)
    if corridor_dilate_px and int(corridor_dilate_px) > 0:
        from scipy.ndimage import maximum_filter
        size = 2 * int(corridor_dilate_px) + 1
        prof = np.empty((T, P), np.float32)
        for t in range(T):
            prof[t] = maximum_filter(stack[t], size=size)[ys, xs]
    else:
        prof = stack[:, ys, xs]                   # (T, P)
    base = prof[0]
    top = prof.max(0)
    span = top - base
    active = span >= min_contrast                # pixels that actually brighten
    thr = base + cross_frac * span
    crossed = (prof >= thr[None, :]) & active[None, :]
    bright_frac = crossed.sum(1).astype(np.float32) / float(P)

    mean = prof.mean(1)
    lo, hi = float(mean.min()), float(mean.max())
    mean_abs_span = hi - lo                       # absolute DN swing of the strand mean
    norm_mean = ((mean - lo) / mean_abs_span).astype(np.float32) if mean_abs_span > 1e-6 \
        else np.zeros(T, np.float32)
    return bright_frac, norm_mean, float(mean_abs_span)


def classify_strand(bright_frac, norm_mean, mean_abs_span=None, thresh=0.7,
                    step_window=2, gradual_reject=0.3, min_rise=0.2,
                    min_contrast=15.0):
    """Classify a strand from its spatial-coverage and temporal signals.

    Returns ``(cls, onset_frame, confidence, peak)`` where ``cls`` is
    ``"cavitation"`` / ``"drying"`` / ``"none"``.

    * cavitation ⇔ the WHOLE strand lights (peak ``bright_frac`` >= ``thresh``)
      AND it lights FAST — the mean brightness rises from 0.2 to 0.8 of its range
      within ``step_window`` frames. Onset = the frame of the steepest rise.
    * drying ⇔ a meaningful rise that is either partial (peak coverage < thresh)
      or gradual (mean-brightness rise slower than ``step_window`` frames).
    * none ⇔ no meaningful rise in either signal.

    ``gradual_reject`` floors the single-frame rise of ``norm_mean`` at the onset,
    guarding against a slow creep that happens to pass the span test on noise.
    """
    bf = np.asarray(bright_frac, np.float32)
    nm = np.asarray(norm_mean, np.float32)
    T = len(bf)
    peak = float(bf.max()) if T else 0.0
    cov_rise = peak - float(bf.min()) if T else 0.0
    # 'none' is gated on ABSOLUTE signal: no meaningful whole-strand coverage AND no real
    # DN swing. norm_mean is self-scaled to 0..1, so its range alone can't tell signal from
    # noise — a flat noisy edge strand would otherwise masquerade as 'drying'.
    span_weak = (mean_abs_span is not None) and (mean_abs_span < min_contrast)
    if cov_rise < min_rise and (span_weak or mean_abs_span is None):
        return "none", None, 0.0, peak

    # temporal sharpness from the normalised mean-brightness trace
    lo_hits = np.where(nm >= 0.2)[0]
    hi_hits = np.where(nm >= 0.8)[0]
    if len(lo_hits) and len(hi_hits):
        lo_t = int(lo_hits[0])
        hi_t = int(hi_hits[hi_hits >= lo_t][0]) if (hi_hits >= lo_t).any() else int(hi_hits[0])
        rise_span = hi_t - lo_t
        steps = np.diff(nm[max(lo_t - 1, 0):hi_t + 1])
        big = float(steps.max()) if len(steps) else mean_rise
        onset = int(max(lo_t - 1, 0) + int(np.argmax(steps))) if len(steps) else lo_t
    else:
        rise_span, big, onset = T, 0.0, (int(lo_hits[0]) if len(lo_hits) else 0)
    onset = min(max(onset, 0), T - 1)

    is_whole = peak >= thresh
    is_sharp = rise_span <= step_window and big >= gradual_reject
    if is_whole and is_sharp:
        conf = float(0.5 * min(big / max(gradual_reject, 1e-6), 1.0)
                     + 0.5 * min(peak / max(thresh, 1e-6), 1.0))
        return "cavitation", onset, conf, peak
    # rose, but partial or gradual -> drying
    return "drying", None, float(0.3 * min(peak / max(thresh, 1e-6), 1.0)), peak


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def detect(final, tissue, stack=None, um_per_px=2.5,
           max_frames=100, ref_tail_n=8, bright_frac_thresh=0.7, step_window=2,
           gradual_reject=0.3, corridor_dilate_px=3, divergence_merge=True,
           clahe_blocksize=127, clahe_bins=256, clahe_maxslope=1.5,
           min_contrast=15.0, clahe_reference=True, ridge_thr=0.04,
           min_strand_px=20, **_ignored):
    """Registry entry point. See module docstring.

    ``final``/``tissue`` come from ``colour_shape.measure`` (dry mean-tail frame);
    this method builds its own late-tail reference from ``stack`` for tracing and
    reads brightness profiles from the *raw* stack for timing. Returns the
    standard masks plus ``tubes`` and ``events``.
    """
    from . import colour_shape as cs
    if stack is None:
        raise ValueError("rhizoid-events needs the frame stack (needs_stack=True)")
    stack = np.asarray(stack, np.float32)[: int(max_frames)]
    T = stack.shape[0]

    # tissue = the thallus, from the dry MEAN-tail final (``measure`` already computed
    # it). The max reference used for tracing biases Otsu toward the bright tubes, so it
    # must not define tissue. Fall back to the mean-tail final only if none was supplied.
    tissue = np.asarray(tissue, bool) if tissue is not None else cs.tissue_from_final(final)
    raw_ref = reference_frame(stack, ref_tail_n)
    if not tissue.any():
        return _empty_result(final.shape, T)

    tubes, ridge = trace_rhizoids(
        raw_ref, tissue, divergence_merge=divergence_merge,
        clahe_reference=clahe_reference, ridge_thr=ridge_thr,
        min_strand_px=min_strand_px, clahe_blocksize=clahe_blocksize,
        clahe_bins=clahe_bins, clahe_maxslope=clahe_maxslope)

    events = []
    onset = np.full(raw_ref.shape, -1, np.int32)
    shape = np.zeros(raw_ref.shape, bool)
    K = int(tubes.max())
    for tid in range(1, K + 1):
        strand = tubes == tid
        length_px = int(strand.sum())
        if length_px == 0:
            continue
        bf, nm, span = strand_signals(stack, strand, corridor_dilate_px=corridor_dilate_px,
                                      min_contrast=min_contrast)
        cls, cav_frame, conf, peak = classify_strand(
            bf, nm, mean_abs_span=span, thresh=bright_frac_thresh, step_window=step_window,
            gradual_reject=gradual_reject, min_contrast=min_contrast)
        events.append(dict(id=tid, cavitation_frame=cav_frame, cls=cls,
                           length_px=length_px, bright_frac_peak=round(peak, 4),
                           confidence=round(conf, 4)))

    # one physical tube may over-segment into several strands cavitating together;
    # fuse touching same-frame cavitations so the count reflects distinct events.
    tubes, events = merge_coincident_cavitations(tubes, events, step_window=step_window)
    for e in events:                             # rebuild onset/shape from merged ids
        if e["cls"] == "cavitation" and e["cavitation_frame"] is not None:
            strand = tubes == e["id"]
            onset[strand] = e["cavitation_frame"]
            shape |= strand

    colour = tissue & (raw_ref >= float(np.percentile(raw_ref[tissue], 70)))
    return dict(tissue=tissue, final=raw_ref, air=None, colour=colour, shape=shape,
                ridge=ridge, onset=onset, tubes=tubes, events=events)


def recompute_events(tubes, stack, params=None):
    """Re-derive per-rhizoid events from a (possibly edited) tube map + stack.

    Used by the Inspector correction loop: after the expert splits/merges/erases
    strands, the count and timing are recomputed from the edited identity map
    without re-tracing. Returns ``(events, onset, shape)``.
    """
    p = dict(params or {})
    max_frames = int(p.get("max_frames", 100))
    raw = np.asarray(stack, np.float32)[:max_frames]       # raw DN for timing, never CLAHE
    tubes = np.asarray(tubes, np.int32)
    events, onset, shape = [], np.full(tubes.shape, -1, np.int32), np.zeros(tubes.shape, bool)
    for tid in range(1, int(tubes.max()) + 1):
        strand = tubes == tid
        if not strand.any():
            continue
        mc = float(p.get("min_contrast", 15.0))
        bf, nm, span = strand_signals(raw, strand,
                                      corridor_dilate_px=int(p.get("corridor_dilate_px", 3)),
                                      min_contrast=mc)
        cls, cav_frame, conf, peak = classify_strand(
            bf, nm, mean_abs_span=span, thresh=float(p.get("bright_frac_thresh", 0.7)),
            step_window=int(p.get("step_window", 2)),
            gradual_reject=float(p.get("gradual_reject", 0.3)), min_contrast=mc)
        events.append(dict(id=tid, cavitation_frame=cav_frame, cls=cls,
                           length_px=int(strand.sum()),
                           bright_frac_peak=round(peak, 4), confidence=round(conf, 4)))
        if cls == "cavitation":
            onset[strand] = cav_frame
            shape |= strand
    return events, onset, shape


def merge_coincident_cavitations(tubes, events, step_window=2, dilate=5):
    """Fuse cavitation strands that touch AND cavitate on the same frame.

    Ridge over-segmentation can split one physical rhizoid into several strands;
    since a tube cavitates as a single whole-strand event, touching cavitation
    strands with onsets within ``step_window`` frames are the same event and must
    count once. Non-cavitation strands are untouched. Returns the relabelled
    ``tubes`` and the reduced ``events`` list (merged rows keep the earliest
    onset, max confidence/peak, summed length).
    """
    from scipy.ndimage import binary_dilation
    tubes = np.asarray(tubes, np.int32).copy()
    cav = [e for e in events if e["cls"] == "cavitation" and e["cavitation_frame"] is not None]
    if len(cav) <= 1:
        return tubes, events
    masks = {e["id"]: binary_dilation(tubes == e["id"], iterations=int(dilate)) for e in cav}
    parent = {e["id"]: e["id"] for e in cav}
    for i in range(len(cav)):
        for j in range(i + 1, len(cav)):
            a, b = cav[i], cav[j]
            if abs(a["cavitation_frame"] - b["cavitation_frame"]) <= step_window \
                    and (masks[a["id"]] & masks[b["id"]]).any():
                parent[_find(parent, a["id"])] = _find(parent, b["id"])
    groups = {}
    for e in cav:
        groups.setdefault(_find(parent, e["id"]), []).append(e)
    merged_ids = {e["id"] for g in groups.values() if len(g) > 1 for e in g}
    out_events = [e for e in events if e["id"] not in merged_ids]
    for root, g in groups.items():
        if len(g) == 1:
            continue
        keep = max(g, key=lambda z: z["length_px"])["id"]
        for e in g:
            if e["id"] != keep:
                tubes[tubes == e["id"]] = keep
        out_events.append(dict(
            id=keep, cavitation_frame=min(e["cavitation_frame"] for e in g),
            cls="cavitation", length_px=sum(e["length_px"] for e in g),
            bright_frac_peak=max(e["bright_frac_peak"] for e in g),
            confidence=max(e["confidence"] for e in g)))
    out_events.sort(key=lambda z: z["id"])
    return tubes, out_events


def count_curve(events, T):
    """Cumulative count of rhizoids cavitated vs frame (the staircase)."""
    counts = np.zeros(int(T), np.int64)
    for e in events:
        if e["cls"] == "cavitation" and e["cavitation_frame"] is not None:
            f = int(e["cavitation_frame"])
            if 0 <= f < T:
                counts[f] += 1
    return np.cumsum(counts)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #

def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros_like(v, dtype=float)


def _find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _relabel(seg):
    out = np.zeros_like(seg, np.int32)
    k = 0
    for v in np.unique(seg):
        if v == 0:
            continue
        k += 1
        out[seg == v] = k
    return out


def _empty_result(shape_hw, T):
    z = np.zeros(shape_hw, bool)
    return dict(tissue=z, final=np.zeros(shape_hw, np.float32), air=None, colour=z,
                shape=z, ridge=np.zeros(shape_hw, np.float32),
                onset=np.full(shape_hw, -1, np.int32),
                tubes=np.zeros(shape_hw, np.int32), events=[])
