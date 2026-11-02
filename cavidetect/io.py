"""Stack IO. Reads raw read-only, resolves ImageJ z-as-t, reads pixel size from metadata.

Two raw layouts are supported, both yielding a (T,Y,X) float32 stack + StackMeta:
  - a single multi-page TIFF (old crop stacks, z-as-time) -> load_stack
  - a folder of one single-page TIFF per timepoint (new whole-image sequences) -> load_frames
Use load() to dispatch on path type."""
from __future__ import annotations
from dataclasses import dataclass, field
import glob, os, re, warnings
import numpy as np
import tifffile


@dataclass
class StackMeta:
    path: str
    shape: tuple          # (T, Y, X)
    um_per_px: float
    scale_from_metadata: bool
    n_frames: int
    notes: tuple = ()     # ingest warnings (dropped duplicates, frame gaps) for the manifest


def read_um_per_px(tif: tifffile.TiffFile, fallback: float) -> tuple[float, bool]:
    """ImageJ stores resolution as XResolution = px per unit; unit in imagej_metadata."""
    try:
        p0 = tif.pages[0]
        xres = p0.tags.get("XResolution")
        if xres is not None:
            num, den = xres.value  # px per unit
            if num:
                px_per_unit = num / den
                return (1.0 / px_per_unit, True)
    except Exception:
        pass
    return (fallback, False)


def load_stack(path: str, fallback_um_per_px: float = 2.5) -> tuple[np.ndarray, StackMeta]:
    """Return (stack float32 [T,Y,X], StackMeta). Treats axis 0 as time (ImageJ slices)."""
    with tifffile.TiffFile(path) as tif:
        arr = tif.asarray()
        um, from_meta = read_um_per_px(tif, fallback_um_per_px)
    if arr.ndim != 3:
        raise ValueError(f"expected 3D (T,Y,X), got shape {arr.shape} for {path}")
    arr = arr.astype(np.float32, copy=False)
    meta = StackMeta(path=path, shape=tuple(arr.shape), um_per_px=um,
                     scale_from_metadata=from_meta, n_frames=arr.shape[0])
    return arr, meta


_DIGITS = re.compile(r"(\d+)")
_CLEAN_STEM = re.compile(r"^[A-Za-z]*[_-]?\d+$")  # e.g. "t_0001"; "t_0001 - Copy" fails this


def _frame_index(stem: str):
    """Numeric frame index = the last run of digits in the stem (None if there is none)."""
    m = list(_DIGITS.finditer(stem))
    return int(m[-1].group(1)) if m else None


def load_frames(folder: str, fallback_um_per_px: float = 2.5) -> tuple[np.ndarray, StackMeta]:
    """Load a folder of single-page TIFF frames as a (T,Y,X) float32 stack.

    Frames are ordered by the numeric index in their filename. When several files map to
    the same index (e.g. a "t_0001 - Copy.tif"), the clean `t_<n>.tif` name wins and the
    rest are dropped with a warning. RGB frames are collapsed to grey. The per-frame files
    carry no trustworthy scale (just a generic dpi tag), so um_per_px is always the fallback.
    """
    paths = sorted(glob.glob(os.path.join(folder, "*.tif")) +
                   glob.glob(os.path.join(folder, "*.tiff")))
    if not paths:
        raise ValueError(f"no .tif/.tiff frames found in folder {folder}")

    by_idx: dict[int, list[str]] = {}
    for p in paths:
        idx = _frame_index(os.path.splitext(os.path.basename(p))[0])
        if idx is not None:
            by_idx.setdefault(idx, []).append(p)
    if not by_idx:
        raise ValueError(f"no frame-numbered .tif files in folder {folder}")

    def _rank(p: str):
        stem = os.path.splitext(os.path.basename(p))[0]
        return (0 if _CLEAN_STEM.match(stem) else 1, len(stem), p)

    chosen: dict[int, str] = {}
    dropped: list[str] = []
    for idx, cands in by_idx.items():
        ranked = sorted(cands, key=_rank)
        chosen[idx] = ranked[0]
        dropped.extend(os.path.basename(x) for x in ranked[1:])

    order = sorted(chosen)
    notes: list[str] = []
    if dropped:
        notes.append(f"dropped {len(dropped)} duplicate frame file(s): {', '.join(sorted(dropped))}")
        warnings.warn(notes[-1])
    gaps = [i for i in range(order[0], order[-1] + 1) if i not in chosen]
    if gaps:
        notes.append(f"non-contiguous frames: {len(gaps)} missing index(es) between {order[0]} and {order[-1]}")
        warnings.warn(notes[-1])

    frames = []
    for i in order:
        a = tifffile.imread(chosen[i])
        if a.ndim == 3 and a.shape[-1] in (3, 4):
            a = a[..., :3].mean(axis=-1)
        elif a.ndim != 2:
            raise ValueError(f"frame {chosen[i]} has unexpected shape {a.shape}")
        frames.append(a)
    stack = np.stack(frames).astype(np.float32, copy=False)
    meta = StackMeta(path=folder, shape=tuple(stack.shape), um_per_px=fallback_um_per_px,
                     scale_from_metadata=False, n_frames=stack.shape[0], notes=tuple(notes))
    return stack, meta


def load(path: str, fallback_um_per_px: float = 2.5) -> tuple[np.ndarray, StackMeta]:
    """Dispatch: a directory is a folder of frames; a file is a multi-page stack."""
    if os.path.isdir(path):
        return load_frames(path, fallback_um_per_px)
    return load_stack(path, fallback_um_per_px)
