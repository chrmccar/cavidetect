"""Diagnostic export (v0.2.0 brief §1): make the detector's decisions inspectable.

The expert's chief complaint is poor recall in crowded bundles, with no way to see
*where* the method drops events. These helpers dump every intermediate stage of a
``colour_shape.measure`` result as a single montage, and compare the COLOUR-only mask
against the COLOUR+SHAPE mask so false negatives introduced by the shape gate are
visible. Nothing here changes a measurement; it only renders what the method computed.
"""
from __future__ import annotations
import os
import numpy as np


def stage_images(result: dict) -> "list[tuple[str, np.ndarray, bool]]":
    """(title, image, is_mask) for every available stage, in pipeline order.

    Reads the ``stages`` dict a ``detect`` method attaches (brightness, frangi, seed,
    pre/post shape, scale, accepted) plus the always-present ``tissue``/``final``/``shape``.
    Missing stages are simply skipped, so this works for any method.
    """
    final = result["final"]
    stages = result.get("stages") or {}
    out: list[tuple[str, np.ndarray, bool]] = [
        ("raw final", final, False),
        ("tissue", result["tissue"], True),
    ]
    order = [("brightness", "brightness-only"), ("frangi", "Frangi tubeness"),
             ("seed", "ridge seeds"), ("pre_shape", "pre shape-filter"),
             ("scale", "scale mask"), ("post_shape", "post shape-filter")]
    for key, title in order:
        if key in stages:
            img = stages[key]
            out.append((title, img, img.dtype == bool))
    out.append(("COLOUR only", result["colour"], True))
    out.append(("COLOUR + SHAPE (accepted)", result["shape"], True))
    return out


def save_stage_montage(result: dict, out_dir: str, stem: str, dpi: int = 95) -> str:
    """One PNG laying out every pipeline stage over the raw final frame.

    Masks are drawn as a red overlay on the grey final so position is obvious; scalar
    maps (Frangi) are shown with a heat colourmap. Use this to see which stage removed a
    rhizoid you expected to keep (brightness missed it vs. the shape gate dropped it)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    final = result["final"]
    vmin, vmax = np.percentile(final, [2, 99])
    panels = stage_images(result)
    n = len(panels)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(4.4 * cols, 4.2 * rows))
    for i, (title, img, is_mask) in enumerate(panels):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(final, cmap="gray", vmin=vmin, vmax=vmax)
        if is_mask:
            ov = np.zeros((*img.shape, 4))
            ov[img.astype(bool)] = [1, 0.15, 0.15, 0.6]
            ax.imshow(ov)
        elif title.startswith("Frangi"):
            ax.imshow(np.where(img > 0, img, np.nan), cmap="magma", alpha=0.85)
        else:  # raw final already drawn
            pass
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.suptitle(f"{stem} — diagnostic stages (method: {result.get('method','?')})",
                 x=0.01, ha="left", fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, f"{stem}_cs_stages.png")
    plt.savefig(path, dpi=dpi)
    plt.close()
    return path
