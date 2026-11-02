"""CLI: `cavidetect run STACK.tif --out results/` and `cavidetect batch FOLDER/ --out results/`.

`run` accepts either a multi-page TIFF or a folder of single-page frames (one per timepoint).
`batch` treats each immediate subfolder of single-page frames as ONE sequence, and still
walks the rest for multi-page crop stacks."""
from __future__ import annotations
import argparse, glob, os, sys
import tifffile
from .config import Config
from .pipeline import run_stack, __version__


def _is_frame_folder(d: str) -> bool:
    """A directory holding >=2 single-page TIFFs is a per-frame sequence, not a crop stack."""
    tifs = sorted(glob.glob(os.path.join(d, "*.tif")) + glob.glob(os.path.join(d, "*.tiff")))
    if len(tifs) < 2:
        return False
    try:
        with tifffile.TiffFile(tifs[0]) as t:
            return len(t.pages) == 1
    except Exception:
        return False


def main(argv=None):
    p = argparse.ArgumentParser(prog="cavidetect", description="Detect cavitation events in rhizoid time-lapses.")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="process one stack")
    pr.add_argument("stack"); pr.add_argument("--out", required=True); pr.add_argument("--config", default=None)
    pr.add_argument("--no-overlay", action="store_true")
    pr.add_argument("--quick", action="store_true", help="only the primary embolism measurement; skip the slow tube/event detection")

    pb = sub.add_parser("batch", help="process every .tif under a folder (recursive)")
    pb.add_argument("folder"); pb.add_argument("--out", required=True); pb.add_argument("--config", default=None)
    pb.add_argument("--no-overlay", action="store_true")
    pb.add_argument("--quick", action="store_true", help="only the primary embolism measurement; skip the slow tube/event detection")

    pc = sub.add_parser("colour-shape", help="colour+shape embolism (calibrated); folder of frames -> viewer+curve")
    pc.add_argument("folder"); pc.add_argument("--out", required=True)
    pc.add_argument("--bin", type=int, default=2)
    pc.add_argument("--air", type=float, default=None, help="manual air-colour threshold (else calibrated percentile)")
    pc.add_argument("--tail-frac", type=float, default=0.12)

    a = p.parse_args(argv)
    cfg = Config.load(getattr(a, "config", None))
    if getattr(a, "quick", False):
        cfg.skip_events = True

    if a.cmd == "colour-shape":
        from .colour_shape import run as run_cs
        run_cs(a.folder, a.out, bin_xy=a.bin, air=a.air, tail_frac=a.tail_frac)
        print(f"-> {a.out}/{os.path.basename(os.path.normpath(a.folder))}_cs_viewer.html")
        return

    if a.cmd == "run":
        res = run_stack(a.stack, a.out, cfg, make_overlay=not a.no_overlay)
        m = res["manifest"]
        print(f"{os.path.basename(os.path.normpath(a.stack))}: "
              f"{m.get('frac_embolised_final', 0)*100:.1f}% tissue embolised, "
              f"{m['n_events']} clean events -> {a.out}")
    elif a.cmd == "batch":
        # frame-folders: each immediate subdir of single-page frames is ONE sequence
        frame_dirs = [os.path.join(a.folder, e) for e in sorted(os.listdir(a.folder))
                      if os.path.isdir(os.path.join(a.folder, e))
                      and _is_frame_folder(os.path.join(a.folder, e))]
        # multi-page crop stacks: every .tif NOT inside one of those frame-folders
        skip = tuple(os.path.abspath(d) + os.sep for d in frame_dirs)
        files = [f for f in sorted(glob.glob(os.path.join(a.folder, "**", "*.tif"), recursive=True))
                 if not os.path.abspath(f).startswith(skip)]
        if not frame_dirs and not files:
            print(f"no .tif under {a.folder}", file=sys.stderr); sys.exit(1)
        for d in frame_dirs:
            stem = os.path.basename(os.path.normpath(d))
            od = os.path.join(a.out, stem)
            res = run_stack(d, od, cfg, make_overlay=not a.no_overlay)
            print(f"{stem}: {res['manifest']['n_events']} events ({res['meta'].n_frames} frames)")
        for f in files:
            stem = os.path.relpath(f, a.folder).replace(os.sep, "__").rsplit(".", 1)[0]
            od = os.path.join(a.out, stem)
            res = run_stack(f, od, cfg, make_overlay=not a.no_overlay)
            print(f"{stem}: {res['manifest']['n_events']} events")


if __name__ == "__main__":
    main()
