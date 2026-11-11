"""CLI: `cavidetect run STACK.tif --out results/` and `cavidetect batch FOLDER/ --out results/`.

`run` accepts either a multi-page TIFF or a folder of single-page frames (one per timepoint).
`batch` treats each immediate subfolder of single-page frames as ONE sequence, and still
walks the rest for multi-page crop stacks."""
from __future__ import annotations
import argparse, glob, json, os, sys
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
    pc.add_argument("--method", default="colour-frangi", help="detection method (see --list-methods)")
    pc.add_argument("--params", default=None, help="JSON file of method parameters")
    pc.add_argument("--list-methods", action="store_true", help="print the available methods and exit")
    pc.add_argument("--frame-interval", type=float, default=15.0,
                    help="seconds between frames (for the onset-time column; default 15)")
    pc.add_argument("--no-tiffs", action="store_true", help="skip the per-frame/cumulative binary TIFFs")
    pc.add_argument("--no-diagnostics", action="store_true", help="skip the diagnostic stage montage")

    pia = sub.add_parser("inspect-apply",
                         help="apply inspector corrections and recompute (Phase 2)")
    pia.add_argument("--run", required=True, help="results dir holding <stem>_inspect_state.npz")
    pia.add_argument("--stem", required=True, help="artefact stem, e.g. cavi490001_cs")
    pia.add_argument("--edits", required=True, help="corrections JSON exported from the inspector")
    pia.add_argument("--source", required=True, help="the input frame folder (for added traces + panels)")
    pia.add_argument("--bin", type=int, default=None, help="binning (default: from the saved state)")

    a = p.parse_args(argv)
    cfg = Config.load(getattr(a, "config", None))
    if getattr(a, "quick", False):
        cfg.skip_events = True

    if a.cmd == "colour-shape":
        from . import methods as M
        from .colour_shape import run as run_cs
        if a.list_methods:
            for name, label in M.list_methods():
                print(f"{name} — {label}")
            sys.exit(0)
        if a.method not in M.REGISTRY:
            print(f"unknown method {a.method!r}; valid methods: {', '.join(M.REGISTRY)}",
                  file=sys.stderr)
            sys.exit(2)
        params: dict = {}
        if a.params:
            try:
                with open(a.params, encoding="utf-8") as fh:
                    params = json.load(fh)
                if not isinstance(params, dict):
                    raise ValueError("params JSON must be an object of key:value")
            except (OSError, ValueError) as ex:
                print(f"could not read --params {a.params!r}: {ex}", file=sys.stderr)
                sys.exit(2)
        run_cs(a.folder, a.out, bin_xy=a.bin, air=a.air, tail_frac=a.tail_frac,
               method=a.method, params=params, frame_interval_s=a.frame_interval,
               write_tiffs=not a.no_tiffs, write_diagnostics=not a.no_diagnostics)
        print(f"-> {a.out}/{os.path.basename(os.path.normpath(a.folder))}_cs_viewer.html")
        return

    if a.cmd == "inspect-apply":
        from .inspector import apply_corrections
        out = apply_corrections(a.run, a.stem, a.edits, a.source, bin_xy=a.bin)
        print(f"corrected: {out['embolised_px']} px embolised, "
              f"{out['frac_final']*100:.1f}% of tissue -> {out['inspector']}")
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
