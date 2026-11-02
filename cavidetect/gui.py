"""Simple desktop GUI for the `colour-shape` embolism measurement.

A thin Tkinter front-end over `cavidetect colour-shape`: choose an input folder
of per-frame TIFFs, choose where results go, tweak a few parameters, click Run,
watch live progress, then view the results. The pipeline runs as a SUBPROCESS
(not in-thread) so Cancel is instant and safe and the window never freezes.

The detection code is untouched; this module only builds the command line and
presents the outputs. Launch with `cavidetect-gui` or `python3 -m cavidetect.gui`.

tkinter is imported lazily inside `main()`/`App` so the pure helpers below (and
the module itself) import on machines without a Tk display, e.g. for unit tests.
"""
from __future__ import annotations
import csv
import os
import platform
import queue
import re
import subprocess
import sys
import threading
import webbrowser


# --------------------------------------------------------------------------- #
# Pure helpers (no tkinter) — unit-tested in tests/test_gui_helpers.py
# --------------------------------------------------------------------------- #

_SUMMARY_RE = re.compile(
    r"air=([\d.]+)\s+colour=([\d.]+)%\s+colour\+shape=([\d.]+)%"
)


def repo_root() -> str:
    """Repo root = the directory holding the `cavidetect` package (and pyproject)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def stem_of(folder: str) -> str:
    """Dataset name = the input folder's basename (e.g. 'cavi600005')."""
    return os.path.basename(os.path.normpath(folder))


def output_dir(save_dir: str, folder: str) -> str:
    """Per-dataset output dir: <save_dir>/<stem>, matching the CLI's `--out`."""
    return os.path.join(save_dir, stem_of(folder))


def build_command(folder: str, outdir: str, bin_xy: int = 2,
                  air: float | None = None, tail_frac: float = 0.12,
                  python_exe: str | None = None, frozen: bool | None = None) -> list[str]:
    """Build the argv for the colour-shape pipeline. `air=None` means auto, so the
    `--air` flag is omitted and the calibrated percentile is used.

    In a PyInstaller build (`frozen`) there is no `python -m`, so we invoke the one
    bundled executable with the subcommand directly; it dispatches to the CLI (see
    `cavidetect.app`). In dev we run `python -m cavidetect.cli`."""
    exe = python_exe or sys.executable
    if frozen is None:
        frozen = bool(getattr(sys, "frozen", False))
    head = [exe] if frozen else [exe, "-m", "cavidetect.cli"]
    cmd = head + ["colour-shape", folder, "--out", outdir,
                  "--bin", str(int(bin_xy)), "--tail-frac", str(tail_frac)]
    if air is not None:
        cmd += ["--air", str(air)]
    return cmd


def parse_summary_line(line: str) -> dict | None:
    """Pull air / colour% / colour+shape% from the pipeline's summary stdout line.
    Returns None for any other line."""
    m = _SUMMARY_RE.search(line)
    if not m:
        return None
    return {"air": float(m.group(1)),
            "colour_pct": float(m.group(2)),
            "shape_pct": float(m.group(3))}


def final_fraction(curve_csv_path: str) -> float | None:
    """Final cumulative fraction of tissue embolised = last row's
    `frac_tissue_embolised` in the `*_cs_curve.csv`. None if unreadable/empty."""
    try:
        with open(curve_csv_path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    except OSError:
        return None
    if not rows or "frac_tissue_embolised" not in rows[-1]:
        return None
    try:
        return float(rows[-1]["frac_tissue_embolised"])
    except (TypeError, ValueError):
        return None


def artefact_paths(outdir: str, stem: str) -> dict:
    """The files `colour-shape` writes, by convention `<stem>_cs_*`."""
    return {"viewer": os.path.join(outdir, f"{stem}_cs_viewer.html"),
            "curve": os.path.join(outdir, f"{stem}_cs_curve.csv"),
            "proof": os.path.join(outdir, f"{stem}_cs_proof.png"),
            "annotate": os.path.join(outdir, f"{stem}_cs_annotate.html")}


def build_bundle(out_dir: str, zip_path: str, extra_files=(), info_text: str = "") -> str:
    """Zip a dataset's result folder (+ any attached annotation PNGs + an info note) into
    `zip_path`, for sending back when a detection needs working through. Returns zip_path."""
    import zipfile
    stem = os.path.basename(os.path.normpath(out_dir))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _dirs, files in os.walk(out_dir):
            for fn in files:
                fp = os.path.join(root, fn)
                z.write(fp, os.path.join(stem, os.path.relpath(fp, out_dir)))
        for fp in extra_files:
            if fp and os.path.exists(fp):
                z.write(fp, os.path.join(stem, "annotations", os.path.basename(fp)))
        if info_text:
            z.writestr(os.path.join(stem, "info.txt"), info_text)
    return zip_path


def reveal(path: str) -> None:
    """Open a file/folder in the OS file manager (best-effort, cross-platform)."""
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.Popen(["open", path])
        elif system == "Windows":
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Tkinter app
# --------------------------------------------------------------------------- #

def main() -> None:
    import tkinter as tk
    from tkinter import ttk, filedialog

    class App:
        def __init__(self, root: "tk.Tk") -> None:
            self.root = root
            self.proc: subprocess.Popen | None = None
            self.q: "queue.Queue[str]" = queue.Queue()
            self.summary: dict | None = None
            self.out_dir = ""
            self.stem = ""
            self._proof_img = None  # keep a reference so Tk doesn't GC the image
            self._proof_path = ""
            self._run_params: dict = {}

            root.title("cavidetect — embolism (colour-shape)")
            root.minsize(1000, 700)
            root.geometry("1040x820")
            pad = {"padx": 8, "pady": 4}

            frm = ttk.Frame(root, padding=10)
            frm.pack(fill="both", expand=True)
            frm.columnconfigure(1, weight=1)

            # --- input folder ---
            ttk.Label(frm, text="Input folder (frames):").grid(row=0, column=0, sticky="w", **pad)
            self.in_var = tk.StringVar()
            ttk.Entry(frm, textvariable=self.in_var).grid(row=0, column=1, sticky="ew", **pad)
            ttk.Button(frm, text="Choose…", command=self._pick_input).grid(row=0, column=2, **pad)

            # --- save location ---
            ttk.Label(frm, text="Save results to:").grid(row=1, column=0, sticky="w", **pad)
            self.out_var = tk.StringVar(value=os.path.join(repo_root(), "results_whole"))
            ttk.Entry(frm, textvariable=self.out_var).grid(row=1, column=1, sticky="ew", **pad)
            ttk.Button(frm, text="Choose…", command=self._pick_output).grid(row=1, column=2, **pad)

            # --- advanced parameters ---
            adv = ttk.Labelframe(frm, text="Parameters", padding=8)
            adv.grid(row=2, column=0, columnspan=3, sticky="ew", **pad)
            ttk.Label(adv, text="Bin (downsample):").grid(row=0, column=0, sticky="w", padx=4)
            self.bin_var = tk.IntVar(value=2)
            ttk.Spinbox(adv, from_=1, to=8, width=5, textvariable=self.bin_var).grid(row=0, column=1, sticky="w", padx=4)

            self.auto_air = tk.BooleanVar(value=True)
            ttk.Checkbutton(adv, text="Auto air threshold", variable=self.auto_air,
                            command=self._toggle_air).grid(row=0, column=2, sticky="w", padx=12)
            self.air_var = tk.StringVar()
            self.air_entry = ttk.Entry(adv, width=8, textvariable=self.air_var, state="disabled")
            self.air_entry.grid(row=0, column=3, sticky="w", padx=4)

            ttk.Label(adv, text="Tail fraction:").grid(row=0, column=4, sticky="w", padx=12)
            self.tail_var = tk.StringVar(value="0.12")
            ttk.Entry(adv, width=6, textvariable=self.tail_var).grid(row=0, column=5, sticky="w", padx=4)

            # --- run / cancel ---
            runrow = ttk.Frame(frm)
            runrow.grid(row=3, column=0, columnspan=3, pady=8)
            self.run_btn = ttk.Button(runrow, text="▶  Run", command=self._run)
            self.run_btn.pack(side="left")
            self.auto_view = tk.BooleanVar(value=True)
            ttk.Checkbutton(runrow, text="Open frame viewer when finished",
                            variable=self.auto_view).pack(side="left", padx=14)

            # --- progress log ---
            ttk.Label(frm, text="Progress:").grid(row=4, column=0, sticky="w", padx=8)
            logwrap = ttk.Frame(frm)
            logwrap.grid(row=5, column=0, columnspan=3, sticky="nsew", padx=8, pady=4)
            frm.rowconfigure(5, weight=1)
            logwrap.columnconfigure(0, weight=1)
            logwrap.rowconfigure(0, weight=1)
            self.log = tk.Text(logwrap, height=10, wrap="word", state="disabled",
                               background="#111417", foreground="#d7dde3")
            self.log.grid(row=0, column=0, sticky="nsew")
            sb = ttk.Scrollbar(logwrap, command=self.log.yview)
            sb.grid(row=0, column=1, sticky="ns")
            self.log.configure(yscrollcommand=sb.set)

            # --- results summary ---
            self.res = ttk.Labelframe(frm, text="Results", padding=8)
            self.res.grid(row=6, column=0, columnspan=3, sticky="ew", **pad)
            self.headline = ttk.Label(self.res, text="— run a dataset to see results —",
                                      font=("TkDefaultFont", 13, "bold"))
            self.headline.grid(row=0, column=0, columnspan=3, sticky="w")
            self.detail = ttk.Label(self.res, text="")
            self.detail.grid(row=1, column=0, columnspan=3, sticky="w", pady=(2, 6))
            self.proof_label = ttk.Label(self.res, cursor="hand2")
            self.proof_label.grid(row=2, column=0, columnspan=3, sticky="w")
            self.proof_label.bind("<Button-1>", lambda e: self._open_proof())
            self.proof_hint = ttk.Label(self.res, foreground="#7a8690", text="")
            self.proof_hint.grid(row=3, column=0, columnspan=3, sticky="w")
            self.view_btn = ttk.Button(self.res, text="▶ Step through frames (viewer)",
                                       command=self._open_viewer, state="disabled")
            self.view_btn.grid(row=4, column=0, sticky="w", pady=6)
            self.annot_btn = ttk.Button(self.res, text="✎ Annotate this dataset",
                                        command=self._open_annotator, state="disabled")
            self.annot_btn.grid(row=4, column=1, sticky="w", pady=6, padx=6)
            self.folder_btn = ttk.Button(self.res, text="Open results folder",
                                         command=self._open_folder, state="disabled")
            self.folder_btn.grid(row=4, column=2, sticky="w", pady=6)
            self.bundle_btn = ttk.Button(self.res, text="📦 Bundle to send…",
                                         command=self._bundle, state="disabled")
            self.bundle_btn.grid(row=5, column=0, sticky="w", pady=(0, 2))

        # -- parameter helpers --
        def _toggle_air(self) -> None:
            self.air_entry.configure(state="disabled" if self.auto_air.get() else "normal")

        def _pick_input(self) -> None:
            d = filedialog.askdirectory(title="Choose the folder of frame TIFFs")
            if d:
                self.in_var.set(d)

        def _pick_output(self) -> None:
            d = filedialog.askdirectory(title="Choose where to save results")
            if d:
                self.out_var.set(d)

        # -- logging --
        def _append(self, text: str) -> None:
            self.log.configure(state="normal")
            self.log.insert("end", text)
            self.log.see("end")
            self.log.configure(state="disabled")

        # -- run / cancel --
        def _run(self) -> None:
            if self.proc is not None:   # button is acting as Cancel
                self._cancel()
                return
            folder = self.in_var.get().strip()
            save = self.out_var.get().strip()
            if not folder or not os.path.isdir(folder):
                self._append("! Choose a valid input folder first.\n")
                return
            if not save:
                self._append("! Choose a save location first.\n")
                return
            try:
                air = None if self.auto_air.get() else float(self.air_var.get())
                tail = float(self.tail_var.get())
                bin_xy = int(self.bin_var.get())
            except ValueError:
                self._append("! Bin / air / tail fraction must be numbers.\n")
                return

            self.stem = stem_of(folder)
            self.out_dir = output_dir(save, folder)
            os.makedirs(self.out_dir, exist_ok=True)
            self._run_params = {"input": folder, "bin": bin_xy,
                                "air": "auto" if air is None else air, "tail_frac": tail}
            cmd = build_command(folder, self.out_dir, bin_xy=bin_xy, air=air, tail_frac=tail)

            # reset UI
            self.log.configure(state="normal"); self.log.delete("1.0", "end"); self.log.configure(state="disabled")
            self.summary = None
            self._reset_results()
            self._append("$ " + " ".join(cmd) + "\n\n")
            self.run_btn.configure(text="■  Cancel")

            cwd = None if getattr(sys, "frozen", False) else repo_root()
            self.proc = subprocess.Popen(
                cmd, cwd=cwd, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
                encoding="utf-8", errors="replace")
            threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()
            self.root.after(100, self._drain)

        def _cancel(self) -> None:
            if self.proc is not None:
                self.proc.terminate()
                self._append("\n■ Cancelled.\n")

        def _reader(self, proc: subprocess.Popen) -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                self.q.put(line)
            proc.wait()
            self.q.put(f"\x00EXIT{proc.returncode}")

        def _drain(self) -> None:
            try:
                while True:
                    line = self.q.get_nowait()
                    if line.startswith("\x00EXIT"):
                        self._finish(int(line[5:]))
                        return
                    self._append(line)
                    s = parse_summary_line(line)
                    if s:
                        self.summary = s
            except queue.Empty:
                pass
            if self.proc is not None:
                self.root.after(100, self._drain)

        def _finish(self, code: int) -> None:
            self.proc = None
            self.run_btn.configure(text="▶  Run")
            if code != 0:
                self._append(f"\n! Finished with error (exit {code}). See log above.\n")
                return
            self._append("\n✓ Done.\n")
            self._show_results()
            if self.auto_view.get():
                self._open_viewer()

        # -- results panel --
        def _reset_results(self) -> None:
            self.headline.configure(text="…")
            self.detail.configure(text="")
            self.proof_label.configure(image="")
            self.proof_hint.configure(text="")
            self._proof_img = None
            self._proof_path = ""
            self.view_btn.configure(state="disabled")
            self.annot_btn.configure(state="disabled")
            self.folder_btn.configure(state="disabled")
            self.bundle_btn.configure(state="disabled")

        def _show_results(self) -> None:
            paths = artefact_paths(self.out_dir, self.stem)
            self._proof_path = paths["proof"]
            frac = final_fraction(paths["curve"])
            if frac is not None:
                self.headline.configure(text=f"{frac * 100:.1f}%  of tissue embolised")
            else:
                self.headline.configure(text="done (curve not found)")
            if self.summary:
                self.detail.configure(
                    text=(f"air threshold = {self.summary['air']:.0f}    "
                          f"colour = {self.summary['colour_pct']:.1f}%    "
                          f"colour+shape = {self.summary['shape_pct']:.1f}%"))
            # proof thumbnail, scaled to fit the window (native PNG in Tk 8.6).
            # It's only a preview — click it to open the full-resolution PNG.
            if os.path.exists(paths["proof"]):
                try:
                    img = tk.PhotoImage(file=paths["proof"])
                    self.root.update_idletasks()
                    target = max(900, self.res.winfo_width() - 24)
                    k = max(1, -(-img.width() // target))   # ceil division
                    if k > 1:
                        img = img.subsample(k, k)
                    self._proof_img = img
                    self.proof_label.configure(image=img)
                    self.proof_hint.configure(
                        text="↑ click the image to open it full-size   ·   "
                             "use “Step through frames” to check the timing")
                except Exception:
                    self.proof_label.configure(image="")
                    self.proof_hint.configure(text=f"proof PNG saved: {paths['proof']}")
            self.view_btn.configure(state="normal" if os.path.exists(paths["viewer"]) else "disabled")
            self.annot_btn.configure(state="normal" if os.path.exists(paths["annotate"]) else "disabled")
            self.folder_btn.configure(state="normal")
            self.bundle_btn.configure(state="normal")

        def _open_proof(self) -> None:
            if getattr(self, "_proof_path", "") and os.path.exists(self._proof_path):
                reveal(self._proof_path)

        def _open_annotator(self) -> None:
            a = artefact_paths(self.out_dir, self.stem)["annotate"]
            if os.path.exists(a):
                webbrowser.open("file://" + os.path.abspath(a))

        def _bundle(self) -> None:
            if not self.out_dir or not os.path.isdir(self.out_dir):
                return
            import datetime
            default = f"{self.stem}_report_{datetime.date.today():%Y%m%d}.zip"
            zip_path = filedialog.asksaveasfilename(
                title="Save bundle to send", defaultextension=".zip",
                initialfile=default, filetypes=[("Zip archive", "*.zip")])
            if not zip_path:
                return
            extra = filedialog.askopenfilenames(
                title="Attach your annotation PNG(s) — Cancel to skip",
                filetypes=[("PNG image", "*.png"), ("All files", "*.*")]) or ()
            info = self._info_text(extra)
            try:
                build_bundle(self.out_dir, zip_path, extra_files=extra, info_text=info)
            except OSError as e:
                self._append(f"! Could not write bundle: {e}\n")
                return
            self._append(f"📦 Bundle written: {zip_path}\n")
            reveal(os.path.dirname(zip_path) or ".")

        def _info_text(self, extra) -> str:
            lines = [f"dataset: {self.stem}",
                     f"input: {self._run_params.get('input', '')}",
                     f"bin: {self._run_params.get('bin', '')}",
                     f"air: {self._run_params.get('air', '')}",
                     f"tail_frac: {self._run_params.get('tail_frac', '')}"]
            frac = final_fraction(artefact_paths(self.out_dir, self.stem)["curve"])
            if frac is not None:
                lines.append(f"final % tissue embolised: {frac * 100:.1f}")
            if self.summary:
                lines.append(f"air threshold used: {self.summary['air']:.0f}")
            if extra:
                lines.append("attached annotations: " + ", ".join(os.path.basename(x) for x in extra))
            try:
                from . import pipeline as _pl
                lines.append(f"cavidetect version: {_pl.__version__}")
            except Exception:
                pass
            return "\n".join(lines) + "\n"

        def _open_viewer(self) -> None:
            v = artefact_paths(self.out_dir, self.stem)["viewer"]
            if os.path.exists(v):
                webbrowser.open("file://" + os.path.abspath(v))

        def _open_folder(self) -> None:
            if self.out_dir:
                reveal(self.out_dir)

    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
