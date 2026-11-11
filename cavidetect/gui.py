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
                  python_exe: str | None = None, frozen: bool | None = None,
                  method: str = "colour-frangi",
                  params_path: str | None = None) -> list[str]:
    """Build the argv for the colour-shape pipeline. `air=None` means auto, so the
    `--air` flag is omitted and the calibrated percentile is used. `method` selects
    the detection method; `params_path` (if given) is a JSON file of its parameters.

    In a PyInstaller build (`frozen`) there is no `python -m`, so we invoke the one
    bundled executable with the subcommand directly; it dispatches to the CLI (see
    `cavidetect.app`). In dev we run `python -m cavidetect.cli`."""
    exe = python_exe or sys.executable
    if frozen is None:
        frozen = bool(getattr(sys, "frozen", False))
    head = [exe] if frozen else [exe, "-m", "cavidetect.cli"]
    cmd = head + ["colour-shape", folder, "--out", outdir,
                  "--bin", str(int(bin_xy)), "--tail-frac", str(tail_frac),
                  "--method", method]
    if params_path is not None:
        cmd += ["--params", params_path]
    if air is not None:
        cmd += ["--air", str(air)]
    return cmd


def params_from_fields(method_name: str, values: dict) -> dict:
    """Coerce raw GUI field values through the method's schema (types, ranges,
    defaults; unknown keys dropped)."""
    from . import methods as M
    return M.resolve_params(method_name, values)


def write_params_tempfile(params: dict) -> str:
    """Write `params` as JSON to a temp file and return its path (the GUI passes
    it to the pipeline via `--params`)."""
    import json
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".json", prefix="cavidetect_params_")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(params, fh)
    return path


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
            "annotate": os.path.join(outdir, f"{stem}_cs_annotate.html"),
            "inspect": os.path.join(outdir, f"{stem}_cs_inspect.html")}


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

            # --- method selector ---
            from . import methods as _methods
            self._tk = tk
            self._ttk = ttk
            self._methods_list = _methods.list_methods()
            self._name_to_label = {n: lbl for n, lbl in self._methods_list}
            self._label_to_name = {lbl: n for n, lbl in self._methods_list}
            mrow = ttk.Frame(frm)
            mrow.grid(row=2, column=0, columnspan=3, sticky="ew", **pad)
            ttk.Label(mrow, text="Method:").pack(side="left")
            first = self._methods_list[0][0]
            self.method_var = tk.StringVar(value=first)
            self.method_display = tk.StringVar(value=self._name_to_label[first])
            combo = ttk.Combobox(mrow, state="readonly", width=52,
                                 values=[lbl for _n, lbl in self._methods_list],
                                 textvariable=self.method_display)
            combo.pack(side="left", padx=8)
            combo.bind("<<ComboboxSelected>>", lambda e: self._on_method_change())

            # --- general run controls ---
            grow = ttk.Frame(frm)
            grow.grid(row=3, column=0, columnspan=3, sticky="ew", **pad)
            ttk.Label(grow, text="Bin (downsample):").pack(side="left")
            self.bin_var = tk.IntVar(value=2)
            ttk.Spinbox(grow, from_=1, to=8, width=5,
                        textvariable=self.bin_var).pack(side="left", padx=(4, 18))
            ttk.Label(grow, text="Tail fraction:").pack(side="left")
            self.tail_var = tk.StringVar(value="0.12")
            ttk.Entry(grow, width=6, textvariable=self.tail_var).pack(side="left", padx=4)

            # --- schema-driven method settings (rebuilt on method change) ---
            self.params_frame = ttk.Labelframe(frm, text="Settings", padding=8)
            self.params_frame.grid(row=4, column=0, columnspan=3, sticky="ew", **pad)
            self.param_vars: dict = {}
            self._build_params_panel()

            # --- run / cancel ---
            runrow = ttk.Frame(frm)
            runrow.grid(row=5, column=0, columnspan=3, pady=8)
            self.run_btn = ttk.Button(runrow, text="▶  Run", command=self._run)
            self.run_btn.pack(side="left")
            self.auto_view = tk.BooleanVar(value=True)
            ttk.Checkbutton(runrow, text="Open frame viewer when finished",
                            variable=self.auto_view).pack(side="left", padx=14)

            # --- progress log ---
            ttk.Label(frm, text="Progress:").grid(row=6, column=0, sticky="w", padx=8)
            logwrap = ttk.Frame(frm)
            logwrap.grid(row=7, column=0, columnspan=3, sticky="nsew", padx=8, pady=4)
            frm.rowconfigure(7, weight=1)
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
            self.res.grid(row=8, column=0, columnspan=3, sticky="ew", **pad)
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
            self.inspect_btn = ttk.Button(self.res, text="🔍 Inspect (where / which tube / when)",
                                          command=self._open_inspector, state="disabled")
            self.inspect_btn.grid(row=4, column=1, sticky="w", pady=6, padx=6)
            self.annot_btn = ttk.Button(self.res, text="✎ Annotate this dataset",
                                        command=self._open_annotator, state="disabled")
            self.annot_btn.grid(row=4, column=2, sticky="w", pady=6, padx=6)
            self.folder_btn = ttk.Button(self.res, text="Open results folder",
                                         command=self._open_folder, state="disabled")
            self.folder_btn.grid(row=5, column=1, sticky="w", pady=6)
            self.bundle_btn = ttk.Button(self.res, text="📦 Bundle to send…",
                                         command=self._bundle, state="disabled")
            self.bundle_btn.grid(row=5, column=0, sticky="w", pady=(0, 2))
            self.apply_btn = ttk.Button(self.res, text="✓ Apply corrections…",
                                        command=self._apply_corrections, state="disabled")
            self.apply_btn.grid(row=5, column=2, sticky="w", pady=(0, 2))

        # -- method + schema-driven settings --
        def _on_method_change(self) -> None:
            self.method_var.set(self._label_to_name[self.method_display.get()])
            self._build_params_panel()

        def _build_params_panel(self) -> None:
            """(Re)build the settings widgets from the current method's schema,
            grouped by ParamSpec.group. One Tk variable per parameter key."""
            from . import methods as _methods
            tk, ttk = self._tk, self._ttk
            for child in self.params_frame.winfo_children():
                child.destroy()
            self.param_vars = {}
            schema = _methods.get_method(self.method_var.get()).schema
            groups: dict = {}
            for spec in schema:
                groups.setdefault(spec.group, []).append(spec)
            col = 0
            for gname, specs in groups.items():
                box = ttk.Labelframe(self.params_frame, text=gname, padding=6)
                box.grid(row=0, column=col, sticky="nw", padx=6, pady=2)
                col += 1
                for r, spec in enumerate(specs):
                    label = spec.label + (f" ({spec.unit})" if spec.unit else "")
                    if spec.type == "bool":
                        var = tk.BooleanVar(value=bool(spec.default))
                        ttk.Checkbutton(box, text=label, variable=var).grid(
                            row=r, column=0, columnspan=2, sticky="w", pady=1)
                    else:
                        ttk.Label(box, text=label + ":").grid(row=r, column=0, sticky="w", pady=1)
                        if spec.type == "choice":
                            var = tk.StringVar(value=str(spec.default))
                            ttk.Combobox(box, state="readonly", width=12,
                                         values=list(spec.choices or ()),
                                         textvariable=var).grid(row=r, column=1, sticky="w", padx=4)
                        else:
                            var = tk.StringVar(value="" if spec.default is None else str(spec.default))
                            ttk.Entry(box, width=10, textvariable=var).grid(
                                row=r, column=1, sticky="w", padx=4)
                    self.param_vars[spec.key] = (spec, var)

        def _collect_params(self) -> dict:
            """Raw field values for params_from_fields. Empty numeric/choice fields
            are omitted so they fall back to the schema default (e.g. blank 'air' = auto)."""
            raw = {}
            for key, (spec, var) in self.param_vars.items():
                v = var.get()
                if spec.type in ("int", "float", "choice") and isinstance(v, str) and v.strip() == "":
                    continue
                raw[key] = v
            return raw

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
                tail = float(self.tail_var.get())
                bin_xy = int(self.bin_var.get())
            except ValueError:
                self._append("! Bin / tail fraction must be numbers.\n")
                return

            method = self.method_var.get()
            params = params_from_fields(method, self._collect_params())
            params_path = write_params_tempfile(params)

            self.stem = stem_of(folder)
            self.out_dir = output_dir(save, folder)
            os.makedirs(self.out_dir, exist_ok=True)
            self._run_params = {"input": folder, "bin": bin_xy, "method": method,
                                "tail_frac": tail, "params": params}
            cmd = build_command(folder, self.out_dir, bin_xy=bin_xy, tail_frac=tail,
                                method=method, params_path=params_path)

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
            method = self._run_params.get("method", "")
            if frac is not None:
                self.headline.configure(text=f"{frac * 100:.1f}%  of tissue embolised   ·   {method}")
            else:
                self.headline.configure(text=f"done (curve not found)   ·   {method}")
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
            self.inspect_btn.configure(state="normal" if os.path.exists(paths["inspect"]) else "disabled")
            _state = os.path.join(self.out_dir, f"{self.stem}_cs_inspect_state.npz")
            self.apply_btn.configure(state="normal" if os.path.exists(_state) else "disabled")
            self.annot_btn.configure(state="normal" if os.path.exists(paths["annotate"]) else "disabled")
            self.folder_btn.configure(state="normal")
            self.bundle_btn.configure(state="normal")

        def _open_proof(self) -> None:
            if getattr(self, "_proof_path", "") and os.path.exists(self._proof_path):
                reveal(self._proof_path)

        def _open_inspector(self) -> None:
            i = artefact_paths(self.out_dir, self.stem)["inspect"]
            if os.path.exists(i):
                webbrowser.open("file://" + os.path.abspath(i))

        def _apply_corrections(self) -> None:
            """Run the Phase-2 recompute: pick the corrections JSON exported from the
            inspector, re-apply it to the preserved baseline, and open the corrected view."""
            rp = getattr(self, "_run_params", None)
            state = os.path.join(self.out_dir, f"{self.stem}_cs_inspect_state.npz")
            if not rp or not os.path.exists(state):
                self._append("! Run a dataset and open Inspect first.\n")
                return
            downloads = os.path.expanduser("~/Downloads")
            edits = filedialog.askopenfilename(
                title="Choose the corrections JSON exported from the inspector",
                initialdir=downloads if os.path.isdir(downloads) else os.path.expanduser("~"),
                filetypes=[("Corrections JSON", "*.json"), ("All files", "*")])
            if not edits:
                return
            exe = sys.executable
            frozen = bool(getattr(sys, "frozen", False))
            head = [exe] if frozen else [exe, "-m", "cavidetect.cli"]
            cmd = head + ["inspect-apply", "--run", self.out_dir,
                          "--stem", f"{self.stem}_cs", "--edits", edits,
                          "--source", rp["input"], "--bin", str(rp["bin"])]
            self._append("$ " + " ".join(cmd) + "\n\n")
            self.apply_btn.configure(state="disabled")

            def worker():
                cwd = None if frozen else repo_root()
                proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, bufsize=1,
                                        encoding="utf-8", errors="replace")
                for line in proc.stdout:
                    self.root.after(0, self._append, line)
                proc.wait()
                corrected = os.path.join(self.out_dir, f"{self.stem}_cs_inspect_corrected.html")

                def done():
                    self.apply_btn.configure(state="normal")
                    if proc.returncode == 0 and os.path.exists(corrected):
                        self._append("\n✓ corrections applied — opening corrected inspector\n")
                        webbrowser.open("file://" + os.path.abspath(corrected))
                    else:
                        self._append("\n! inspect-apply failed (see log above)\n")
                self.root.after(0, done)

            threading.Thread(target=worker, daemon=True).start()

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
                     f"method: {self._run_params.get('method', '')}",
                     f"bin: {self._run_params.get('bin', '')}",
                     f"tail_frac: {self._run_params.get('tail_frac', '')}",
                     f"params: {self._run_params.get('params', '')}"]
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
