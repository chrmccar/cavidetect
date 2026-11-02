# Distributing cavidetect as a self-contained app

The GUI (`cavidetect-gui`) can be shipped as a **double-click application** that needs
no Python install — a `cavidetect.app` on macOS and a `cavidetect.exe` on Windows.
This doc covers building and handing it over.

## How the packaged app is wired

PyInstaller bundles **one** executable from `cavidetect/app.py`. That executable is
dual-purpose:

- launched normally → opens the **GUI**;
- launched with a CLI subcommand (`colour-shape …`) → runs the **pipeline**.

The GUI runs each analysis by spawning that same executable in worker mode
(`gui.build_command` emits `<exe> colour-shape …` when frozen), so **Cancel** still
works by terminating a real child process — exactly as in the dev version.

## Building (must be done on the target OS — PyInstaller does not cross-compile)

```bash
pip install -e . pyinstaller
pyinstaller cavidetect.spec --noconfirm
```

- **macOS** → `dist/cavidetect.app`
- **Windows** → `dist/cavidetect/` (contains `cavidetect.exe` + its support files; ship
  the whole folder, zipped)

The spec `collect_all`s scipy / scikit-image / matplotlib / tifffile / pandas / Pillow
so nothing is missing at runtime. Bundles are large (~300–500 MB) — that's expected.

## The easy cross-platform route: GitHub Actions (recommended)

I cannot build or test macOS/Windows binaries from the Linux VM, so CI is the
reproducible way to get both. The workflow `.github/workflows/build.yml` builds on
`macos-latest` **and** `windows-latest` and uploads each app as an artifact.

One-time setup (run on your Mac, where the repo lives):

```bash
cd .../project-liverwort-cavitation-event/cavidetect
git init                      # if not already a repo
git add -A && git commit -m "cavidetect app + GUI + CI"
gh repo create cavidetect --private --source=. --push   # needs the gh CLI logged in
```

Then to produce apps:

```bash
git tag v0.1.0 && git push --tags
```

Watch the **Actions** tab; when the run finishes, download `cavidetect-macOS` and
`cavidetect-Windows` from its **Artifacts**. (You can also trigger it manually with
**Run workflow** — no tag needed.)

## Handing it to the other person

1. Send them the zip for their OS (`cavidetect-macos.zip` or `cavidetect-windows.zip`).
2. **macOS:** unzip, drag `cavidetect.app` to Applications. First launch: right-click →
   **Open** → **Open** (the app is unsigned, so a double-click alone is blocked once).
3. **Windows:** unzip the folder, run `cavidetect.exe`. SmartScreen may warn on an
   unsigned app → **More info** → **Run anyway** (one time).
4. In the app: **Choose** the folder of frame TIFFs, **Choose** where to save results,
   click **Run**. When it finishes the frame viewer opens automatically.

## The feedback loop when a detection looks wrong

1. They click **Annotate this dataset**, paint the air-filled tissue on the raw final
   frame, and **Export PNG** (saves to their Downloads).
2. They click **Bundle to send…**, choose where to save the zip, and (optionally) attach
   the annotation PNG they just exported.
3. They send you that one zip — it contains the results, the curve, the proof image,
   their annotation, and an `info.txt` (dataset, parameters, final %). That's everything
   needed to re-tune the detector against their data.

## Notes / limits

- Large stacks need RAM; keep **Bin = 2** (default). Full-res (Bin 1) on a laptop with
  little memory may be slow or swap.
- The apps are **unsigned** (no Apple notarization / Windows code-signing). That only
  means the one-time "open anyway" step above; it is otherwise fine.
- Verified from the dev VM: all helper/onset tests, the pipeline end-to-end, and the
  dual-entry worker dispatch. The actual `.app`/`.exe` and the CI run are validated by
  the GitHub Actions build and by launching on each OS — they cannot be tested on the VM
  (no macOS/Windows, no Tk display).
