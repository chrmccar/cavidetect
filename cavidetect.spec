# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the self-contained cavidetect app.

Builds ONE windowed executable from cavidetect/app.py. That executable is
dual-purpose (GUI by default; pipeline worker when the first arg is a CLI
subcommand), so the GUI can spawn it as a child process for the colour-shape run.

Build on the target OS (PyInstaller does not cross-compile):
    pip install -e . pyinstaller
    pyinstaller cavidetect.spec --noconfirm
Output: dist/cavidetect.app (macOS) or dist/cavidetect/ with cavidetect.exe (Windows).
CI (.github/workflows/build.yml) runs this on macos-latest and windows-latest.
"""
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], []
# Pull in everything these scientific packages need; they ship data files and lazily
# imported submodules that a bare analysis would otherwise miss.
for pkg in ("skimage", "scipy", "matplotlib", "tifffile", "pandas", "PIL", "numpy"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

a = Analysis(
    ["cavidetect/app.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["napari", "pystackreg", "ruptures", "sklearn"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="cavidetect",
    console=False,           # windowed GUI app
    disable_windowed_traceback=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name="cavidetect",
)
app = BUNDLE(
    coll,
    name="cavidetect.app",   # only used on macOS
    icon=None,
    bundle_identifier="au.christophermccarthy.cavidetect",
)
