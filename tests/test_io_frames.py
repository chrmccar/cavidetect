"""Folder-of-frames ingest: load a directory of single-page TIFFs as one (T,Y,X) stack.

The new raw format is one picture file per timepoint (t_0001.tif ...) in a folder,
unlike the old multi-page crop stacks. These tests pin the behaviour that matters:
numeric frame ordering, dropping a stray duplicate ("- Copy"), RGB->grey collapse,
and using the fallback micron scale (the per-frame files carry no real scale)."""
import os, sys
import numpy as np
import tifffile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cavidetect import io as cio


def _write_frame(path, value, shape=(8, 10), rgb=False):
    a = np.full(shape, value, dtype=np.uint8)
    if rgb:
        a = np.stack([a, a, a], axis=-1)
    tifffile.imwrite(path, a)


def test_load_frames_orders_numerically_and_drops_duplicate(tmp_path):
    # written out of order; each frame's pixel value encodes its index * 10
    for i in (3, 1, 5, 2, 4):
        _write_frame(tmp_path / f"t_{i:04d}.tif", i * 10)
    # a stray duplicate of frame 1 that must NOT become a second frame
    _write_frame(tmp_path / "t_0001 - Copy.tif", 99, rgb=True)

    stack, meta = cio.load_frames(str(tmp_path))

    assert stack.shape == (5, 8, 10), f"expected 5 frames, got {stack.shape}"
    # frames in ascending index order: means ~ 10,20,30,40,50 (duplicate excluded)
    assert [round(float(f.mean())) for f in stack] == [10, 20, 30, 40, 50]
    assert meta.n_frames == 5


def test_load_frames_uses_fallback_scale_not_bogus_metadata(tmp_path):
    for i in (1, 2, 3):
        _write_frame(tmp_path / f"t_{i:04d}.tif", i)
    stack, meta = cio.load_frames(str(tmp_path), fallback_um_per_px=2.5)
    assert meta.um_per_px == 2.5
    assert meta.scale_from_metadata is False


def test_load_frames_collapses_rgb_to_grey(tmp_path):
    for i in (1, 2):
        _write_frame(tmp_path / f"t_{i:04d}.tif", i * 10, rgb=True)
    stack, meta = cio.load_frames(str(tmp_path))
    assert stack.ndim == 3, f"frames not collapsed to 2D: {stack.shape}"
    assert stack.dtype == np.float32


def test_load_dispatches_directory_to_frames(tmp_path):
    for i in (1, 2, 3):
        _write_frame(tmp_path / f"t_{i:04d}.tif", i * 10)
    stack, meta = cio.load(str(tmp_path))
    assert stack.shape == (3, 8, 10)
