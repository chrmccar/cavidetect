"""Tests for the colour-shape onset timing (start-of-rise, not late global crossing)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cavidetect import colour_shape as cs


def _step_stack(T=30, step=10, lo=20.0, hi=200.0):
    """A (T,1,2) stack: pixel 0 steps lo->hi at frame `step` and stays; pixel 1 stays dark."""
    stack = np.full((T, 1, 2), lo, np.float32)
    stack[step:, 0, 0] = hi
    return stack


def test_onset_dates_the_start_of_the_rise_not_the_end():
    stack = _step_stack(T=30, step=10)
    mask = np.array([[True, False]])
    onset = cs.onset_start_of_rise(stack, mask, rise_frac=0.4, persist=3, smooth=3)
    # the embolised pixel is dated near its step (frame 10), NOT near the end of the clip
    assert 8 <= onset[0, 0] <= 12, onset[0, 0]
    # the non-embolised pixel is marked -1
    assert onset[0, 1] == -1


def test_onset_is_earlier_than_the_old_last_crossing_rule():
    # a pixel that rises at frame 10 but flickers brighter again late would, under the old
    # "last frame below a high global threshold + 1" rule, be dated late. Start-of-rise stays early.
    stack = _step_stack(T=40, step=10)
    stack[30:, 0, 0] = 255.0            # a late extra-bright flicker
    mask = np.array([[True, False]])
    onset = cs.onset_start_of_rise(stack, mask)
    assert onset[0, 0] <= 14, onset[0, 0]


def test_onset_all_in_range_and_masked():
    stack = _step_stack()
    mask = np.array([[True, True]])
    onset = cs.onset_start_of_rise(stack, mask)
    assert onset.min() >= 0 and onset.max() <= stack.shape[0] - 1
