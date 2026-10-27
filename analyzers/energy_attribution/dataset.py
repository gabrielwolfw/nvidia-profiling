"""Window datasets: train/test/excluded splits and the activity design matrix."""

from dataclasses import dataclass
from functools import cached_property

import numpy as np


def split_point(window_count, train_fraction, strict=True):
    if strict and window_count < 7:
        raise ValueError(
            f"Only {window_count} complete windows overlap kernels and "
            "telemetry; at least 7 are required for a temporal holdout"
        )
    train_count = int(window_count * train_fraction)
    return min(max(train_count, 5), window_count - 2)


def window_durations_ns(window_rows):
    return np.asarray([row["duration_ns"] for row in window_rows], dtype=np.float64)


def activity_design(activity_ns, window_rows):
    """Intercept plus each family's busy fraction of the window's real length."""
    fraction = activity_ns / window_durations_ns(window_rows)[:, None]
    return np.column_stack([np.ones(len(fraction)), fraction])


def assign_splits(
    window_rows, train_fraction, window_ns, min_window_fraction, strict=True
):
    """Label windows train/test/excluded and return the training count.

    Windows shorter than ``min_window_fraction * window_ns`` are "excluded":
    their mean power is unreliable, so they are left out of fitting and
    evaluation. Only a session's final window can be short, so training is
    still the chronological prefix ``[:train_count]``.
    """
    complete = [
        row["duration_ns"] >= min_window_fraction * window_ns
        for row in window_rows
    ]
    train_count = split_point(sum(complete), train_fraction, strict)
    if not all(complete[:train_count]):
        raise ValueError("A truncated window falls inside the training prefix")
    for index, row in enumerate(window_rows):
        if index < train_count:
            row["split"] = "train"
        else:
            row["split"] = "test" if complete[index] else "excluded"
    return train_count


def split_mask(window_rows, *splits):
    return np.asarray([row["split"] in splits for row in window_rows])



@dataclass
class WindowDataset:
    """Windows ready for fitting: one diagnostic row per window.

    ``rows`` are the CSV records (they carry each window's ``split``), and
    rows ``[:train_count]`` are the training windows. ``activity_ns`` is
    (windows, families) busy time and ``target_power_w`` the measured mean
    dynamic power the regression explains.
    """

    rows: list
    activity_ns: np.ndarray
    target_power_w: np.ndarray
    train_count: int

    @property
    def durations_ns(self):
        """Real length of every window (the last one may be truncated)."""
        return window_durations_ns(self.rows)

    @cached_property
    def design(self):
        """Intercept column plus each family's busy fraction per window."""
        return activity_design(self.activity_ns, self.rows)

    def mask(self, *splits):
        """Boolean mask of the rows whose split is one of ``splits``."""
        return split_mask(self.rows, *splits)
