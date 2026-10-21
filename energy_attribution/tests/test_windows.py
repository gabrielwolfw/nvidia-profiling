"""Check vectorized window aggregation against straightforward loop references."""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from energy_attribution import windows  # pylint: disable=wrong-import-position
from energy_attribution.diagnostics import (  # pylint: disable=wrong-import-position
    check_identifiability,
)
from energy_attribution.io import (  # pylint: disable=wrong-import-position
    PowerTrace,
    load_kernels,
)


def reference_activity(starts, ends, columns, family_count, lo, hi, window_ns):
    """Row-by-row algorithm of the original streaming implementation."""
    count = windows.window_count(lo, hi, window_ns)
    activity = np.zeros((count, family_count))
    union = np.zeros(count)
    union_end = np.full(count, -1, dtype=np.int64)
    for start, end, column in zip(starts.tolist(), ends.tolist(), columns.tolist()):
        clipped_start, clipped_end = max(start, lo), min(end, hi)
        if clipped_end <= clipped_start:
            continue
        for window in range(
            (clipped_start - lo) // window_ns, (clipped_end - 1 - lo) // window_ns + 1
        ):
            window_start = lo + window * window_ns
            window_end = min(window_start + window_ns, hi)
            overlap_start = max(clipped_start, window_start)
            overlap_end = min(clipped_end, window_end)
            activity[window, column] += overlap_end - overlap_start
            if overlap_end > union_end[window]:
                union[window] += overlap_end - max(overlap_start, union_end[window])
                union_end[window] = overlap_end
    return activity, union


def reference_integral(timestamps, power, start, end, idle):
    device = dynamic = 0.0
    for left_t, right_t, left_p, right_p in zip(
        timestamps, timestamps[1:], power, power[1:]
    ):
        overlap_start, overlap_end = max(start, left_t), min(end, right_t)
        if overlap_start >= overlap_end:
            continue
        values = []
        for timestamp in (overlap_start, overlap_end):
            value = left_p + (timestamp - left_t) / (right_t - left_t) * (right_p - left_p)
            values.append((value, max(0.0, value - idle)))
        seconds = (overlap_end - overlap_start) / 1e9
        device += 0.5 * (values[0][0] + values[1][0]) * seconds
        dynamic += 0.5 * (values[0][1] + values[1][1]) * seconds
    return device, dynamic


class FakeTable:
    def __init__(self, starts, ends):
        self.start_ns = starts
        self.end_ns = ends

    def __len__(self):
        return len(self.start_ns)


def random_events(rng, count, horizon):
    starts = np.sort(rng.integers(0, horizon, count)).astype(np.int64)
    # Mix of short kernels and a few long ones that span many windows.
    durations = np.where(
        rng.random(count) < 0.05,
        rng.integers(1, horizon // 3, count),
        rng.integers(1, 2_000, count),
    )
    return starts, starts + durations


class AggregationTests(unittest.TestCase):
    def test_matches_reference_across_chunks(self):
        rng = np.random.default_rng(7)
        starts, ends = random_events(rng, 3_000, 1_000_000)
        columns = rng.integers(0, 4, len(starts))
        lo, hi, window_ns = 50_000, 900_001, 37_000
        expected_activity, expected_union = reference_activity(
            starts, ends, columns, 4, lo, hi, window_ns
        )
        original_chunk = windows.EVENT_CHUNK
        try:
            for chunk in (len(starts) + 1, 257, 1):
                windows.EVENT_CHUNK = chunk
                activity, union = windows.aggregate_family_activity(
                    FakeTable(starts, ends), columns, 4, lo, hi, window_ns
                )
                np.testing.assert_array_equal(activity, expected_activity)
                np.testing.assert_array_equal(union, expected_union)
        finally:
            windows.EVENT_CHUNK = original_chunk

    def test_event_distribution_conserves_family_window_energy(self):
        rng = np.random.default_rng(11)
        starts, ends = random_events(rng, 2_000, 500_000)
        columns = rng.integers(0, 3, len(starts))
        lo, hi, window_ns = 0, int(ends.max()), 25_000
        table = FakeTable(starts, ends)
        activity, _ = windows.aggregate_family_activity(
            table, columns, 3, lo, hi, window_ns
        )
        energy = rng.random(activity.shape) * (activity > 0)
        (per_event,) = windows.distribute_to_events(
            table, columns, activity, [energy], lo, hi, window_ns
        )
        self.assertAlmostEqual(per_event.sum(), energy.sum(), places=9)

    def test_window_integration_matches_full_scan(self):
        rng = np.random.default_rng(3)
        timestamps = np.cumsum(rng.integers(5_000, 60_000, 200)).astype(np.int64)
        power = rng.uniform(20_000.0, 250_000.0, len(timestamps))
        trace = PowerTrace(timestamps, power)
        starts, ends = windows.window_bounds(
            int(timestamps[3]) + 17, int(timestamps[-5]) - 3, 41_000
        )
        device, dynamic = windows.integrate_windows(trace, starts, ends, 60_000.0)
        for index, (start, end) in enumerate(zip(starts.tolist(), ends.tolist())):
            expected = reference_integral(
                timestamps.tolist(), power.tolist(), start, end, 60_000.0
            )
            self.assertEqual((device[index], dynamic[index]), expected)


class LoaderTests(unittest.TestCase):
    def test_invalid_rows_are_skipped_and_ties_keep_first_appearance(self):
        rows = [
            ["event_id", "start_ns", "end_ns", "kernel_name"],
            ["0", "10", "20", "b"],
            ["1", "11", "x", "a"],
            ["2", "12", "12", "a"],
            ["3", "13", "23", ""],
            ["4", "14", "24", "a,with,commas"],
            ["5", "15", "25", "a"],
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kernels.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerows(rows)
            table = load_kernels(path, load_event_ids=True, chunk_rows=2)
        self.assertEqual(table.skipped, 3)
        self.assertEqual(table.event_ids.tolist(), [0, 4, 5])
        self.assertEqual(table.names, ["b", "a,with,commas", "a"])
        durations, counts = table.name_statistics()
        self.assertEqual([name for name, _ in durations.most_common()], table.names)
        self.assertEqual(counts["a"], 1)


class IdentifiabilityTests(unittest.TestCase):
    def test_underdetermined_design_is_rejected(self):
        design = np.column_stack([np.ones(10), np.arange(10.0), np.arange(10.0) ** 2])
        with self.assertRaises(ValueError):
            check_identifiability(design, 8, ["b0", "f1", "f2"])
        report = check_identifiability(
            design, 8, ["b0", "f1", "f2"], allow_underdetermined=True
        )
        self.assertTrue(report["underdetermined"])

    def test_aliased_and_unseen_families_are_reported(self):
        rng = np.random.default_rng(0)
        shared = rng.random(40)
        design = np.column_stack(
            [np.ones(40), shared, 2.0 * shared, rng.random(40), np.zeros(40)]
        )
        design[35:, 4] = 0.5
        report = check_identifiability(design, 30, ["b0", "a", "b", "c", "d"])
        self.assertEqual(report["aliased_terms"], ["a", "b"])
        self.assertEqual(report["test_families_unseen_in_training"], ["d"])


if __name__ == "__main__":
    unittest.main()
