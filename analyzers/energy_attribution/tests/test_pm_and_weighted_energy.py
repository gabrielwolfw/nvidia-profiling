"""Regression tests for PM association, weighted energy and PM plotting.

Every scenario uses a small hand-built capture whose expected values can be
derived by hand, so the assertions document the attribution rules.
"""

import csv
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from energy_attribution.commands import (  # pylint: disable=wrong-import-position
    pm_sampling_plot,
)
from energy_attribution.pm_association import (  # pylint: disable=wrong-import-position
    associate_pm_samples,
    build_kernel_rows,
    load_kernels,
    pm_metric_columns,
    read_pm_duration_stats,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
ANALYZERS_ROOT = REPO_ROOT / "analyzers"
PM_METRIC = "sm__inst_executed_realtime.avg.pct_of_peak_sustained_elapsed"
NS_PER_SECOND = 1_000_000_000


def write_rows(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination)
        writer.writerow(fieldnames)
        writer.writerows(rows)
    return path


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def read_summary(path):
    return {row["metric"]: row["value"] for row in read_rows(path)}


def run_script(script, *arguments, check=True):
    environment = dict(os.environ, MPLBACKEND="Agg")
    return subprocess.run(
        [sys.executable, str(ANALYZERS_ROOT / script), *map(str, arguments)],
        cwd=REPO_ROOT,
        env=environment,
        check=check,
        capture_output=True,
        text=True,
    )


def write_pm_capture(root):
    """Writes kernels and PM samples with hand-derivable associations.

    Kernels k1 [100, 300) and k2 [200, 400) are fully covered by 100 ns
    samples; k3 [1000, 1100) sees 50 ns of one valid sample, then a sample
    with a NaN metric. The 550 ns hole before sample s4 exceeds the 500 ns
    maximum duration (5 x the 100 ns median) and is reported as a gap.
    """
    kernels = write_rows(
        root / "kernels.csv",
        ["event_id", "start_ns", "end_ns", "kernel_name"],
        [
            ["3", 1000, 1100, "k3"],
            ["1", 100, 300, "k1"],
            ["bad", 500, 500, "empty"],
            ["2", 200, 400, "k2"],
        ],
    )
    samples = write_rows(
        root / "pm_samples.csv",
        ["sample", "start_ns", "end_ns", "duration_ns", "metric_a"],
        [
            [0, 0, 100, 100, 10],
            [1, 100, 200, 100, 20],
            [2, 200, 300, 100, 40],
            [3, 300, 400, 100, 60],
            [4, 950, 1050, 100, 80],
            [5, 1050, 1150, 100, "nan"],
            [6, 1150, 2150, 1000, 99],
            ["x", "x", 2200, 0, 1],
        ],
    )
    return kernels, samples


class PmAssociationTests(unittest.TestCase):
    """Time-weighted association of PM samples with kernel intervals."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.kernels_csv, self.samples_csv = write_pm_capture(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def test_library_accumulates_overlap_weighted_means(self):
        kernels, _, skipped = load_kernels(self.kernels_csv)
        self.assertEqual(skipped, 1)
        self.assertEqual([row["event_id"] for row in kernels], ["1", "2", "3"])
        metrics = pm_metric_columns(self.samples_csv)
        self.assertEqual(metrics, ["metric_a"])
        self.assertEqual(read_pm_duration_stats(self.samples_csv), (8, 1, 100))

        accumulators, stats = associate_pm_samples(
            self.samples_csv, kernels, metrics, (10, 500)
        )
        self.assertEqual(stats.valid, 5)
        self.assertEqual(stats.rejected_duration, 1)
        self.assertEqual(stats.rejected_values, 1)
        self.assertEqual(stats.gap_events, [(400, 950, 550)])

        rows, covered, complete = build_kernel_rows(
            kernels, accumulators, metrics
        )
        self.assertEqual((covered, complete), (3, 2))
        by_event = {row["event_id"]: row for row in rows}
        self.assertAlmostEqual(by_event["1"]["metric_a"], 30.0)
        self.assertAlmostEqual(by_event["2"]["metric_a"], 50.0)
        self.assertAlmostEqual(by_event["3"]["metric_a"], 80.0)
        self.assertEqual(by_event["1"]["pm_sample_count"], 2)
        self.assertEqual(by_event["3"]["pm_covered_ns"], 50)
        self.assertAlmostEqual(by_event["3"]["pm_coverage_pct"], 50.0)

    def test_overlapping_samples_count_covered_time_once(self):
        kernels, _, _ = load_kernels(self.kernels_csv)
        overlapping = write_rows(
            self.root / "overlapping.csv",
            ["sample", "start_ns", "end_ns", "duration_ns", "metric_a"],
            [[0, 100, 250, 150, 10], [1, 150, 300, 150, 40]],
        )
        accumulators, _ = associate_pm_samples(
            overlapping, kernels[:1], ["metric_a"], (1, 1_000)
        )
        rows, _, complete = build_kernel_rows(
            kernels[:1], accumulators, ["metric_a"]
        )
        self.assertEqual(rows[0]["pm_covered_ns"], 200)
        self.assertAlmostEqual(rows[0]["pm_coverage_pct"], 100.0)
        self.assertAlmostEqual(rows[0]["metric_a"], 25.0)
        self.assertEqual(complete, 1)

    def test_rejects_pm_csv_without_metrics(self):
        no_metrics = write_rows(
            self.root / "no_metrics.csv",
            ["sample", "start_ns", "end_ns", "duration_ns"],
            [[0, 0, 100, 100]],
        )
        with self.assertRaisesRegex(ValueError, "no metric columns"):
            pm_metric_columns(no_metrics)

    def test_command_writes_kernel_metrics_summary_and_plots(self):
        output = self.root / "pm-kernel-analysis"
        result = run_script(
            "analyze_kernel_pm_metrics.py",
            "--pm-samples",
            self.samples_csv,
            "--kernels",
            self.kernels_csv,
            "--output-dir",
            output,
            "--plot",
        )
        self.assertIn("Kernels with >=99.9% PM coverage: 2", result.stdout)

        rows = read_rows(output / "kernel_pm_metrics.csv")
        self.assertEqual(
            list(rows[0].keys()),
            [
                "event_id",
                "start_ns",
                "end_ns",
                "kernel_name",
                "pm_sample_count",
                "pm_covered_ns",
                "pm_coverage_pct",
                "metric_a",
            ],
        )
        self.assertEqual([row["event_id"] for row in rows], ["1", "2", "3"])
        self.assertEqual(
            [float(row["metric_a"]) for row in rows], [30.0, 50.0, 80.0]
        )

        summary = read_summary(output / "pm_kernel_summary.csv")
        self.assertEqual(summary["kernels"], "3")
        self.assertEqual(summary["kernels_skipped"], "1")
        self.assertEqual(summary["kernels_with_pm_coverage"], "3")
        self.assertEqual(summary["pm_samples_raw"], "8")
        self.assertEqual(summary["pm_samples_invalid"], "1")
        self.assertEqual(summary["pm_samples_valid"], "5")
        self.assertEqual(summary["pm_samples_rejected_duration"], "1")
        self.assertEqual(summary["pm_samples_rejected_values"], "1")
        self.assertEqual(float(summary["pm_max_duration_ns"]), 500.0)
        for name in ("pm_coverage_timeline.png", "pm_sampling_gaps.png"):
            self.assertGreater((output / name).stat().st_size, 0)

    def test_command_rejects_inverted_duration_factors(self):
        result = run_script(
            "analyze_kernel_pm_metrics.py",
            "--pm-samples",
            self.samples_csv,
            "--kernels",
            self.kernels_csv,
            "--output-dir",
            self.root / "unused",
            "--min-duration-factor",
            "2",
            "--max-duration-factor",
            "1",
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--max-duration-factor", result.stderr)


def write_energy_capture(root, b_coverage_pct=100.0):
    """Writes one second of constant 1 W telemetry and three kernels.

    A runs over [0, 0.5) s with PM weight 3, B over [0.25, 0.5) s with
    weight 1 and C over [0.75, 1) s without PM metrics. Each quarter second
    therefore carries 250 mJ, shared as: A alone; A and B 3:1 (or equally
    without PM weights); nobody (unassigned); C alone.
    """
    telemetry = write_rows(
        root / "telemetry.csv",
        ["timestamp_ns", "power_mw"],
        [[0, 1000.0], [NS_PER_SECOND // 2, 1000.0], [NS_PER_SECOND, 1000.0]],
    )
    quarter = NS_PER_SECOND // 4
    kernels = write_rows(
        root / "kernels.csv",
        ["event_id", "pid", "start_ns", "end_ns", "kernel_name"],
        [
            ["a", 7, 0, 2 * quarter, "kernel_a"],
            ["b", 7, quarter, 2 * quarter, "kernel_b"],
            ["broken", 7, "x", 0, "kernel_broken"],
            ["c", 7, 3 * quarter, 4 * quarter, "kernel_c"],
        ],
    )
    pm_metrics = write_rows(
        root / "kernel_pm_metrics.csv",
        ["event_id", "pm_coverage_pct", PM_METRIC],
        [["a", 100.0, 3.0], ["b", b_coverage_pct, 1.0]],
    )
    return telemetry, kernels, pm_metrics


class WeightedKernelEnergyTests(unittest.TestCase):
    """Streaming split of measured energy among concurrent kernels."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def analyze(self, use_pm=True, b_coverage_pct=100.0, extra=()):
        telemetry, kernels, pm_metrics = write_energy_capture(
            self.root, b_coverage_pct
        )
        output = self.root / "energy-analysis"
        arguments = [
            "--telemetry",
            telemetry,
            "--kernels",
            kernels,
            "--window-ms",
            "500",
            "--output-dir",
            output,
            *extra,
        ]
        if use_pm:
            arguments += ["--pm-kernel-metrics", pm_metrics]
        result = run_script("analyze_kernel_energy.py", *arguments)
        energy = {
            row["event_id"]: row
            for row in read_rows(output / "kernel_energy.csv")
        }
        summary = read_summary(output / "energy_summary.csv")
        return result, output, energy, summary

    def assert_energy(self, row, column, expected):
        self.assertAlmostEqual(float(row[column]), expected, places=6)

    def assert_summary(self, summary, metric, expected):
        self.assertAlmostEqual(float(summary[metric]), expected, places=6)

    def test_pm_weights_split_concurrent_energy(self):
        _, output, energy, summary = self.analyze()
        self.assertEqual(list(energy), ["a", "b", "c"])
        self.assert_energy(energy["a"], "estimated_dynamic_energy_mj", 437.5)
        self.assert_energy(energy["a"], "pm_weighted_energy_mj", 437.5)
        self.assert_energy(energy["b"], "estimated_dynamic_energy_mj", 62.5)
        self.assert_energy(energy["c"], "estimated_dynamic_energy_mj", 250.0)
        self.assert_energy(energy["c"], "equal_share_fallback_energy_mj", 250.0)
        self.assertEqual(energy["a"]["pm_energy_weight"], "3.0")
        self.assertEqual(energy["c"]["pm_coverage_pct"], "")

        self.assertEqual(
            summary["attribution_method"],
            "hybrid_pm_weighted_with_equal_share_fallback",
        )
        self.assert_summary(summary, "device_energy_mj", 1000.0)
        self.assert_summary(summary, "assigned_dynamic_energy_mj", 750.0)
        self.assert_summary(summary, "unassigned_dynamic_energy_mj", 250.0)
        self.assert_summary(summary, "pm_weighted_energy_mj", 500.0)
        self.assert_summary(summary, "equal_share_fallback_energy_mj", 250.0)
        self.assert_summary(summary, "conservation_error_mj", 0.0)
        self.assertEqual(summary["pm_matched_kernels"], "2")
        self.assertEqual(summary["pm_eligible_kernels"], "2")

        windows = read_rows(output / "energy_windows.csv")
        self.assertEqual(
            [float(row["kernel_active_pct"]) for row in windows], [100.0, 50.0]
        )
        self.assertEqual(
            [float(row["dynamic_energy_mj"]) for row in windows], [500.0, 500.0]
        )

    def test_low_coverage_falls_back_to_equal_share(self):
        _, _, energy, summary = self.analyze(b_coverage_pct=99.0)
        self.assertEqual(energy["b"]["pm_energy_weight"], "")
        self.assert_energy(energy["a"], "estimated_dynamic_energy_mj", 375.0)
        self.assert_energy(energy["a"], "pm_weighted_energy_mj", 250.0)
        self.assert_energy(energy["b"], "equal_share_fallback_energy_mj", 125.0)
        self.assertEqual(summary["pm_eligible_kernels"], "1")
        self.assert_summary(summary, "pm_weighted_energy_mj", 250.0)
        self.assert_summary(summary, "equal_share_fallback_energy_mj", 500.0)

    def test_coverage_threshold_is_configurable(self):
        _, _, energy, _ = self.analyze(
            b_coverage_pct=99.0, extra=("--pm-min-coverage-pct", "90")
        )
        self.assert_energy(energy["b"], "pm_weighted_energy_mj", 62.5)

    def test_without_pm_metrics_kernels_share_equally(self):
        result, output, energy, summary = self.analyze(
            use_pm=False, extra=("--plot",)
        )
        self.assertEqual(
            summary["attribution_method"], "equal_share_concurrent_kernels"
        )
        self.assert_energy(energy["a"], "estimated_dynamic_energy_mj", 375.0)
        self.assert_energy(energy["b"], "estimated_dynamic_energy_mj", 125.0)
        self.assert_energy(energy["c"], "estimated_dynamic_energy_mj", 250.0)
        self.assert_summary(summary, "pm_weighted_energy_mj", 0.0)
        self.assertEqual(summary["pm_min_coverage_pct"], "")
        self.assertTrue(
            math.isnan(float(summary["utilization_dynamic_energy_pearson_r"]))
        )
        self.assertIn("WARNING: Skipped 1 invalid kernel rows.", result.stdout)
        for name in (
            "energy_timeline.png",
            "utilization_energy_correlation.png",
        ):
            self.assertGreater((output / name).stat().st_size, 0)

    def test_idle_baseline_is_removed_from_dynamic_energy(self):
        _, _, energy, summary = self.analyze(
            use_pm=False, extra=("--idle-power-mw", "400")
        )
        self.assert_summary(summary, "device_energy_mj", 1000.0)
        self.assert_summary(summary, "dynamic_energy_mj", 600.0)
        self.assert_energy(energy["c"], "estimated_dynamic_energy_mj", 150.0)

    def test_unsorted_kernels_are_rejected(self):
        telemetry, _, _ = write_energy_capture(self.root)
        unsorted = write_rows(
            self.root / "unsorted.csv",
            ["event_id", "start_ns", "end_ns"],
            [["late", 500, 600], ["early", 100, 200]],
        )
        result = run_script(
            "analyze_kernel_energy.py",
            "--telemetry",
            telemetry,
            "--kernels",
            unsorted,
            "--output-dir",
            self.root / "unused",
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("sorted by start_ns", result.stderr)


class PmSamplingPlotTests(unittest.TestCase):
    """Standalone per-metric plots of a raw PM Sampling CSV."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.samples = write_rows(
            self.root / "pm_sampling.csv",
            ["sample", "start_ns", "end_ns", "duration_ns", PM_METRIC, "x/y"],
            [
                [0, 0, 5_000_000, 5_000_000, 10.0, "n/a"],
                [2, 3_000_000, 4_000_000, 1_000_000, 30.0, "n/a"],
                [1, 2_000_000, 3_000_000, 1_000_000, 20.0, "n/a"],
            ],
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_helpers_name_and_label_metrics(self):
        self.assertEqual(pm_sampling_plot.safe_filename("a/b c:d"), "a_b_c_d")
        self.assertEqual(len(pm_sampling_plot.safe_filename("m" * 300)), 200)
        self.assertEqual(
            pm_sampling_plot.get_ylabel(PM_METRIC),
            "Percent of peak sustained elapsed (%)",
        )
        self.assertEqual(
            pm_sampling_plot.get_ylabel("dram__bytes.per_second"),
            "Rate (per second)",
        )
        self.assertEqual(pm_sampling_plot.get_ylabel("other"), "Metric value")

    def test_load_discards_first_sample_and_orders_by_time(self):
        frame = pm_sampling_plot.load_samples(self.samples, False)
        self.assertEqual(frame["sample"].tolist(), [1, 2])
        self.assertEqual(frame["time_ms"].tolist(), [0.0, 1.0])
        kept = pm_sampling_plot.load_samples(self.samples, True)
        self.assertEqual(kept["sample"].tolist(), [0, 1, 2])
        self.assertEqual(
            pm_sampling_plot.detect_metrics(kept), [PM_METRIC, "x/y"]
        )

    def test_command_plots_numeric_metrics_and_skips_others(self):
        output = self.root / "plots"
        result = run_script(
            "plot_pm_sampling.py", self.samples, "--output", output
        )
        self.assertEqual(
            sorted(path.name for path in output.iterdir()),
            [f"{PM_METRIC}.png"],
        )
        self.assertIn("[SKIP] x/y: no valid numeric samples", result.stdout)
        self.assertIn("Samples analysed : 2", result.stdout)
        self.assertIn("Total time       : 2.000 ms", result.stdout)

    def test_command_rejects_missing_columns(self):
        broken = write_rows(
            self.root / "broken.csv",
            ["sample", "start_ns", "metric"],
            [[0, 0, 1.0]],
        )
        result = run_script(
            "plot_pm_sampling.py",
            broken,
            "--output",
            self.root / "unused",
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("duration_ns, end_ns", result.stderr)


if __name__ == "__main__":
    unittest.main()
