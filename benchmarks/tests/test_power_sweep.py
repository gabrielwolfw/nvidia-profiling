"""Tests for deterministic planning and marker-based phase analysis."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIRECTORY = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIRECTORY))

import analyze_power_sweep  # pylint: disable=wrong-import-position
import run_power_sweep  # pylint: disable=wrong-import-position


CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "power_sweep.default.json"
)


class PlanTests(unittest.TestCase):
    """Validate balanced and reproducible execution plans."""

    def test_default_plan_sizes(self) -> None:
        """Pilot and production modes contain every configured condition."""
        config = run_power_sweep.load_config(CONFIG_PATH)
        pilot = run_power_sweep.build_plan(config, "pilot", config.seed)
        production = run_power_sweep.build_plan(
            config, "production", config.seed
        )
        self.assertEqual(len(pilot), 19)
        self.assertEqual(len(production), 110)
        self.assertEqual({entry.repetition for entry in production}, set(range(1, 6)))

    def test_plan_is_reproducible_and_balanced(self) -> None:
        """A seed produces stable ordering and complete repetition blocks."""
        config = run_power_sweep.load_config(CONFIG_PATH)
        first = run_power_sweep.build_plan(config, "production", 42)
        second = run_power_sweep.build_plan(config, "production", 42)
        different = run_power_sweep.build_plan(config, "production", 43)
        first_ids = [entry.condition_id for entry in first]
        self.assertEqual(first_ids, [entry.condition_id for entry in second])
        self.assertNotEqual(first_ids, [entry.condition_id for entry in different])

        reference = {
            entry.condition_id for entry in first if entry.repetition == 1
        }
        for repetition in range(1, 6):
            block = {
                entry.condition_id
                for entry in first
                if entry.repetition == repetition
            }
            self.assertEqual(block, reference)

    def test_result_directory_parser(self) -> None:
        """Launcher output accepts its quoted result-directory format."""
        output = 'Session: abc\nResults: "/tmp/nvidia metrics/abc"\n'
        self.assertEqual(
            run_power_sweep.extract_result_directory(output),
            Path("/tmp/nvidia metrics/abc"),
        )

    def test_production_plan_accepts_pilot_levels(self) -> None:
        """Pilot-selected levels replace defaults without removing idle controls."""
        config = run_power_sweep.load_config(CONFIG_PATH)
        overrides = {
            "compute-fp32": (20.0, 80.0),
            "memory-sequential": (25.0, 75.0),
            "mixed-50-50": (30.0, 70.0),
        }
        plan = run_power_sweep.build_plan(
            config, "production", config.seed, overrides
        )
        self.assertEqual(len(plan), 35)
        idle = [entry for entry in plan if entry.workload_kind == "idle"]
        self.assertEqual(len(idle), 5)
        self.assertTrue(all(entry.duty_cycle_pct == 0.0 for entry in idle))

    def test_level_file_requires_every_active_workload(self) -> None:
        """Production cannot silently fall back for an uncalibrated workload."""
        config = run_power_sweep.load_config(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as temporary:
            levels_path = Path(temporary) / "levels.csv"
            with levels_path.open("w", newline="", encoding="utf-8") as destination:
                writer = csv.DictWriter(
                    destination,
                    fieldnames=("workload_name", "duty_cycle_pct"),
                )
                writer.writeheader()
                writer.writerow(
                    {"workload_name": "compute-fp32", "duty_cycle_pct": 50}
                )
            with self.assertRaisesRegex(ValueError, "missing non-idle workloads"):
                run_power_sweep.load_level_overrides(levels_path, config)


class AnalysisTests(unittest.TestCase):
    """Validate CUPTI-marker phase separation with synthetic CSV data."""

    @staticmethod
    def write_session(study: Path) -> None:
        """Create one minimal completed benchmark session."""
        session = study / "sessions" / "001-compute-rep-01"
        session.mkdir(parents=True)
        condition = {
            "sequence": 1,
            "repetition": 1,
            "condition_id": "compute-fp32-duty-50",
            "workload_name": "compute-fp32",
            "workload_kind": "compute",
            "duty_cycle_pct": 50,
            "status": "completed",
        }
        (session / "condition.json").write_text(
            json.dumps(condition), encoding="utf-8"
        )
        markers = (
            ("benchmark_warmup_start_marker", 100, 110),
            ("benchmark_measurement_start_marker", 200, 210),
            ("benchmark_cooldown_start_marker", 400, 410),
            ("benchmark_complete_marker", 500, 510),
        )
        with (session / "kernel_activity.csv").open(
            "w", newline="", encoding="utf-8"
        ) as destination:
            writer = csv.DictWriter(
                destination, fieldnames=("kernel_name", "start_ns", "end_ns")
            )
            writer.writeheader()
            for name, start, end in markers:
                writer.writerow(
                    {"kernel_name": name, "start_ns": start, "end_ns": end}
                )
        with (session / "gpu_telemetry.csv").open(
            "w", newline="", encoding="utf-8"
        ) as destination:
            fields = ("timestamp_ns", *analyze_power_sweep.TELEMETRY_FIELDS)
            writer = csv.DictWriter(destination, fieldnames=fields)
            writer.writeheader()
            for timestamp, power in ((150, 1000), (250, 2000), (350, 3000), (450, 1000)):
                writer.writerow(
                    {
                        "timestamp_ns": timestamp,
                        "power_mw": power,
                        "temperature_c": 40,
                        "gpu_utilization_pct": 50,
                        "memory_utilization_pct": 25,
                        "graphics_clock_mhz": 1000,
                    }
                )

    def test_phase_boundaries_exclude_markers(self) -> None:
        """Only samples strictly inside semantic phases are summarized."""
        with tempfile.TemporaryDirectory() as temporary:
            study = Path(temporary)
            self.write_session(study)
            phases, samples, errors = analyze_power_sweep.summarize_study(study)
            self.assertFalse(errors)
            self.assertEqual(len(phases), 3)
            self.assertEqual(len(samples), 4)
            measurement = next(row for row in phases if row["phase"] == "measurement")
            self.assertEqual(measurement["telemetry_samples"], 2)
            self.assertEqual(measurement["power_mw_median"], 2500.0)

    def test_power_level_recommendations_follow_requested_grid(self) -> None:
        """Pilot points nearest a uniform wattage grid are selected once."""
        rows = [
            {
                "workload_name": "compute-fp32",
                "workload_kind": "compute",
                "duty_cycle_pct": duty,
                "power_median_mean_w": power,
            }
            for duty, power in ((0, 20), (25, 39), (50, 61), (75, 82), (100, 101))
        ]
        recommendations = analyze_power_sweep.recommend_power_levels(rows, 20.0)
        self.assertEqual(
            [row["duty_cycle_pct"] for row in recommendations],
            [25.0, 50.0, 75.0, 100.0],
        )


if __name__ == "__main__":
    unittest.main()
