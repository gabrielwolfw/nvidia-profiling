"""Tests for the ViT benchmark helpers and the ViT sweep analysis."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

BENCHMARKS_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARKS_DIRECTORY / "scripts"))
sys.path.insert(0, str(BENCHMARKS_DIRECTORY / "pytorch"))

import analyze_vit_sweep  # pylint: disable=wrong-import-position
import vit_inference  # pylint: disable=wrong-import-position

NS = 1_000_000_000


def write_session(root: Path, dtype: str, power_w: float, images: int) -> Path:
    """Writes a session: 10 s measurement at ``power_w``, 5 s idle at 2 W."""
    session = root / "sessions" / f"01-{dtype}-rep-01"
    session.mkdir(parents=True)
    markers = [
        ("benchmark_warmup_start_marker_kernel", 0),
        ("benchmark_measurement_start_marker_kernel", 2 * NS),
        ("benchmark_cooldown_start_marker_kernel", 12 * NS),
        ("benchmark_complete_marker_kernel", 17 * NS),
    ]
    with (session / "kernel_activity.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["event_id", "start_ns", "end_ns", "kernel_name"])
        for index, (name, start) in enumerate(markers):
            writer.writerow([index, start, start + 1000, name])
    with (session / "gpu_telemetry.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "timestamp_ns",
                "power_mw",
                "temperature_c",
                "gpu_utilization_pct",
                "memory_utilization_pct",
                "graphics_clock_mhz",
            ]
        )
        for tenth in range(170):
            timestamp = tenth * NS // 10 + NS // 20
            busy = 2 * NS <= timestamp < 12 * NS
            power = power_w if busy else 2.0
            writer.writerow(
                [timestamp, power * 1000, 50, 100 if busy else 0, 10, 1000]
            )
    (session / "benchmark.log").write_text(
        f"DTYPE={dtype}\nMODEL=vit_base_patch16_224\n"
        f"MEASUREMENT_IMAGES={images}\nIMAGES_PER_SECOND={images / 10}\n"
        'STATUS=OK\nResults: "/tmp/x"\n',
        encoding="utf-8",
    )
    return session


class AnalyzeVitSweepTests(unittest.TestCase):
    """Energy per image from marker-delimited NVML telemetry."""

    def test_session_energy_per_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session = write_session(Path(directory), "fp16", 20.0, 4000)
            row = analyze_vit_sweep.summarize_session(session)
        self.assertEqual(row["dtype"], "fp16")
        self.assertEqual(row["telemetry_samples"], 100)
        self.assertAlmostEqual(row["mean_power_w"], 20.0)
        self.assertAlmostEqual(row["idle_power_w"], 2.0)
        # 20 W over ~10 s is 200 J; 200 J / 4000 images = 50 mJ.
        self.assertAlmostEqual(row["energy_per_image_mj"], 50.0, places=2)
        self.assertAlmostEqual(
            row["dynamic_energy_per_image_mj"], 45.0, places=2
        )
        self.assertAlmostEqual(row["images_per_joule"], 20.0, places=2)

    def test_failed_benchmark_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session = write_session(Path(directory), "fp8", 20.0, 10)
            (session / "benchmark.log").write_text("STATUS=FAILED\n")
            with self.assertRaisesRegex(ValueError, "STATUS=OK"):
                analyze_vit_sweep.summarize_session(session)

    def test_aggregate_by_precision(self) -> None:
        rows = [
            {
                "dtype": "bf16",
                **{f: v for f in analyze_vit_sweep.AGGREGATE_FIELDS},
            }
            for v in (1.0, 3.0)
        ]
        summaries = analyze_vit_sweep.aggregate(rows)
        self.assertEqual(len(summaries), 1)
        summary = summaries[0]
        self.assertEqual(summary["sessions"], 2)
        self.assertEqual(summary["mean_power_w_mean"], 2.0)
        self.assertAlmostEqual(summary["mean_power_w_stddev"], 2**0.5)


class VitInferenceTests(unittest.TestCase):
    """Benchmark helpers that do not need PyTorch."""

    def test_strip_profiler_preload_keeps_other_libraries(self) -> None:
        environment = {
            "LD_PRELOAD": "/opt/a.so:/x/libnvidia-process-metrics.so /b.so"
        }
        self.assertTrue(vit_inference.strip_profiler_preload(environment))
        self.assertEqual(environment["LD_PRELOAD"], "/opt/a.so /b.so")
        self.assertFalse(vit_inference.strip_profiler_preload(environment))

    def test_strip_profiler_preload_removes_empty_variable(self) -> None:
        environment = {"LD_PRELOAD": "/x/libnvidia-process-metrics.so"}
        self.assertTrue(vit_inference.strip_profiler_preload(environment))
        self.assertNotIn("LD_PRELOAD", environment)

    def test_arguments_are_validated(self) -> None:
        args = vit_inference.parse_args(["--dtype", "fp8", "--compile"])
        self.assertEqual(args.compile_mode, "default")
        with self.assertRaises(SystemExit):
            vit_inference.parse_args(["--batch-size", "0"])
        with self.assertRaises(SystemExit):
            vit_inference.parse_args(["--compile-mode", "reduce-overhead"])

    def test_preflight_reports_unsupported_precision_and_continues(
        self,
    ) -> None:
        class FakeBenchmark:
            """Benchmark whose fp8 build fails like it does below CC 8.9."""

            args = vit_inference.parse_args([])
            built: list[str] = []

            def prepare_markers(self) -> bool:
                return False

            def build(self, dtype: str):
                self.built.append(dtype)
                if dtype == "fp8":
                    raise SystemExit("fp8 needs compute capability >= 8.9")
                return None, None, 0.0

            def run_phase(self, model, inputs, seconds):
                del model, inputs, seconds
                return 1.0, 1

        benchmark = FakeBenchmark()
        self.assertEqual(vit_inference.preflight(benchmark), 1)
        self.assertEqual(benchmark.built, list(vit_inference.DTYPES))


if __name__ == "__main__":
    unittest.main()
