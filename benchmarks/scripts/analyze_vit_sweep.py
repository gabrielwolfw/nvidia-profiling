#!/usr/bin/env python3
"""Summarize ViT sweep sessions: throughput, power and energy per image.

For every session written by ``run_vit_sweep.sh``, the measurement phase is
located with the CUPTI marker kernels, NVML power is averaged over it and
combined with the image count reported by the benchmark. The cooldown phase
(context idle) gives the baseline for dynamic energy.
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from analyze_power_sweep import (  # pylint: disable=wrong-import-position
    atomic_write_csv,
    load_markers,
    numeric_values,
    phase_ranges,
    read_phase_samples,
)

KEY_VALUE = re.compile(r"^([A-Z][A-Z0-9_]*)=(.*)$", re.MULTILINE)
LOG_FIELDS = (
    "MODEL",
    "DTYPE",
    "BATCH_SIZE",
    "COMPILE",
    "DEVICE_NAME",
    "TORCH_VERSION",
    "REL_L2_ERROR_VS_FP32",
    "MEASUREMENT_IMAGES",
    "IMAGES_PER_SECOND",
    "MS_PER_BATCH",
)
AGGREGATE_FIELDS = (
    "images_per_second",
    "mean_power_w",
    "energy_per_image_mj",
    "dynamic_energy_per_image_mj",
    "images_per_joule",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--study-dir", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Default: <study-dir>/summary",
    )
    return parser.parse_args()


def read_log_fields(path: Path) -> dict[str, str]:
    """Returns the last value of each KEY=VALUE line in a benchmark log."""
    return dict(KEY_VALUE.findall(path.read_text(encoding="utf-8")))


def mean_or_none(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def summarize_session(session: Path) -> dict[str, object]:
    """Returns one summary row for a completed ViT session."""
    log = read_log_fields(session / "benchmark.log")
    if log.get("STATUS") != "OK":
        raise ValueError("benchmark did not report STATUS=OK")
    ranges = {
        phase.name: phase
        for phase in phase_ranges(load_markers(session / "kernel_activity.csv"))
    }
    samples = read_phase_samples(
        session / "gpu_telemetry.csv", tuple(ranges.values())
    )
    measurement = ranges["measurement"]
    seconds = (measurement.end_ns - measurement.start_ns) / 1e9
    power_w = mean_or_none(
        [
            value / 1000
            for value in numeric_values(samples["measurement"], "power_mw")
        ]
    )
    idle_w = mean_or_none(
        [
            value / 1000
            for value in numeric_values(samples["cooldown"], "power_mw")
        ]
    )
    if power_w is None:
        raise ValueError("no NVML power samples in the measurement phase")
    images = int(log["MEASUREMENT_IMAGES"])
    energy_j = power_w * seconds
    row: dict[str, object] = {"session": session.name}
    row.update({field.lower(): log.get(field, "") for field in LOG_FIELDS})
    row.update(
        {
            "measurement_seconds": seconds,
            "telemetry_samples": len(samples["measurement"]),
            "mean_power_w": power_w,
            "idle_power_w": idle_w,
            "energy_j": energy_j,
            "energy_per_image_mj": 1000 * energy_j / images,
            "dynamic_energy_per_image_mj": (
                1000 * (power_w - idle_w) * seconds / images
                if idle_w is not None
                else None
            ),
            "images_per_joule": images / energy_j,
            "mean_gpu_utilization_pct": mean_or_none(
                numeric_values(samples["measurement"], "gpu_utilization_pct")
            ),
            "mean_graphics_clock_mhz": mean_or_none(
                numeric_values(samples["measurement"], "graphics_clock_mhz")
            ),
            "mean_temperature_c": mean_or_none(
                numeric_values(samples["measurement"], "temperature_c")
            ),
        }
    )
    row["images_per_second"] = float(row["images_per_second"])
    return row


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Returns mean and sample standard deviation per precision."""
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["dtype"])].append(row)
    summary = []
    for dtype, members in groups.items():
        result: dict[str, object] = {"dtype": dtype, "sessions": len(members)}
        for field in AGGREGATE_FIELDS:
            values = [float(m[field]) for m in members if m[field] is not None]
            result[f"{field}_mean"] = mean_or_none(values)
            result[f"{field}_stddev"] = (
                statistics.stdev(values) if len(values) > 1 else None
            )
        summary.append(result)
    return summary


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.study_dir / "summary"
    sessions_dir = args.study_dir / "sessions"
    if not sessions_dir.is_dir():
        raise SystemExit(f"sessions directory not found: {sessions_dir}")
    rows = []
    for session in sorted(
        path for path in sessions_dir.iterdir() if path.is_dir()
    ):
        try:
            rows.append(summarize_session(session))
        except (OSError, KeyError, ValueError) as error:
            print(f"SKIP {session.name}: {error}")
    if not rows:
        raise SystemExit("no session could be summarized")
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(output_dir / "vit_sessions.csv", rows)
    summary = aggregate(rows)
    atomic_write_csv(output_dir / "vit_precision_summary.csv", summary)
    print("dtype       img/s        W     mJ/img    img/J")
    for row in summary:
        dtype = row["dtype"]
        throughput = row["images_per_second_mean"]
        power = row["mean_power_w_mean"]
        energy = row["energy_per_image_mj_mean"]
        efficiency = row["images_per_joule_mean"]
        print(
            f"{dtype:<6} {throughput:>10.1f} {power:>8.2f} "
            f"{energy:>10.3f} {efficiency:>8.2f}"
        )
    print(f"Results: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
