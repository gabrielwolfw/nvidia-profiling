#!/usr/bin/env python3
"""Summarize phased GPU power-sweep sessions for model training."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PHASE_MARKERS = {
    "warmup": "benchmark_warmup_start_marker",
    "measurement": "benchmark_measurement_start_marker",
    "cooldown": "benchmark_cooldown_start_marker",
    "complete": "benchmark_complete_marker",
}
TELEMETRY_FIELDS = (
    "power_mw",
    "temperature_c",
    "gpu_utilization_pct",
    "memory_utilization_pct",
    "graphics_clock_mhz",
)


@dataclass(frozen=True)
class Marker:
    """One phase-marker kernel interval in the CUPTI timestamp domain."""

    start_ns: int
    end_ns: int


@dataclass(frozen=True)
class PhaseRange:
    """Inclusive/exclusive timestamp range for one benchmark phase."""

    name: str
    start_ns: int
    end_ns: int


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--target-step-w",
        type=float,
        help=(
            "Select pilot duty cycles nearest to uniformly spaced power "
            "targets and write recommended_power_levels.csv."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return failure if any completed session cannot be summarized.",
    )
    return parser.parse_args()


def require_fields(reader: csv.DictReader, required: Iterable[str], label: str) -> None:
    """Validate required CSV columns."""
    missing = set(required) - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def load_markers(path: Path) -> dict[str, Marker]:
    """Find exactly one ordered boundary marker for every phase."""
    found: dict[str, list[Marker]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        require_fields(reader, ("kernel_name", "start_ns", "end_ns"), str(path))
        for row in reader:
            name = row["kernel_name"]
            for phase, marker_name in PHASE_MARKERS.items():
                if marker_name in name:
                    found[phase].append(
                        Marker(int(row["start_ns"]), int(row["end_ns"]))
                    )

    missing = set(PHASE_MARKERS) - set(found)
    if missing:
        raise ValueError(f"missing phase markers: {sorted(missing)}")
    duplicates = {phase: len(items) for phase, items in found.items() if len(items) != 1}
    if duplicates:
        raise ValueError(f"phase markers must appear exactly once: {duplicates}")
    markers = {phase: items[0] for phase, items in found.items()}
    ordered_starts = [markers[name].start_ns for name in PHASE_MARKERS]
    if ordered_starts != sorted(ordered_starts):
        raise ValueError("phase markers are not chronologically ordered")
    return markers


def phase_ranges(markers: dict[str, Marker]) -> tuple[PhaseRange, ...]:
    """Derive non-overlapping phase ranges without marker-kernel activity."""
    return (
        PhaseRange("warmup", markers["warmup"].end_ns, markers["measurement"].start_ns),
        PhaseRange(
            "measurement",
            markers["measurement"].end_ns,
            markers["cooldown"].start_ns,
        ),
        PhaseRange(
            "cooldown", markers["cooldown"].end_ns, markers["complete"].start_ns
        ),
    )


def optional_float(value: str | None) -> float | None:
    """Parse an optional numeric CSV field."""
    if value is None or value.strip() == "":
        return None
    return float(value)


def phase_for_timestamp(timestamp_ns: int, ranges: tuple[PhaseRange, ...]) -> str | None:
    """Return the phase containing a telemetry timestamp."""
    for phase in ranges:
        if phase.start_ns <= timestamp_ns < phase.end_ns:
            return phase.name
    return None


def read_phase_samples(
    path: Path, ranges: tuple[PhaseRange, ...]
) -> dict[str, list[dict[str, float | int | None]]]:
    """Partition NVML telemetry into marker-delimited phases."""
    samples: dict[str, list[dict[str, float | int | None]]] = {
        item.name: [] for item in ranges
    }
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        require_fields(reader, ("timestamp_ns", *TELEMETRY_FIELDS), str(path))
        for row in reader:
            timestamp_ns = int(row["timestamp_ns"])
            phase = phase_for_timestamp(timestamp_ns, ranges)
            if phase is None:
                continue
            parsed: dict[str, float | int | None] = {"timestamp_ns": timestamp_ns}
            for field in TELEMETRY_FIELDS:
                parsed[field] = optional_float(row[field])
            samples[phase].append(parsed)
    return samples


def numeric_values(
    samples: list[dict[str, float | int | None]], field: str
) -> list[float]:
    """Extract non-missing numeric values from phase samples."""
    return [float(sample[field]) for sample in samples if sample[field] is not None]


def summarize_values(values: list[float], prefix: str) -> dict[str, float | int | None]:
    """Return stable descriptive statistics for one metric."""
    if not values:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_mean": None,
            f"{prefix}_median": None,
            f"{prefix}_stddev": None,
            f"{prefix}_minimum": None,
            f"{prefix}_maximum": None,
        }
    return {
        f"{prefix}_count": len(values),
        f"{prefix}_mean": statistics.fmean(values),
        f"{prefix}_median": statistics.median(values),
        f"{prefix}_stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
        f"{prefix}_minimum": min(values),
        f"{prefix}_maximum": max(values),
    }


def summarize_phase(
    condition: dict[str, object], phase_range: PhaseRange,
    samples: list[dict[str, float | int | None]],
) -> dict[str, object]:
    """Create one phase-level summary row."""
    row: dict[str, object] = {
        "sequence": condition["sequence"],
        "repetition": condition["repetition"],
        "condition_id": condition["condition_id"],
        "workload_name": condition["workload_name"],
        "workload_kind": condition["workload_kind"],
        "duty_cycle_pct": condition["duty_cycle_pct"],
        "phase": phase_range.name,
        "phase_start_ns": phase_range.start_ns,
        "phase_end_ns": phase_range.end_ns,
        "phase_duration_seconds": (
            phase_range.end_ns - phase_range.start_ns
        ) / 1_000_000_000,
        "telemetry_samples": len(samples),
    }
    for field in TELEMETRY_FIELDS:
        row.update(summarize_values(numeric_values(samples, field), field))
    return row


def atomic_write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a non-empty CSV atomically."""
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def discover_sessions(study_dir: Path) -> list[Path]:
    """Return completed result directories in execution order."""
    sessions_dir = study_dir / "sessions"
    if not sessions_dir.is_dir():
        raise ValueError(f"sessions directory not found: {sessions_dir}")
    return sorted(path for path in sessions_dir.iterdir() if path.is_dir())


def load_condition(path: Path) -> dict[str, object]:
    """Load one runner-generated condition document."""
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"condition must be a JSON object: {path}")
    return value


def sample_identity(
    condition: dict[str, object], phase_name: str
) -> dict[str, object]:
    """Return stable condition columns shared by every telemetry sample."""
    return {
        "sequence": condition["sequence"],
        "repetition": condition["repetition"],
        "condition_id": condition["condition_id"],
        "workload_name": condition["workload_name"],
        "workload_kind": condition["workload_kind"],
        "duty_cycle_pct": condition["duty_cycle_pct"],
        "phase": phase_name,
    }


def summarize_session(
    session: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Summarize one completed session and label each telemetry sample."""
    condition = load_condition(session / "condition.json")
    if condition.get("status") != "completed":
        return [], []
    ranges = phase_ranges(load_markers(session / "kernel_activity.csv"))
    samples_by_phase = read_phase_samples(session / "gpu_telemetry.csv", ranges)
    phase_rows = []
    sample_rows = []
    for phase_range in ranges:
        samples = samples_by_phase[phase_range.name]
        if not samples:
            raise ValueError(f"no telemetry samples in {phase_range.name} phase")
        phase_rows.append(summarize_phase(condition, phase_range, samples))
        identity = sample_identity(condition, phase_range.name)
        sample_rows.extend({**identity, **sample} for sample in samples)
    return phase_rows, sample_rows


def summarize_study(
    study_dir: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, str]]]:
    """Summarize every valid session and retain per-sample phase labels."""
    phase_rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    for session in discover_sessions(study_dir):
        try:
            session_phases, session_samples = summarize_session(session)
            phase_rows.extend(session_phases)
            sample_rows.extend(session_samples)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            errors.append({"session": session.name, "error": str(error)})
    if not phase_rows:
        raise ValueError("no completed session could be summarized")
    return phase_rows, sample_rows, errors


def aggregate_measurements(
    phase_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Aggregate stable-phase session medians across repetitions."""
    groups: dict[tuple[str, str, float], list[dict[str, object]]] = defaultdict(list)
    for row in phase_rows:
        if row["phase"] == "measurement":
            key = (
                str(row["workload_name"]),
                str(row["workload_kind"]),
                float(row["duty_cycle_pct"]),
            )
            groups[key].append(row)

    return [
        aggregate_group(name, kind, duty_cycle, rows)
        for (name, kind, duty_cycle), rows in sorted(groups.items())
    ]


def available_values(rows: list[dict[str, object]], field: str) -> list[float]:
    """Return numeric group values while preserving unsupported metrics."""
    return [float(row[field]) for row in rows if row[field] is not None]


def aggregate_group(
    name: str,
    kind: str,
    duty_cycle: float,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    """Aggregate repeated stable phases for one workload level."""
    power_medians = [
        value / 1000.0 for value in available_values(rows, "power_mw_median")
    ]
    if not power_medians:
        raise ValueError(f"no valid power samples for {name} at {duty_cycle}%")
    temperatures = available_values(rows, "temperature_c_mean")
    clocks = available_values(rows, "graphics_clock_mhz_mean")
    return {
        "workload_name": name,
        "workload_kind": kind,
        "duty_cycle_pct": duty_cycle,
        "repetitions": len(rows),
        "power_median_mean_w": statistics.fmean(power_medians),
        "power_median_stddev_w": (
            statistics.stdev(power_medians) if len(rows) > 1 else 0.0
        ),
        "power_median_minimum_w": min(power_medians),
        "power_median_maximum_w": max(power_medians),
        "temperature_mean_c": (
            statistics.fmean(temperatures) if temperatures else None
        ),
        "graphics_clock_mean_mhz": statistics.fmean(clocks) if clocks else None,
    }


def plot_summary(output_dir: Path, summary_rows: list[dict[str, object]]) -> None:
    """Plot measured power against requested duty cycle by workload."""
    try:
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    except ImportError as error:
        raise RuntimeError("--plot requires matplotlib") from error

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in summary_rows:
        grouped[str(row["workload_name"])].append(row)
    figure, axis = plt.subplots(figsize=(10, 6))
    for name, rows in grouped.items():
        rows.sort(key=lambda row: float(row["duty_cycle_pct"]))
        axis.errorbar(
            [float(row["duty_cycle_pct"]) for row in rows],
            [float(row["power_median_mean_w"]) for row in rows],
            yerr=[float(row["power_median_stddev_w"]) for row in rows],
            marker="o",
            capsize=3,
            label=name,
        )
    axis.set_xlabel("Requested duty cycle (%)")
    axis.set_ylabel("Stable-phase median power (W)")
    axis.set_title("GPU power-sweep calibration")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "power_sweep_calibration.png", dpi=180)
    plt.close(figure)


def recommend_power_levels(
    summary_rows: list[dict[str, object]], target_step_w: float
) -> list[dict[str, object]]:
    """Select measured pilot points nearest to a uniform power grid."""
    if target_step_w <= 0.0:
        raise ValueError("--target-step-w must be greater than zero")
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in summary_rows:
        if row["workload_kind"] != "idle" and float(row["duty_cycle_pct"]) > 0.0:
            grouped[str(row["workload_name"])].append(row)

    recommendations = []
    for workload_name, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: float(row["power_median_mean_w"]))
        minimum = float(rows[0]["power_median_mean_w"])
        maximum = float(rows[-1]["power_median_mean_w"])
        targets = []
        target = minimum
        while target <= maximum:
            targets.append(target)
            target += target_step_w
        if not targets or targets[-1] < maximum:
            targets.append(maximum)

        selected: dict[float, tuple[dict[str, object], float]] = {}
        for nominal_target in targets:
            nearest = nearest_power_row(rows, nominal_target)
            selected[float(nearest["duty_cycle_pct"])] = (
                nearest,
                nominal_target,
            )
        for duty_cycle, (row, nominal_target) in sorted(selected.items()):
            recommendations.append(
                {
                    "workload_name": workload_name,
                    "workload_kind": row["workload_kind"],
                    "duty_cycle_pct": duty_cycle,
                    "pilot_power_w": row["power_median_mean_w"],
                    "nominal_target_power_w": nominal_target,
                }
            )
    if not recommendations:
        raise ValueError("no non-idle pilot levels are available")
    return recommendations


def nearest_power_row(
    rows: list[dict[str, object]], target_power_w: float
) -> dict[str, object]:
    """Return the measured pilot point closest to one power target."""
    return min(
        rows,
        key=lambda row: abs(
            float(row["power_median_mean_w"]) - target_power_w
        ),
    )


def main() -> int:
    """Program entry point."""
    args = parse_args()
    output_dir = args.output_dir or args.study_dir / "summary"
    try:
        phase_rows, sample_rows, errors = summarize_study(args.study_dir)
        summary_rows = aggregate_measurements(phase_rows)
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_csv(output_dir / "phase_summary.csv", phase_rows)
        atomic_write_csv(output_dir / "phase_samples.csv", sample_rows)
        atomic_write_csv(output_dir / "power_level_summary.csv", summary_rows)
        if args.target_step_w is not None:
            recommendations = recommend_power_levels(
                summary_rows, args.target_step_w
            )
            atomic_write_csv(
                output_dir / "recommended_power_levels.csv", recommendations
            )
        with (output_dir / "analysis_errors.json").open(
            "w", encoding="utf-8"
        ) as destination:
            json.dump(errors, destination, indent=2)
            destination.write("\n")
        if args.plot:
            plot_summary(output_dir, summary_rows)
        print(f"Valid sessions: {len(phase_rows) // 3}")
        print(f"Invalid sessions: {len(errors)}")
        print(f"Power levels: {len(summary_rows)}")
        print(f"Results: {output_dir}")
        return 1 if args.strict and errors else 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
