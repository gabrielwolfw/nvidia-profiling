#!/usr/bin/env python3
"""Associate CUPTI PM Sampling metrics with CUPTI kernel activity records."""

import argparse
import csv
import math
import statistics
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate time-weighted PM metrics and PM coverage for each "
            "CUPTI kernel interval."
        )
    )
    parser.add_argument("--pm-samples", required=True, type=Path)
    parser.add_argument("--kernels", required=True, type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("pm-kernel-analysis")
    )
    parser.add_argument(
        "--min-duration-factor",
        type=float,
        default=0.1,
        help="Reject PM samples shorter than this multiple of the median.",
    )
    parser.add_argument(
        "--max-duration-factor",
        type=float,
        default=5.0,
        help="Reject PM samples longer than this multiple of the median.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate PM coverage and sampling-gap PNG plots (requires matplotlib).",
    )
    return parser.parse_args()


def read_csv_header(path):
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.reader(source)
        return next(reader, [])


def load_kernels(path):
    kernels = []
    skipped = 0
    fieldnames = read_csv_header(path)
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            try:
                start_ns = int(row["start_ns"])
                end_ns = int(row["end_ns"])
            except (KeyError, TypeError, ValueError):
                skipped += 1
                continue
            if end_ns <= start_ns:
                skipped += 1
                continue
            row["start_ns"] = start_ns
            row["end_ns"] = end_ns
            kernels.append(row)
    kernels.sort(key=lambda row: (row["start_ns"], row["end_ns"]))
    return kernels, fieldnames, skipped


def read_pm_duration_stats(path):
    durations = []
    raw_samples = 0
    invalid_samples = 0
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            raw_samples += 1
            try:
                start_ns = int(row["start_ns"])
                end_ns = int(row["end_ns"])
            except (KeyError, TypeError, ValueError):
                invalid_samples += 1
                continue
            if end_ns <= start_ns:
                invalid_samples += 1
                continue
            durations.append(end_ns - start_ns)
    if not durations:
        raise ValueError("No valid PM sample intervals were found")
    return raw_samples, invalid_samples, statistics.median(durations)


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_plots(kernels, output_rows, gap_events, gap_threshold_ns, output_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "--plot requires matplotlib; install it with: "
            "python3 -m pip install matplotlib"
        ) from error

    origin_ns = kernels[0]["start_ns"]
    kernel_time_s = [
        (kernel["start_ns"] - origin_ns) / 1_000_000_000
        for kernel in kernels
    ]
    coverage_pct = [float(row["pm_coverage_pct"]) for row in output_rows]

    figure, axis = plt.subplots(figsize=(12, 4.8))
    axis.scatter(kernel_time_s, coverage_pct, s=7, alpha=0.65, linewidths=0)
    axis.axhline(99.9, color="tab:green", linestyle="--", linewidth=1,
                 label="99.9% coverage target")
    axis.set_title("PM coverage by kernel")
    axis.set_xlabel("Kernel start relative to first kernel (s)")
    axis.set_ylabel("PM coverage (%)")
    axis.set_ylim(-2, 102)
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "pm_coverage_timeline.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 4.8))
    threshold_ms = gap_threshold_ns / 1_000_000
    if gap_events:
        gap_time_s = [
            ((start_ns + end_ns) / 2 - origin_ns) / 1_000_000_000
            for start_ns, end_ns, _ in gap_events
        ]
        gap_duration_ms = [duration_ns / 1_000_000 for _, _, duration_ns in gap_events]
        axis.scatter(gap_time_s, gap_duration_ms, s=20, color="tab:red")
        axis.set_yscale("log")
    else:
        axis.text(0.5, 0.5, "No PM gaps above the reporting threshold",
                  transform=axis.transAxes, ha="center", va="center")
    axis.axhline(threshold_ms, color="tab:orange", linestyle="--", linewidth=1,
                 label=f"Reporting threshold ({threshold_ms:.3f} ms)")
    axis.set_title("PM sampling gaps")
    axis.set_xlabel("Gap midpoint relative to first kernel (s)")
    axis.set_ylabel("Gap duration (ms)")
    axis.grid(True, alpha=0.25, which="both")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "pm_sampling_gaps.png", dpi=160)
    plt.close(figure)


def main():
    args = parse_args()
    if args.min_duration_factor <= 0:
        raise ValueError("--min-duration-factor must be greater than zero")
    if args.max_duration_factor < args.min_duration_factor:
        raise ValueError(
            "--max-duration-factor must be at least --min-duration-factor"
        )

    kernels, kernel_fields, skipped_kernels = load_kernels(args.kernels)
    if not kernels:
        raise ValueError("No valid kernel intervals were found")

    pm_fields = read_csv_header(args.pm_samples)
    required_fields = {"start_ns", "end_ns"}
    missing_fields = required_fields - set(pm_fields)
    if missing_fields:
        raise ValueError(f"PM CSV is missing columns: {sorted(missing_fields)}")

    pm_metrics = [
        field
        for field in pm_fields
        if field not in {"sample", "start_ns", "end_ns", "duration_ns"}
    ]
    if not pm_metrics:
        raise ValueError("PM CSV contains no metric columns")

    raw_samples, invalid_samples, median_duration_ns = read_pm_duration_stats(
        args.pm_samples
    )
    min_duration_ns = median_duration_ns * args.min_duration_factor
    max_duration_ns = median_duration_ns * args.max_duration_factor

    accumulators = [
        {
            "sample_count": 0,
            "covered_ns": 0,
            "last_covered_end_ns": 0,
            "weighted_duration_ns": 0,
            "metric_sums": {metric: 0.0 for metric in pm_metrics},
        }
        for _ in kernels
    ]

    pm_valid_samples = 0
    pm_rejected_duration = 0
    pm_rejected_values = 0
    gap_events = []
    previous_valid_end_ns = None
    kernel_index = 0

    with args.pm_samples.open(newline="", encoding="utf-8") as source:
        for sample in csv.DictReader(source):
            try:
                sample_start = int(sample["start_ns"])
                sample_end = int(sample["end_ns"])
            except (KeyError, TypeError, ValueError):
                continue

            sample_duration = sample_end - sample_start
            if sample_duration < min_duration_ns or sample_duration > max_duration_ns:
                pm_rejected_duration += 1
                continue

            if previous_valid_end_ns is not None and sample_start > previous_valid_end_ns:
                gap_duration_ns = sample_start - previous_valid_end_ns
                if gap_duration_ns > max_duration_ns:
                    gap_events.append(
                        (previous_valid_end_ns, sample_start, gap_duration_ns)
                    )
            previous_valid_end_ns = (
                sample_end
                if previous_valid_end_ns is None
                else max(previous_valid_end_ns, sample_end)
            )

            while (
                kernel_index < len(kernels)
                and kernels[kernel_index]["end_ns"] <= sample_start
            ):
                kernel_index += 1

            if (
                kernel_index == len(kernels)
                or kernels[kernel_index]["start_ns"] >= sample_end
            ):
                pm_valid_samples += 1
                continue

            try:
                metric_values = {
                    metric: float(sample[metric]) for metric in pm_metrics
                }
            except (KeyError, TypeError, ValueError):
                pm_rejected_values += 1
                continue

            if not all(math.isfinite(value) for value in metric_values.values()):
                pm_rejected_values += 1
                continue

            pm_valid_samples += 1
            candidate_index = kernel_index
            while (
                candidate_index < len(kernels)
                and kernels[candidate_index]["start_ns"] < sample_end
            ):
                kernel = kernels[candidate_index]
                overlap_start = max(sample_start, kernel["start_ns"])
                overlap_end = min(sample_end, kernel["end_ns"])
                overlap_ns = overlap_end - overlap_start
                if overlap_ns > 0:
                    accumulator = accumulators[candidate_index]
                    accumulator["sample_count"] += 1
                    accumulator["weighted_duration_ns"] += overlap_ns
                    for metric, value in metric_values.items():
                        accumulator["metric_sums"][metric] += value * overlap_ns

                    uncovered_start = max(
                        overlap_start, accumulator["last_covered_end_ns"]
                    )
                    if overlap_end > uncovered_start:
                        accumulator["covered_ns"] += overlap_end - uncovered_start
                    accumulator["last_covered_end_ns"] = max(
                        accumulator["last_covered_end_ns"], overlap_end
                    )
                candidate_index += 1

    output_rows = []
    covered_kernels = 0
    complete_coverage_kernels = 0
    for kernel, accumulator in zip(kernels, accumulators):
        duration_ns = kernel["end_ns"] - kernel["start_ns"]
        coverage_pct = 100.0 * accumulator["covered_ns"] / duration_ns
        row = dict(kernel)
        row["pm_sample_count"] = accumulator["sample_count"]
        row["pm_covered_ns"] = accumulator["covered_ns"]
        row["pm_coverage_pct"] = min(100.0, coverage_pct)
        if accumulator["weighted_duration_ns"] > 0:
            covered_kernels += 1
            if coverage_pct >= 99.9:
                complete_coverage_kernels += 1
            for metric in pm_metrics:
                row[metric] = (
                    accumulator["metric_sums"][metric]
                    / accumulator["weighted_duration_ns"]
                )
        else:
            for metric in pm_metrics:
                row[metric] = ""
        output_rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_fields = kernel_fields + [
        "pm_sample_count",
        "pm_covered_ns",
        "pm_coverage_pct",
    ] + pm_metrics
    write_csv(
        args.output_dir / "kernel_pm_metrics.csv",
        output_rows,
        output_fields,
    )

    summary_rows = [
        {"metric": "kernels", "value": len(kernels)},
        {"metric": "kernels_skipped", "value": skipped_kernels},
        {"metric": "kernels_with_pm_coverage", "value": covered_kernels},
        {
            "metric": "kernels_with_at_least_99_9_pct_coverage",
            "value": complete_coverage_kernels,
        },
        {"metric": "pm_samples_raw", "value": raw_samples},
        {"metric": "pm_samples_invalid", "value": invalid_samples},
        {"metric": "pm_samples_valid", "value": pm_valid_samples},
        {
            "metric": "pm_samples_rejected_duration",
            "value": pm_rejected_duration,
        },
        {"metric": "pm_samples_rejected_values", "value": pm_rejected_values},
        {"metric": "pm_median_duration_ns", "value": median_duration_ns},
        {"metric": "pm_min_duration_ns", "value": min_duration_ns},
        {"metric": "pm_max_duration_ns", "value": max_duration_ns},
    ]
    write_csv(
        args.output_dir / "pm_kernel_summary.csv",
        summary_rows,
        ["metric", "value"],
    )

    if args.plot:
        generate_plots(
            kernels,
            output_rows,
            gap_events,
            max_duration_ns,
            args.output_dir,
        )

    print("=== PM Sampling kernel association ===")
    print(f"Kernels: {len(kernels)}")
    print(f"Kernels with PM coverage: {covered_kernels}")
    print(f"Kernels with >=99.9% PM coverage: {complete_coverage_kernels}")
    print(f"PM samples: {pm_valid_samples} valid / {raw_samples} raw")
    print(f"Median PM interval: {median_duration_ns:.0f} ns")
    print(f"Results: {args.output_dir}")
    if args.plot:
        print("Plots: pm_coverage_timeline.png, pm_sampling_gaps.png")


if __name__ == "__main__":
    main()
