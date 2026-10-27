"""Associate CUPTI PM Sampling metrics with CUPTI kernel activity records."""

import argparse
from pathlib import Path

from energy_attribution.io import write_csv
from energy_attribution.pm_association import (
    associate_pm_samples,
    build_kernel_rows,
    load_kernels,
    pm_metric_columns,
    read_pm_duration_stats,
)
from energy_attribution.reporting.pm_plots import PLOT_FILES, generate_plots


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
        help="Generate PM coverage and sampling-gap PNG plots (requires "
        "matplotlib).",
    )
    return parser.parse_args()


def validate_args(args):
    if args.min_duration_factor <= 0:
        raise ValueError("--min-duration-factor must be greater than zero")
    if args.max_duration_factor < args.min_duration_factor:
        raise ValueError(
            "--max-duration-factor must be at least --min-duration-factor"
        )


def summary_rows(counts, stats, duration_stats, duration_limits_ns):
    """Returns the metric/value rows of ``pm_kernel_summary.csv``."""
    kernels, skipped_kernels, covered_kernels, complete_kernels = counts
    raw_samples, invalid_samples, median_duration_ns = duration_stats
    values = {
        "kernels": kernels,
        "kernels_skipped": skipped_kernels,
        "kernels_with_pm_coverage": covered_kernels,
        "kernels_with_at_least_99_9_pct_coverage": complete_kernels,
        "pm_samples_raw": raw_samples,
        "pm_samples_invalid": invalid_samples,
        "pm_samples_valid": stats.valid,
        "pm_samples_rejected_duration": stats.rejected_duration,
        "pm_samples_rejected_values": stats.rejected_values,
        "pm_median_duration_ns": median_duration_ns,
        "pm_min_duration_ns": duration_limits_ns[0],
        "pm_max_duration_ns": duration_limits_ns[1],
    }
    return [{"metric": key, "value": value} for key, value in values.items()]


def print_report(args, summary):
    values = {row["metric"]: row["value"] for row in summary}
    complete = values["kernels_with_at_least_99_9_pct_coverage"]
    valid_samples = values["pm_samples_valid"]
    raw_samples = values["pm_samples_raw"]
    median_ns = values["pm_median_duration_ns"]
    print("=== PM Sampling kernel association ===")
    print("Kernels:", values["kernels"])
    print("Kernels with PM coverage:", values["kernels_with_pm_coverage"])
    print(f"Kernels with >=99.9% PM coverage: {complete}")
    print(f"PM samples: {valid_samples} valid / {raw_samples} raw")
    print(f"Median PM interval: {median_ns:.0f} ns")
    print(f"Results: {args.output_dir}")
    if args.plot:
        print("Plots:", ", ".join(PLOT_FILES))


def main():
    args = parse_args()
    validate_args(args)
    kernels, kernel_fields, skipped_kernels = load_kernels(args.kernels)
    if not kernels:
        raise ValueError("No valid kernel intervals were found")
    pm_metrics = pm_metric_columns(args.pm_samples)
    duration_stats = read_pm_duration_stats(args.pm_samples)
    median_duration_ns = duration_stats[2]
    duration_limits_ns = (
        median_duration_ns * args.min_duration_factor,
        median_duration_ns * args.max_duration_factor,
    )
    accumulators, stats = associate_pm_samples(
        args.pm_samples, kernels, pm_metrics, duration_limits_ns
    )
    output_rows, covered, complete = build_kernel_rows(
        kernels, accumulators, pm_metrics
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_fields = (
        kernel_fields
        + ["pm_sample_count", "pm_covered_ns", "pm_coverage_pct"]
        + pm_metrics
    )
    write_csv(
        args.output_dir / "kernel_pm_metrics.csv", output_rows, output_fields
    )
    summary = summary_rows(
        (len(kernels), skipped_kernels, covered, complete),
        stats,
        duration_stats,
        duration_limits_ns,
    )
    write_csv(
        args.output_dir / "pm_kernel_summary.csv", summary, ["metric", "value"]
    )
    if args.plot:
        generate_plots(
            kernels,
            output_rows,
            stats.gap_events,
            duration_limits_ns[1],
            args.output_dir,
        )
    print_report(args, summary)


if __name__ == "__main__":
    main()
