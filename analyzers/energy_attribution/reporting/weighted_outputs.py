"""CSV, PNG and console output for the PM-weighted kernel energy analysis."""

import csv

from energy_attribution.io import write_csv
from energy_attribution.streaming_io import parse_kernel_row

KERNEL_ENERGY_FIELDS = [
    "event_id",
    "pid",
    "device_id",
    "context_id",
    "stream_id",
    "correlation_id",
    "start_ns",
    "end_ns",
    "duration_ns",
    "kernel_name",
    "estimated_dynamic_energy_mj",
    "pm_coverage_pct",
    "pm_energy_weight",
    "pm_weighted_energy_mj",
    "equal_share_fallback_energy_mj",
]
PLOT_FILES = ("energy_timeline.png", "utilization_energy_correlation.png")


def _energy_at(values, index):
    return values[index] if index < len(values) else 0.0


def _annotate_kernel_row(row, index, pm_values, attribution):
    """Adds the energy and PM columns for the ``index``-th valid kernel."""
    row["estimated_dynamic_energy_mj"] = _energy_at(
        attribution["estimated_energy"], index
    )
    row["pm_coverage_pct"] = pm_values[0] if pm_values is not None else ""
    row["pm_energy_weight"] = (
        pm_values[1]
        if pm_values is not None and pm_values[1] is not None
        else ""
    )
    row["pm_weighted_energy_mj"] = _energy_at(
        attribution["pm_weighted_energy"], index
    )
    row["equal_share_fallback_energy_mj"] = _energy_at(
        attribution["fallback_energy"], index
    )


def write_kernel_energy_csv(
    source_path, output_path, pm_by_event_id, attribution
):
    """Copies valid kernel rows from ``source_path`` with their energy."""
    valid_index = 0
    with source_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        with output_path.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(
                destination,
                fieldnames=KERNEL_ENERGY_FIELDS,
                extrasaction="ignore",
            )
            writer.writeheader()
            for row in reader:
                if parse_kernel_row(row) is None:
                    continue
                pm_values = pm_by_event_id.get(row.get("event_id"))
                _annotate_kernel_row(row, valid_index, pm_values, attribution)
                writer.writerow(row)
                valid_index += 1
                if valid_index % 1_000_000 == 0:
                    print(
                        f"Wrote energy for {valid_index} kernel events...",
                        flush=True,
                    )


def summary_rows(args, report):
    """Returns the metric/value rows of ``energy_summary.csv``."""
    attribution = report.attribution
    energy_correlation, power_correlation = report.correlations()
    uses_pm = bool(args.pm_kernel_metrics)
    values = {
        "analysis_start_ns": report.analysis_start,
        "analysis_end_ns": report.analysis_end,
        "idle_power_mw": args.idle_power_mw,
        "device_energy_mj": report.device_energy,
        "dynamic_energy_mj": report.dynamic_energy,
        "assigned_dynamic_energy_mj": attribution["assigned"],
        "unassigned_dynamic_energy_mj": attribution["unassigned"],
        "conservation_error_mj": report.conservation_error,
        "utilization_dynamic_energy_pearson_r": energy_correlation,
        "utilization_dynamic_power_pearson_r": power_correlation,
        "attribution_method": (
            "hybrid_pm_weighted_with_equal_share_fallback"
            if uses_pm
            else "equal_share_concurrent_kernels"
        ),
        "pm_weight_metric": args.pm_weight_metric if uses_pm else "",
        "pm_min_coverage_pct": args.pm_min_coverage_pct if uses_pm else "",
        "pm_matched_kernels": report.kernel_metadata["pm_matched"],
        "pm_eligible_kernels": report.kernel_metadata["pm_eligible"],
        "pm_weighted_energy_mj": attribution["pm_weighted"],
        "equal_share_fallback_energy_mj": attribution["fallback"],
    }
    return [{"metric": key, "value": value} for key, value in values.items()]


def write_outputs(args, report, pm_by_event_id):
    """Writes per-kernel, per-window and summary CSVs, plus optional plots."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Writing per-kernel energy results...", flush=True)
    write_kernel_energy_csv(
        args.kernels,
        args.output_dir / "kernel_energy.csv",
        pm_by_event_id,
        report.attribution,
    )
    write_csv(
        args.output_dir / "energy_windows.csv",
        report.windows,
        list(report.windows[0].keys()),
    )
    write_csv(
        args.output_dir / "energy_summary.csv",
        summary_rows(args, report),
        ["metric", "value"],
    )
    if args.plot:
        create_plots(args.output_dir, report.windows)


def _plot_energy_timeline(plt, output_dir, elapsed_ms, energy, utilization):
    figure, axis_energy = plt.subplots(figsize=(10, 5))
    axis_utilization = axis_energy.twinx()
    axis_energy.plot(
        elapsed_ms, energy, color="tab:red", marker="o", label="Energy"
    )
    axis_utilization.plot(
        elapsed_ms,
        utilization,
        color="tab:blue",
        marker=".",
        label="Kernel active time",
    )
    axis_energy.set_xlabel("Window start (ms)")
    axis_energy.set_ylabel("Energy per window (mJ)", color="tab:red")
    axis_utilization.set_ylabel("Kernel active time (%)", color="tab:blue")
    axis_energy.set_title("GPU energy and kernel activity over time")
    figure.tight_layout()
    figure.savefig(output_dir / PLOT_FILES[0], dpi=160)
    plt.close(figure)


def _plot_energy_correlation(plt, output_dir, energy, utilization):
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.scatter(utilization, energy, alpha=0.75)
    axis.set_xlabel("Kernel active time (%)")
    axis.set_ylabel("Energy per window (mJ)")
    axis.set_title("Kernel activity–energy correlation")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / PLOT_FILES[1], dpi=160)
    plt.close(figure)


def create_plots(output_dir, windows):
    try:
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    except ImportError as error:
        raise RuntimeError(
            "--plot requires matplotlib; install it with: python3 -m pip "
            "install matplotlib"
        ) from error

    elapsed_ms = [
        (row["start_ns"] - windows[0]["start_ns"]) / 1_000_000
        for row in windows
    ]
    energy = [row["dynamic_energy_mj"] for row in windows]
    utilization = [row["kernel_active_pct"] for row in windows]
    _plot_energy_timeline(plt, output_dir, elapsed_ms, energy, utilization)
    _plot_energy_correlation(plt, output_dir, energy, utilization)


def print_report(args, report):
    """Prints the console summary of the weighted energy analysis."""
    attribution = report.attribution
    kernel_count = report.kernel_metadata["count"]
    pm_eligible = report.kernel_metadata["pm_eligible"]
    assigned = attribution["assigned"]
    unassigned = attribution["unassigned"]
    pm_weighted = attribution["pm_weighted"]
    fallback = attribution["fallback"]
    energy_correlation, power_correlation = report.correlations()
    print("=== Kernel energy association ===")
    print(f"Kernels: {kernel_count}")
    print(f"Windows: {len(report.windows)}")
    print(f"Dynamic energy: {report.dynamic_energy:.6f} mJ")
    print(f"Assigned to kernels: {assigned:.6f} mJ")
    print(f"Unassigned: {unassigned:.6f} mJ")
    if args.pm_kernel_metrics:
        print(
            f"PM-weighted kernels: {pm_eligible} / {kernel_count} "
            f"(>= {args.pm_min_coverage_pct:.1f}% coverage)"
        )
        print(f"PM-weighted energy: {pm_weighted:.6f} mJ")
        print(f"Equal-share fallback energy: {fallback:.6f} mJ")
    print(f"Utilization-energy Pearson r: {energy_correlation:.6f}")
    print(f"Utilization-power Pearson r: {power_correlation:.6f}")
    print(f"Results: {args.output_dir}")
    if args.plot:
        print("Plots:", ", ".join(PLOT_FILES))
