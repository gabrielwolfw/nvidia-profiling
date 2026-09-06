#!/usr/bin/env python3
"""Associate device-level NVML power samples with CUPTI kernel intervals."""

import argparse
import csv
import math
from pathlib import Path


NS_PER_SECOND = 1_000_000_000

def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate per-kernel energy from synchronized NVML and CUPTI timestamps."
    )
    parser.add_argument("--telemetry", required=True, type=Path)
    parser.add_argument("--kernels", required=True, type=Path)
    parser.add_argument(
        "--idle-power-mw",
        type=float,
        default=0.0,
        help=(
            "Optional idle baseline. The default 0 attributes total measured "
            "device energy; a positive value estimates dynamic energy."
        ),
    )
    parser.add_argument("--window-ms", type=float, default=200.0)
    parser.add_argument("--output-dir", type=Path, default=Path("energy-analysis"))
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate PNG plots (requires matplotlib).",
    )
    return parser.parse_args()


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def load_telemetry(path):
    samples = []
    skipped = 0
    for row in read_csv(path):
        try:
            samples.append((int(row["timestamp_ns"]), float(row["power_mw"])))
        except (KeyError, TypeError, ValueError):
            skipped += 1
    if skipped:
        print(f"WARNING: Skipped {skipped} invalid telemetry rows.")
    samples.sort()
    deduplicated = []
    for timestamp, power in samples:
        if deduplicated and timestamp == deduplicated[-1][0]:
            deduplicated[-1] = (timestamp, power)
        else:
            deduplicated.append((timestamp, power))
    if len(deduplicated) < 2:
        raise ValueError("At least two distinct telemetry timestamps are required")
    return deduplicated


def load_kernels(path):
    kernels = []
    skipped = 0
    for row in read_csv(path):
        try:
            start = int(row["start_ns"])
            end = int(row["end_ns"])
        except (KeyError, TypeError, ValueError):
            skipped += 1
            continue
        if end <= start:
            skipped += 1
            continue
        row["start_ns"] = start
        row["end_ns"] = end
        row["estimated_dynamic_energy_mj"] = 0.0
        kernels.append(row)
    if skipped:
        print(f"WARNING: Skipped {skipped} invalid kernel rows.")
    kernels.sort(key=lambda row: row["start_ns"])
    return kernels


def interpolated_power(timestamp, left, right, idle_power_mw):
    left_t, left_p = left
    right_t, right_p = right
    fraction = (timestamp - left_t) / (right_t - left_t)
    device_power = left_p + fraction * (right_p - left_p)
    return device_power, max(0.0, device_power - idle_power_mw)


def interval_energy_mj(start, end, left, right, idle_power_mw):
    start_device, start_dynamic = interpolated_power(
        start, left, right, idle_power_mw
    )
    end_device, end_dynamic = interpolated_power(
        end, left, right, idle_power_mw
    )
    duration_seconds = (end - start) / NS_PER_SECOND
    return (
        0.5 * (start_device + end_device) * duration_seconds,
        0.5 * (start_dynamic + end_dynamic) * duration_seconds,
    )


def integrate_range(samples, start, end, idle_power_mw):
    device_energy = 0.0
    dynamic_energy = 0.0
    for left, right in zip(samples, samples[1:]):
        overlap_start = max(start, left[0])
        overlap_end = min(end, right[0])
        if overlap_start >= overlap_end:
            continue
        device, dynamic = interval_energy_mj(
            overlap_start, overlap_end, left, right, idle_power_mw
        )
        device_energy += device
        dynamic_energy += dynamic
    return device_energy, dynamic_energy


def union_duration_ns(kernels, start, end):
    intervals = []
    for kernel in kernels:
        overlap_start = max(start, kernel["start_ns"])
        overlap_end = min(end, kernel["end_ns"])
        if overlap_start < overlap_end:
            intervals.append((overlap_start, overlap_end))
    if not intervals:
        return 0
    intervals.sort()
    total = 0
    current_start, current_end = intervals[0]
    for interval_start, interval_end in intervals[1:]:
        if interval_start <= current_end:
            current_end = max(current_end, interval_end)
        else:
            total += current_end - current_start
            current_start, current_end = interval_start, interval_end
    return total + current_end - current_start


def attribute_energy(samples, kernels, start, end, idle_power_mw):
    assigned = 0.0
    unassigned = 0.0
    for left, right in zip(samples, samples[1:]):
        segment_start = max(start, left[0])
        segment_end = min(end, right[0])
        if segment_start >= segment_end:
            continue

        candidates = [
            (index, kernel)
            for index, kernel in enumerate(kernels)
            if kernel["start_ns"] < segment_end and kernel["end_ns"] > segment_start
        ]
        boundaries = {segment_start, segment_end}
        for _, kernel in candidates:
            boundaries.add(max(segment_start, kernel["start_ns"]))
            boundaries.add(min(segment_end, kernel["end_ns"]))
        boundaries = sorted(boundaries)

        for sub_start, sub_end in zip(boundaries, boundaries[1:]):
            _, dynamic_energy = interval_energy_mj(
                sub_start, sub_end, left, right, idle_power_mw
            )
            midpoint = sub_start + (sub_end - sub_start) // 2
            active = [
                index
                for index, kernel in candidates
                if kernel["start_ns"] <= midpoint < kernel["end_ns"]
            ]
            if not active:
                unassigned += dynamic_energy
                continue
            share = dynamic_energy / len(active)
            for index in active:
                kernels[index]["estimated_dynamic_energy_mj"] += share
            assigned += dynamic_energy
    return assigned, unassigned


def pearson(values_x, values_y):
    finite_pairs = [
        (x, y)
        for x, y in zip(values_x, values_y)
        if math.isfinite(x) and math.isfinite(y)
    ]
    values_x = [pair[0] for pair in finite_pairs]
    values_y = [pair[1] for pair in finite_pairs]
    if len(values_x) < 2:
        return math.nan
    mean_x = sum(values_x) / len(values_x)
    mean_y = sum(values_y) / len(values_y)
    centered_x = [value - mean_x for value in values_x]
    centered_y = [value - mean_y for value in values_y]
    denominator = math.sqrt(
        sum(value * value for value in centered_x)
        * sum(value * value for value in centered_y)
    )
    if denominator == 0:
        return math.nan
    return sum(x * y for x, y in zip(centered_x, centered_y)) / denominator


def build_windows(samples, kernels, start, end, idle_power_mw, window_ns):
    windows = []
    window_start = start
    index = 0
    while window_start < end:
        window_end = min(window_start + window_ns, end)
        duration_ns = window_end - window_start
        active_ns = union_duration_ns(kernels, window_start, window_end)
        device_energy, dynamic_energy = integrate_range(
            samples, window_start, window_end, idle_power_mw
        )
        duration_seconds = duration_ns / NS_PER_SECOND
        windows.append(
            {
                "window": index,
                "start_ns": window_start,
                "end_ns": window_end,
                "duration_ms": duration_ns / 1_000_000,
                "kernel_active_ms": active_ns / 1_000_000,
                "kernel_active_pct": 100.0 * active_ns / duration_ns,
                "device_energy_mj": device_energy,
                "dynamic_energy_mj": dynamic_energy,
                "avg_device_power_mw": device_energy / duration_seconds,
                "avg_dynamic_power_mw": dynamic_energy / duration_seconds,
            }
        )
        index += 1
        window_start = window_end
    return windows


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def create_plots(output_dir, windows):
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "--plot requires matplotlib; install it with: python3 -m pip install matplotlib"
        ) from error

    elapsed_ms = [
        (row["start_ns"] - windows[0]["start_ns"]) / 1_000_000
        for row in windows
    ]
    energy = [row["dynamic_energy_mj"] for row in windows]
    utilization = [row["kernel_active_pct"] for row in windows]

    figure, axis_energy = plt.subplots(figsize=(10, 5))
    axis_utilization = axis_energy.twinx()
    axis_energy.plot(elapsed_ms, energy, color="tab:red", marker="o", label="Energy")
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
    figure.savefig(output_dir / "energy_timeline.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 5))
    axis.scatter(utilization, energy, alpha=0.75)
    axis.set_xlabel("Kernel active time (%)")
    axis.set_ylabel("Energy per window (mJ)")
    axis.set_title("Kernel activity–energy correlation")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "utilization_energy_correlation.png", dpi=160)
    plt.close(figure)

def main():
    args = parse_args()
    if args.idle_power_mw < 0:
        raise ValueError("--idle-power-mw cannot be negative")
    if args.window_ms <= 0:
        raise ValueError("--window-ms must be greater than zero")
    window_ns = int(args.window_ms * 1_000_000)
    if window_ns < 1:
        raise ValueError("--window-ms must be at least one nanosecond")

    samples = load_telemetry(args.telemetry)
    kernels = load_kernels(args.kernels)
    if kernels:
        analysis_start = max(samples[0][0], kernels[0]["start_ns"])
        last_kernel_end = max(kernel["end_ns"] for kernel in kernels)
        kernel_span_ns = max(0, last_kernel_end - analysis_start)
        full_window_end = analysis_start + (
            (kernel_span_ns + window_ns - 1) // window_ns
        ) * window_ns
        analysis_end = min(samples[-1][0], full_window_end)
        if analysis_end < full_window_end:
            print(
                "WARNING: Telemetry does not cover the complete final window; "
                "capture a new run with additional tail sampling."
            )
    else:
        print(
            "WARNING: No valid kernel intervals were found; "
            "analyzing the complete NVML timeline with 0% kernel activity."
        )
        analysis_start = samples[0][0]
        analysis_end = samples[-1][0]
    if analysis_start >= analysis_end:
        raise ValueError("Telemetry and kernel timelines do not overlap")

    device_energy, dynamic_energy = integrate_range(
        samples, analysis_start, analysis_end, args.idle_power_mw
    )
    if kernels:
        assigned, unassigned = attribute_energy(
            samples, kernels, analysis_start, analysis_end, args.idle_power_mw
        )
    else:
        assigned = 0.0
        unassigned = dynamic_energy
    windows = build_windows(
        samples,
        kernels,
        analysis_start,
        analysis_end,
        args.idle_power_mw,
        window_ns,
    )
    utilization = [row["kernel_active_pct"] for row in windows]
    dynamic_window_energy = [row["dynamic_energy_mj"] for row in windows]
    dynamic_power = [row["avg_dynamic_power_mw"] for row in windows]
    energy_correlation = pearson(utilization, dynamic_window_energy)
    power_correlation = pearson(utilization, dynamic_power)
    conservation_error = dynamic_energy - assigned - unassigned

    args.output_dir.mkdir(parents=True, exist_ok=True)
    kernel_fields = [
        "event_id", "pid", "device_id", "context_id", "stream_id",
        "correlation_id", "start_ns", "end_ns", "duration_ns",
        "kernel_name", "estimated_dynamic_energy_mj",
    ]
    window_fields = list(windows[0].keys())
    write_csv(args.output_dir / "kernel_energy.csv", kernels, kernel_fields)
    write_csv(args.output_dir / "energy_windows.csv", windows, window_fields)

    summary = [
        {"metric": "analysis_start_ns", "value": analysis_start},
        {"metric": "analysis_end_ns", "value": analysis_end},
        {"metric": "idle_power_mw", "value": args.idle_power_mw},
        {"metric": "device_energy_mj", "value": device_energy},
        {"metric": "dynamic_energy_mj", "value": dynamic_energy},
        {"metric": "assigned_dynamic_energy_mj", "value": assigned},
        {"metric": "unassigned_dynamic_energy_mj", "value": unassigned},
        {"metric": "conservation_error_mj", "value": conservation_error},
        {"metric": "utilization_dynamic_energy_pearson_r", "value": energy_correlation},
        {"metric": "utilization_dynamic_power_pearson_r", "value": power_correlation},
        {"metric": "attribution_method", "value": "equal_share_concurrent_kernels"},
    ]
    write_csv(args.output_dir / "energy_summary.csv", summary, ["metric", "value"])
    if args.plot:
        create_plots(args.output_dir, windows)

    print("=== Kernel energy association ===")
    print(f"Kernels: {len(kernels)}")
    print(f"Windows: {len(windows)}")
    print(f"Dynamic energy: {dynamic_energy:.6f} mJ")
    print(f"Assigned to kernels: {assigned:.6f} mJ")
    print(f"Unassigned: {unassigned:.6f} mJ")
    print(f"Utilization-energy Pearson r: {energy_correlation:.6f}")
    print(f"Utilization-power Pearson r: {power_correlation:.6f}")
    print(f"Results: {args.output_dir}")
    if args.plot:
        print("Plots: energy_timeline.png, utilization_energy_correlation.png")


if __name__ == "__main__":
    main()
