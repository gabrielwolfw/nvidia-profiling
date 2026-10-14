#!/usr/bin/env python3
"""Bayesian attribution of device-level GPU energy to CUDA kernel events."""

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

from analyze_kernel_energy import integrate_range, load_telemetry


NS_PER_SECOND = 1_000_000_000
OTHER_FAMILY = "__other__"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fit a non-negative Bayesian kernel-family power model and "
            "attribute measured NVML energy to CUPTI kernel events."
        )
    )
    parser.add_argument("--telemetry", required=True, type=Path)
    parser.add_argument("--kernels", required=True, type=Path)
    parser.add_argument("--window-ms", type=float, default=200.0)
    parser.add_argument("--idle-power-mw", type=float, default=0.0)
    parser.add_argument("--max-families", type=int, default=64)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--prior-scale-w", type=float, default=250.0)
    parser.add_argument("--noise-prior-scale-w", type=float, default=20.0)
    parser.add_argument("--burn-in", type=int, default=1000)
    parser.add_argument("--posterior-draws", type=int, default=2000)
    parser.add_argument("--thin", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate posterior attribution diagnostic plots (requires matplotlib).",
    )
    parser.add_argument(
        "--skip-event-output",
        action="store_true",
        help="Skip the potentially large per-event CSV while retaining summaries.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("bayesian-kernel-energy-analysis"),
    )
    return parser.parse_args()


def require_fields(reader, required, label):
    fields = set(reader.fieldnames or [])
    missing = required - fields
    if missing:
        raise ValueError(f"{label} CSV is missing columns: {sorted(missing)}")


def parse_kernel_row(row):
    try:
        start_ns = int(row["start_ns"])
        end_ns = int(row["end_ns"])
        name = row["kernel_name"]
    except (KeyError, TypeError, ValueError):
        return None
    if end_ns <= start_ns or not name:
        return None
    return start_ns, end_ns, name


def scan_kernel_families(path, max_families):
    duration_by_name = Counter()
    event_count_by_name = Counter()
    first_start = None
    last_end = None
    previous_start = None
    skipped = 0

    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        require_fields(reader, {"start_ns", "end_ns", "kernel_name"}, "kernels")
        for row in reader:
            parsed = parse_kernel_row(row)
            if parsed is None:
                skipped += 1
                continue
            start_ns, end_ns, name = parsed
            if previous_start is not None and start_ns < previous_start:
                raise ValueError(
                    "Kernel rows must be sorted by start_ns for streaming analysis"
                )
            previous_start = start_ns
            duration_by_name[name] += end_ns - start_ns
            event_count_by_name[name] += 1
            first_start = start_ns if first_start is None else first_start
            last_end = end_ns if last_end is None else max(last_end, end_ns)

    if not duration_by_name:
        raise ValueError("No valid kernel events were found")
    if skipped:
        print(f"WARNING: Skipped {skipped} invalid kernel rows.")

    ranked = [name for name, _ in duration_by_name.most_common()]
    if len(ranked) > max_families:
        selected = ranked[: max_families - 1]
        families = selected + [OTHER_FAMILY]
    else:
        selected = ranked
        families = selected
    return {
        "families": families,
        "selected": set(selected),
        "duration_by_name": duration_by_name,
        "event_count_by_name": event_count_by_name,
        "first_start": first_start,
        "last_end": last_end,
        "valid_events": sum(event_count_by_name.values()),
    }


def family_for_name(name, selected):
    return name if name in selected else OTHER_FAMILY


def aggregate_family_activity(
    path, selected, family_index, analysis_start, analysis_end, window_ns
):
    window_count = math.ceil((analysis_end - analysis_start) / window_ns)
    activity_ns = np.zeros((window_count, len(family_index)), dtype=np.float64)
    union_active_ns = np.zeros(window_count, dtype=np.float64)
    union_end_ns = np.full(window_count, -1, dtype=np.int64)
    valid_index = 0

    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        for row in reader:
            parsed = parse_kernel_row(row)
            if parsed is None:
                continue
            start_ns, end_ns, name = parsed
            clipped_start = max(start_ns, analysis_start)
            clipped_end = min(end_ns, analysis_end)
            if clipped_end > clipped_start:
                family = family_for_name(name, selected)
                column = family_index[family]
                first_window = (clipped_start - analysis_start) // window_ns
                last_window = (clipped_end - 1 - analysis_start) // window_ns
                for window in range(first_window, last_window + 1):
                    window_start = analysis_start + window * window_ns
                    window_end = min(window_start + window_ns, analysis_end)
                    overlap_start = max(clipped_start, window_start)
                    overlap_end = min(clipped_end, window_end)
                    activity_ns[window, column] += overlap_end - overlap_start
                    previous_end = union_end_ns[window]
                    if overlap_end > previous_end:
                        uncovered_start = max(overlap_start, previous_end)
                        union_active_ns[window] += overlap_end - uncovered_start
                        union_end_ns[window] = overlap_end
            valid_index += 1
            if valid_index % 1_000_000 == 0:
                print(f"Aggregated {valid_index} kernel events...", flush=True)

    return activity_ns, union_active_ns


def build_windows(
    samples,
    activity_ns,
    union_active_ns,
    analysis_start,
    analysis_end,
    window_ns,
    idle_power_mw,
):
    rows = []
    dynamic_power_w = []
    for index in range(len(activity_ns)):
        start_ns = analysis_start + index * window_ns
        end_ns = min(start_ns + window_ns, analysis_end)
        duration_ns = end_ns - start_ns
        device_energy_mj, dynamic_energy_mj = integrate_range(
            samples, start_ns, end_ns, idle_power_mw
        )
        duration_seconds = duration_ns / NS_PER_SECOND
        active_fraction = min(1.0, union_active_ns[index] / duration_ns)
        assignable_energy_mj = dynamic_energy_mj * active_fraction
        rows.append(
            {
                "window_index": index,
                "start_ns": start_ns,
                "end_ns": end_ns,
                "duration_ns": duration_ns,
                "kernel_active_fraction": active_fraction,
                "device_energy_mj": device_energy_mj,
                "dynamic_energy_mj": dynamic_energy_mj,
                "assignable_energy_mj": assignable_energy_mj,
                "unassigned_energy_mj": dynamic_energy_mj
                - assignable_energy_mj,
            }
        )
        dynamic_power_w.append(dynamic_energy_mj / duration_seconds / 1000.0)
    return rows, np.asarray(dynamic_power_w, dtype=np.float64)


def sample_lower_truncated_normal(mean, standard_deviation, rng):
    """Draw from N(mean, sd) truncated to [0, infinity)."""
    if standard_deviation <= 0.0:
        return max(0.0, mean)
    lower = -mean / standard_deviation
    if lower <= 0.0:
        while True:
            value = rng.normal()
            if value >= lower:
                return mean + standard_deviation * value
    alpha = 0.5 * (lower + math.sqrt(lower * lower + 4.0))
    while True:
        value = lower + rng.exponential(1.0 / alpha)
        if rng.random() <= math.exp(-0.5 * (value - alpha) ** 2):
            return mean + standard_deviation * value


def fit_nonnegative_bayesian_regression(
    design,
    target,
    train_count,
    prior_scale_w,
    noise_prior_scale_w,
    burn_in,
    posterior_draws,
    thin,
    seed,
):
    train_design = design[:train_count]
    train_target = target[:train_count]
    column_scale = np.sqrt(np.mean(train_design * train_design, axis=0))
    column_scale[column_scale < 1e-12] = 1.0
    scaled_design = design / column_scale
    train_scaled = scaled_design[:train_count]

    beta = np.maximum(
        0.0, np.linalg.lstsq(train_scaled, train_target, rcond=None)[0]
    )
    residual = train_target - train_scaled @ beta
    sigma2 = max(float(np.mean(residual * residual)), 1e-6)
    prior_variance = prior_scale_w**2
    noise_shape = 2.0
    noise_rate = noise_prior_scale_w**2
    rng = np.random.default_rng(seed)
    draws = np.empty((posterior_draws, design.shape[1]), dtype=np.float64)
    noise_draws = np.empty(posterior_draws, dtype=np.float64)
    total_iterations = burn_in + posterior_draws * thin
    saved = 0

    for iteration in range(total_iterations):
        for column in range(train_scaled.shape[1]):
            values = train_scaled[:, column]
            residual += values * beta[column]
            precision = float(values @ values) / sigma2 + 1.0 / prior_variance
            variance = 1.0 / precision
            mean = variance * float(values @ residual) / sigma2
            beta[column] = sample_lower_truncated_normal(
                mean, math.sqrt(variance), rng
            )
            residual -= values * beta[column]

        posterior_shape = noise_shape + train_count / 2.0
        posterior_rate = noise_rate + 0.5 * float(residual @ residual)
        sigma2 = 1.0 / rng.gamma(posterior_shape, 1.0 / posterior_rate)

        if iteration >= burn_in and (iteration - burn_in) % thin == 0:
            draws[saved] = beta / column_scale
            noise_draws[saved] = sigma2
            saved += 1

    prediction_draws = draws @ design.T
    return draws, noise_draws, prediction_draws, column_scale


def regression_metrics(actual, predicted):
    residual = actual - predicted
    total_variation = float(np.sum((actual - actual.mean()) ** 2))
    squared_error = float(residual @ residual)
    return {
        "count": int(len(actual)),
        "mae_w": float(np.mean(np.abs(residual))),
        "rmse_w": float(np.sqrt(np.mean(residual * residual))),
        "r_squared": (
            1.0 - squared_error / total_variation
            if total_variation > 0.0
            else None
        ),
    }


def posterior_energy_allocation(
    coefficient_draws, activity_ns, window_rows, window_ns
):
    family_draws = coefficient_draws[:, 1:]
    draw_count, family_count = family_draws.shape
    window_count = len(window_rows)
    means = np.zeros((window_count, family_count), dtype=np.float64)
    lowers = np.zeros_like(means)
    uppers = np.zeros_like(means)
    family_total_draws = np.zeros((draw_count, family_count), dtype=np.float64)

    for window in range(window_count):
        activity_fraction = activity_ns[window] / window_ns
        scores = family_draws * activity_fraction[None, :]
        totals = scores.sum(axis=1)
        active = activity_fraction > 0.0
        zero = totals <= 0.0
        if np.any(zero) and np.any(active):
            scores[zero] = activity_fraction
            totals[zero] = activity_fraction.sum()
        allocation = np.zeros_like(scores)
        positive = totals > 0.0
        allocation[positive] = (
            scores[positive] / totals[positive, None]
            * window_rows[window]["assignable_energy_mj"]
        )
        means[window] = allocation.mean(axis=0)
        lowers[window], uppers[window] = np.quantile(
            allocation, [0.025, 0.975], axis=0
        )
        family_total_draws += allocation

    return means, lowers, uppers, family_total_draws


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_event_attribution(
    kernel_path,
    output_path,
    selected,
    family_index,
    activity_ns,
    energy_mean,
    energy_lower,
    energy_upper,
    analysis_start,
    analysis_end,
    window_ns,
):
    event_count = 0
    attributed_total = 0.0
    with kernel_path.open(newline="", encoding="utf-8") as source, output_path.open(
        "w", newline="", encoding="utf-8"
    ) as destination:
        reader = csv.DictReader(source)
        fields = list(reader.fieldnames or []) + [
            "kernel_family",
            "bayesian_energy_mean_mj",
            "bayesian_energy_lower_95_mj",
            "bayesian_energy_upper_95_mj",
        ]
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for row in reader:
            parsed = parse_kernel_row(row)
            if parsed is None:
                continue
            start_ns, end_ns, name = parsed
            family = family_for_name(name, selected)
            column = family_index[family]
            clipped_start = max(start_ns, analysis_start)
            clipped_end = min(end_ns, analysis_end)
            mean = 0.0
            lower = 0.0
            upper = 0.0
            if clipped_end > clipped_start:
                first_window = (clipped_start - analysis_start) // window_ns
                last_window = (clipped_end - 1 - analysis_start) // window_ns
                for window in range(first_window, last_window + 1):
                    window_start = analysis_start + window * window_ns
                    window_end = min(window_start + window_ns, analysis_end)
                    overlap = max(
                        0,
                        min(clipped_end, window_end)
                        - max(clipped_start, window_start),
                    )
                    family_duration = activity_ns[window, column]
                    if overlap and family_duration > 0.0:
                        fraction = overlap / family_duration
                        mean += energy_mean[window, column] * fraction
                        lower += energy_lower[window, column] * fraction
                        upper += energy_upper[window, column] * fraction
            row.update(
                {
                    "kernel_family": family,
                    "bayesian_energy_mean_mj": mean,
                    "bayesian_energy_lower_95_mj": lower,
                    "bayesian_energy_upper_95_mj": upper,
                }
            )
            writer.writerow(row)
            attributed_total += mean
            event_count += 1
            if event_count % 1_000_000 == 0:
                print(f"Wrote {event_count} attributed kernel events...", flush=True)
    return event_count, attributed_total


def short_kernel_name(name, limit=52):
    if name == OTHER_FAMILY:
        return "Other kernels"
    matrix_vector = re.search(
        r"mul_mat_vec_qIL9ggml_type(\d+)ELi(\d+)ELb([01])ELb([01])ELb([01])",
        name,
    )
    if matrix_vector:
        metric_type, integer, first, second, third = matrix_vector.groups()
        return (
            f"mul_mat_vec_q<t{metric_type},i{integer},"
            f"b{first}{second}{third}>"
        )
    if "rope_norm" in name:
        return "rope_norm<half>" if "6__half" in name else "rope_norm<float>"
    attention = re.search(
        r"flash_attn_tileILi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)", name
    )
    if attention:
        return "flash_attn_tile<" + ",".join(attention.groups()) + ">"
    if name.startswith("_Z"):
        position = 2
        while position < len(name) and name[position].isdigit():
            position += 1
        if position > 2:
            length = int(name[2:position])
            candidate = name[position : position + length]
            if candidate:
                return candidate
    return name if len(name) <= limit else name[: limit - 1] + "…"


def create_plots(output_dir, window_rows, family_rows, coefficient_rows):
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("--plot requires matplotlib") from error

    top_families = family_rows[: min(15, len(family_rows))]
    labels = [short_kernel_name(row["kernel_family"]) for row in top_families]
    means = np.asarray(
        [row["energy_posterior_mean_mj"] / 1000.0 for row in top_families]
    )
    lowers = np.asarray(
        [row["energy_posterior_lower_95_mj"] / 1000.0 for row in top_families]
    )
    uppers = np.asarray(
        [row["energy_posterior_upper_95_mj"] / 1000.0 for row in top_families]
    )
    positions = np.arange(len(top_families))[::-1]
    figure, axis = plt.subplots(figsize=(11, 7))
    axis.barh(positions, means, color="tab:blue", alpha=0.8)
    axis.errorbar(
        means,
        positions,
        xerr=np.vstack([means - lowers, uppers - means]),
        fmt="none",
        ecolor="black",
        capsize=3,
        linewidth=1,
        label="95% credible interval",
    )
    axis.set_yticks(positions, labels)
    axis.set_xlabel("Attributed energy (J)")
    axis.set_title("Bayesian energy attribution by kernel family")
    axis.grid(axis="x", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "bayesian_kernel_family_energy.png", dpi=180)
    plt.close(figure)

    elapsed_seconds = np.asarray(
        [
            (row["start_ns"] - window_rows[0]["start_ns"]) / NS_PER_SECOND
            for row in window_rows
        ]
    )
    measured = np.asarray(
        [
            row["dynamic_energy_mj"]
            / (row["duration_ns"] / NS_PER_SECOND)
            / 1000.0
            for row in window_rows
        ]
    )
    predicted = np.asarray(
        [row["predicted_dynamic_power_mean_w"] for row in window_rows]
    )
    predicted_lower = np.asarray(
        [row["predicted_dynamic_power_lower_95_w"] for row in window_rows]
    )
    predicted_upper = np.asarray(
        [row["predicted_dynamic_power_upper_95_w"] for row in window_rows]
    )
    test_start = next(
        (index for index, row in enumerate(window_rows) if row["split"] == "test"),
        len(window_rows),
    )
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(elapsed_seconds, measured, label="Measured NVML power", linewidth=1.2)
    axis.plot(elapsed_seconds, predicted, label="Posterior mean", linewidth=1.2)
    axis.fill_between(
        elapsed_seconds,
        predicted_lower,
        predicted_upper,
        alpha=0.2,
        label="95% credible interval",
    )
    if test_start < len(window_rows):
        axis.axvline(
            elapsed_seconds[test_start],
            color="black",
            linestyle="--",
            linewidth=1,
            label="Temporal test split",
        )
    axis.set_xlabel("Elapsed time (s)")
    axis.set_ylabel("Dynamic power (W)")
    axis.set_title("Kernel-family Bayesian power fit")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "bayesian_kernel_power_fit.png", dpi=180)
    plt.close(figure)

    attributed = np.asarray(
        [row["bayesian_attributed_energy_mean_mj"] / 1000.0 for row in window_rows]
    )
    unassigned = np.asarray(
        [row["unassigned_energy_mj"] / 1000.0 for row in window_rows]
    )
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.stackplot(
        elapsed_seconds,
        attributed,
        unassigned,
        labels=["Attributed to kernels", "Unassigned"],
        colors=["tab:green", "tab:gray"],
        alpha=0.8,
    )
    axis.set_xlabel("Elapsed time (s)")
    axis.set_ylabel("Energy per window (J)")
    axis.set_title("Measured dynamic energy partition")
    axis.grid(alpha=0.2)
    axis.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(output_dir / "bayesian_energy_partition.png", dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    if args.window_ms <= 0.0:
        raise ValueError("--window-ms must be greater than zero")
    if args.idle_power_mw < 0.0:
        raise ValueError("--idle-power-mw cannot be negative")
    if args.max_families < 2:
        raise ValueError("--max-families must be at least 2")
    if not 0.5 <= args.train_fraction < 1.0:
        raise ValueError("--train-fraction must be in [0.5, 1.0)")
    if args.prior_scale_w <= 0.0 or args.noise_prior_scale_w <= 0.0:
        raise ValueError("Prior scales must be greater than zero")
    if args.burn_in < 0 or args.posterior_draws < 100 or args.thin < 1:
        raise ValueError("Invalid MCMC settings")

    window_ns = int(args.window_ms * 1_000_000)
    samples = load_telemetry(args.telemetry)
    print("Scanning kernel families...", flush=True)
    metadata = scan_kernel_families(args.kernels, args.max_families)
    analysis_start = max(samples[0][0], metadata["first_start"])
    analysis_end = min(samples[-1][0], metadata["last_end"])
    if analysis_end <= analysis_start:
        raise ValueError("Telemetry and kernel timelines do not overlap")

    families = metadata["families"]
    family_index = {name: index for index, name in enumerate(families)}
    print(
        f"Using {len(families)} kernel families for "
        f"{metadata['valid_events']} events.",
        flush=True,
    )
    activity_ns, union_active_ns = aggregate_family_activity(
        args.kernels,
        metadata["selected"],
        family_index,
        analysis_start,
        analysis_end,
        window_ns,
    )
    window_rows, target_power_w = build_windows(
        samples,
        activity_ns,
        union_active_ns,
        analysis_start,
        analysis_end,
        window_ns,
        args.idle_power_mw,
    )
    activity_fraction = activity_ns / window_ns
    design = np.column_stack([np.ones(len(activity_fraction)), activity_fraction])
    train_count = int(len(window_rows) * args.train_fraction)
    train_count = min(max(train_count, 5), len(window_rows) - 2)
    print(
        f"Sampling posterior with {train_count} training windows...", flush=True
    )
    coefficient_draws, noise_draws, prediction_draws, column_scale = (
        fit_nonnegative_bayesian_regression(
            design,
            target_power_w,
            train_count,
            args.prior_scale_w,
            args.noise_prior_scale_w,
            args.burn_in,
            args.posterior_draws,
            args.thin,
            args.seed,
        )
    )
    predicted_mean = prediction_draws.mean(axis=0)
    predicted_lower, predicted_upper = np.quantile(
        prediction_draws, [0.025, 0.975], axis=0
    )
    energy_mean, energy_lower, energy_upper, family_total_draws = (
        posterior_energy_allocation(
            coefficient_draws, activity_ns, window_rows, window_ns
        )
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(window_rows):
        row.update(
            {
                "split": "train" if index < train_count else "test",
                "predicted_dynamic_power_mean_w": predicted_mean[index],
                "predicted_dynamic_power_lower_95_w": predicted_lower[index],
                "predicted_dynamic_power_upper_95_w": predicted_upper[index],
                "bayesian_attributed_energy_mean_mj": energy_mean[index].sum(),
            }
        )
    write_csv(
        args.output_dir / "bayesian_energy_windows.csv",
        window_rows,
        list(window_rows[0]),
    )

    coefficient_rows = []
    terms = ["unattributed_baseline"] + families
    for index, term in enumerate(terms):
        values = coefficient_draws[:, index]
        lower, upper = np.quantile(values, [0.025, 0.975])
        coefficient_rows.append(
            {
                "term": term,
                "posterior_power_mean_w": values.mean(),
                "posterior_power_lower_95_w": lower,
                "posterior_power_upper_95_w": upper,
                "activity_rms_scale": column_scale[index],
            }
        )
    write_csv(
        args.output_dir / "bayesian_kernel_coefficients.csv",
        coefficient_rows,
        list(coefficient_rows[0]),
    )

    family_rows = []
    for family, column in family_index.items():
        values = family_total_draws[:, column]
        lower, upper = np.quantile(values, [0.025, 0.975])
        if family == OTHER_FAMILY:
            source_names = metadata["duration_by_name"].keys() - metadata["selected"]
            events = sum(metadata["event_count_by_name"][name] for name in source_names)
            duration = sum(metadata["duration_by_name"][name] for name in source_names)
        else:
            events = metadata["event_count_by_name"][family]
            duration = metadata["duration_by_name"][family]
        family_rows.append(
            {
                "kernel_family": family,
                "event_count": events,
                "total_duration_ns": duration,
                "energy_posterior_mean_mj": values.mean(),
                "energy_posterior_lower_95_mj": lower,
                "energy_posterior_upper_95_mj": upper,
            }
        )
    family_rows.sort(key=lambda row: row["energy_posterior_mean_mj"], reverse=True)
    write_csv(
        args.output_dir / "bayesian_kernel_family_energy.csv",
        family_rows,
        list(family_rows[0]),
    )

    if args.plot:
        create_plots(
            args.output_dir, window_rows, family_rows, coefficient_rows
        )

    if args.skip_event_output:
        event_count = metadata["valid_events"]
        attributed_total = float(energy_mean.sum())
    else:
        print("Writing per-event Bayesian attribution...", flush=True)
        event_count, attributed_total = write_event_attribution(
            args.kernels,
            args.output_dir / "bayesian_kernel_event_energy.csv",
            metadata["selected"],
            family_index,
            activity_ns,
            energy_mean,
            energy_lower,
            energy_upper,
            analysis_start,
            analysis_end,
            window_ns,
        )

    train_metrics = regression_metrics(
        target_power_w[:train_count], predicted_mean[:train_count]
    )
    test_metrics = regression_metrics(
        target_power_w[train_count:], predicted_mean[train_count:]
    )
    total_dynamic = sum(row["dynamic_energy_mj"] for row in window_rows)
    total_assignable = sum(row["assignable_energy_mj"] for row in window_rows)
    summary = {
        "model": "nonnegative_bayesian_kernel_family_regression",
        "inference": "Gibbs sampling with half-normal coefficient priors",
        "attribution": (
            "Posterior family contributions normalized within each window; "
            "family energy distributed to events by temporal overlap"
        ),
        "important_limitation": (
            "No direct per-kernel energy ground truth is available. Credible "
            "intervals quantify model uncertainty, not physical meter accuracy."
        ),
        "event_interval_note": (
            "Per-event bounds sum marginal 95% family-window intervals and are "
            "an approximation when an event crosses more than one window."
        ),
        "window_ms": args.window_ms,
        "idle_power_mw": args.idle_power_mw,
        "windows": len(window_rows),
        "train_windows": train_count,
        "test_windows": len(window_rows) - train_count,
        "kernel_events": event_count,
        "kernel_families": len(families),
        "design_matrix_rank": int(np.linalg.matrix_rank(design[:train_count])),
        "design_parameters": int(design.shape[1]),
        "posterior_draws": args.posterior_draws,
        "posterior_noise_mean_w": float(np.sqrt(noise_draws).mean()),
        "train_power_metrics": train_metrics,
        "test_power_metrics": test_metrics,
        "measured_dynamic_energy_mj": total_dynamic,
        "assignable_dynamic_energy_mj": total_assignable,
        "unassigned_dynamic_energy_mj": total_dynamic - total_assignable,
        "attributed_posterior_mean_mj": attributed_total,
        "conservation_error_mj": total_assignable - attributed_total,
    }
    with (args.output_dir / "bayesian_kernel_energy_summary.json").open(
        "w", encoding="utf-8"
    ) as destination:
        json.dump(summary, destination, indent=2)
        destination.write("\n")

    print("=== Bayesian kernel-energy attribution ===")
    print(f"Kernel families: {len(families)}")
    print(f"Kernel events: {event_count}")
    print(f"Test power RMSE: {test_metrics['rmse_w']:.3f} W")
    if test_metrics["r_squared"] is not None:
        print(f"Test power R-squared: {test_metrics['r_squared']:.4f}")
    print(f"Measured dynamic energy: {total_dynamic:.3f} mJ")
    print(f"Bayesian attributed energy: {attributed_total:.3f} mJ")
    print(f"Unassigned energy: {total_dynamic - total_assignable:.3f} mJ")
    print(f"Conservation error: {total_assignable - attributed_total:.6g} mJ")
    if summary["design_matrix_rank"] < summary["design_parameters"]:
        print(
            "WARNING: Kernel-family activity is not fully identifiable "
            f"(rank {summary['design_matrix_rank']} of "
            f"{summary['design_parameters']}); interpret overlapping family "
            "intervals conservatively."
        )
    print(f"Results: {args.output_dir}")


if __name__ == "__main__":
    main()
