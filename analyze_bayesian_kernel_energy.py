#!/usr/bin/env python3
"""Bayesian attribution of device-level GPU energy to CUDA kernel events."""

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from energy_attribution.diagnostics import (
    DEFAULT_MIN_WINDOWS_PER_PARAMETER,
    check_identifiability,
    print_identifiability_warnings,
)
from energy_attribution.io import (
    NS_PER_SECOND,
    load_kernels,
    load_power_trace,
    write_csv,
)
from energy_attribution.windows import (
    EVENT_CHUNK,
    OTHER_FAMILY,
    aggregate_family_activity,
    build_windows,
    distribute_to_events,
    family_columns,
    select_families,
)


# "model": split each window's measured dynamic energy in proportion to the
# fitted terms b0 + sum(beta_f * a_f), so the intercept keeps its share.
# "active-fraction": legacy rule that scales energy by the kernel-busy
# fraction and splits all of it among families, ignoring b0.
ATTRIBUTION_MODES = ("model", "active-fraction")
DEFAULT_MIN_WINDOW_FRACTION = 0.9


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
    add_model_check_arguments(parser)
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


def add_model_check_arguments(parser):
    parser.add_argument(
        "--attribution",
        choices=ATTRIBUTION_MODES,
        default="model",
        help=(
            "'model' (default) keeps the intercept's share of each window as "
            "baseline energy; 'active-fraction' reproduces the legacy rule."
        ),
    )
    parser.add_argument(
        "--min-windows-per-parameter",
        type=float,
        default=DEFAULT_MIN_WINDOWS_PER_PARAMETER,
        help="Minimum training windows per regression parameter.",
    )
    parser.add_argument(
        "--allow-underdetermined",
        action="store_true",
        help="Fit even when training windows per parameter is below the minimum.",
    )
    parser.add_argument(
        "--min-window-fraction",
        type=float,
        default=DEFAULT_MIN_WINDOW_FRACTION,
        help=(
            "Windows shorter than this fraction of --window-ms (the truncated "
            "last window of a session) are excluded from fitting, metrics and "
            "plots; their energy is still attributed. 0 keeps every window."
        ),
    )


def scan_kernel_families(table, max_families):
    """Summarize a loaded KernelTable and choose its family basis."""
    if not len(table):
        raise ValueError("No valid kernel events were found")
    if table.skipped:
        print(f"WARNING: Skipped {table.skipped} invalid kernel rows.")
    duration_by_name, event_count_by_name = table.name_statistics()
    families, selected = select_families(duration_by_name, max_families)
    return {
        "table": table,
        "families": families,
        "selected": selected,
        "duration_by_name": duration_by_name,
        "event_count_by_name": event_count_by_name,
        "first_start": table.first_start,
        "last_end": table.last_end,
        "valid_events": len(table),
    }


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


def interval_coverage(actual, lower, upper):
    if not len(actual):
        return None
    return float(np.mean((actual >= lower) & (actual <= upper)))


def summarize_predictions(prediction_draws, noise_draws, seed):
    """Posterior mean plus credible and predictive 95% intervals per window.

    ``prediction_draws`` reflect only coefficient uncertainty (credible). A
    measured window also carries observation noise sigma, so intervals meant
    to contain measurements add N(0, sigma^2) from the same draw (predictive).
    Power is non-negative, so predictive draws are clipped at zero.
    """
    rng = np.random.default_rng(seed)
    predictive_draws = np.maximum(
        0.0,
        prediction_draws
        + rng.normal(size=prediction_draws.shape) * np.sqrt(noise_draws)[:, None],
    )
    credible_lower, credible_upper = np.quantile(
        prediction_draws, [0.025, 0.975], axis=0
    )
    predictive_lower, predictive_upper = np.quantile(
        predictive_draws, [0.025, 0.975], axis=0
    )
    return {
        "mean": prediction_draws.mean(axis=0),
        "credible_lower": credible_lower,
        "credible_upper": credible_upper,
        "predictive_lower": predictive_lower,
        "predictive_upper": predictive_upper,
        "prediction_draws": prediction_draws,
        "predictive_draws": predictive_draws,
    }


def add_prediction_columns(window_rows, predictions):
    for index, row in enumerate(window_rows):
        row.update(
            {
                "predicted_dynamic_power_mean_w": predictions["mean"][index],
                "predicted_dynamic_power_lower_95_w": (
                    predictions["credible_lower"][index]
                ),
                "predicted_dynamic_power_upper_95_w": (
                    predictions["credible_upper"][index]
                ),
                "predicted_dynamic_power_predictive_lower_95_w": (
                    predictions["predictive_lower"][index]
                ),
                "predicted_dynamic_power_predictive_upper_95_w": (
                    predictions["predictive_upper"][index]
                ),
            }
        )


def prediction_metrics(target_power_w, predictions, mask):
    """Point-error metrics plus 95% interval coverage on the masked windows."""
    actual = target_power_w[mask]
    metrics = regression_metrics(actual, predictions["mean"][mask])
    metrics["credible_interval_95_coverage"] = interval_coverage(
        actual, predictions["credible_lower"][mask], predictions["credible_upper"][mask]
    )
    metrics["predictive_interval_95_coverage"] = interval_coverage(
        actual,
        predictions["predictive_lower"][mask],
        predictions["predictive_upper"][mask],
    )
    return metrics


def attribution_pool_mj(window_rows, attribution):
    """Energy each window hands to the allocation step under ``attribution``."""
    if attribution == "model":
        for row in window_rows:
            row["assignable_energy_mj"] = row["dynamic_energy_mj"]
            row["unassigned_energy_mj"] = 0.0
    return np.asarray(
        [row["assignable_energy_mj"] for row in window_rows], dtype=np.float64
    )


def posterior_energy_allocation(
    coefficient_draws, activity_ns, pool_energy_mj, duration_ns, attribution="model"
):
    """Split each window's energy pool among families for every posterior draw.

    ``duration_ns`` holds each window's real length, so activity fractions of
    a truncated final window are not understated.

    Returns per-window family mean/lower/upper (mJ), per-draw family totals,
    the per-window baseline (intercept) mean and per-draw baseline totals.
    """
    if attribution not in ATTRIBUTION_MODES:
        raise ValueError(f"Unknown attribution mode: {attribution}")
    baseline_draws = coefficient_draws[:, 0]
    family_draws = coefficient_draws[:, 1:]
    draw_count, family_count = family_draws.shape
    window_count = len(pool_energy_mj)
    means = np.zeros((window_count, family_count), dtype=np.float64)
    lowers = np.zeros_like(means)
    uppers = np.zeros_like(means)
    family_total_draws = np.zeros((draw_count, family_count), dtype=np.float64)
    baseline_mean = np.zeros(window_count, dtype=np.float64)
    baseline_total_draws = np.zeros(draw_count, dtype=np.float64)

    for window in range(window_count):
        activity_fraction = activity_ns[window] / duration_ns[window]
        scores = family_draws * activity_fraction[None, :]
        denominators = scores.sum(axis=1)
        if attribution == "model":
            denominators = denominators + baseline_draws
        active = activity_fraction > 0.0
        degenerate = denominators <= 0.0
        if np.any(degenerate) and np.any(active):
            scores[degenerate] = activity_fraction
            denominators[degenerate] = activity_fraction.sum()
        allocation = np.zeros_like(scores)
        positive = denominators > 0.0
        allocation[positive] = (
            scores[positive] / denominators[positive, None]
            * pool_energy_mj[window]
        )
        means[window] = allocation.mean(axis=0)
        lowers[window], uppers[window] = np.quantile(
            allocation, [0.025, 0.975], axis=0
        )
        family_total_draws += allocation
        baseline = pool_energy_mj[window] - allocation.sum(axis=1)
        baseline_mean[window] = baseline.mean()
        baseline_total_draws += baseline

    return {
        "mean": means,
        "lower": lowers,
        "upper": uppers,
        "family_total_draws": family_total_draws,
        "baseline_mean": baseline_mean,
        "baseline_total_draws": baseline_total_draws,
    }


def apply_allocation_to_rows(window_rows, allocation):
    for row, family_energy, baseline in zip(
        window_rows, allocation["mean"], allocation["baseline_mean"]
    ):
        row["bayesian_attributed_energy_mean_mj"] = float(family_energy.sum())
        row["baseline_energy_mean_mj"] = float(baseline)


def energy_accounting(window_rows, allocation):
    """Energy totals; dynamic = attributed + baseline + unassigned."""
    total_dynamic = sum(row["dynamic_energy_mj"] for row in window_rows)
    total_assignable = sum(row["assignable_energy_mj"] for row in window_rows)
    attributed = float(allocation["mean"].sum())
    baseline_draws = allocation["baseline_total_draws"]
    baseline_lower, baseline_upper = np.quantile(baseline_draws, [0.025, 0.975])
    baseline = float(baseline_draws.mean())
    return {
        "measured_dynamic_energy_mj": total_dynamic,
        "assignable_dynamic_energy_mj": total_assignable,
        "unassigned_dynamic_energy_mj": total_dynamic - total_assignable,
        "attributed_posterior_mean_mj": attributed,
        "baseline_posterior_mean_mj": baseline,
        "baseline_posterior_lower_95_mj": float(baseline_lower),
        "baseline_posterior_upper_95_mj": float(baseline_upper),
        "conservation_error_mj": total_assignable - attributed - baseline,
    }


def write_event_attribution(
    table,
    columns,
    families,
    output_path,
    activity_ns,
    allocation,
    analysis_start,
    analysis_end,
    window_ns,
):
    """Write per-event energy; join back to the CUPTI trace via event_id."""
    mean, lower, upper = distribute_to_events(
        table, columns, activity_ns,
        [allocation["mean"], allocation["lower"], allocation["upper"]],
        analysis_start, analysis_end, window_ns,
    )
    family_names = np.asarray(families, dtype=object)
    for offset in range(0, len(table), EVENT_CHUNK):
        chunk = slice(offset, offset + EVENT_CHUNK)
        frame = pd.DataFrame(
            {
                "start_ns": table.start_ns[chunk],
                "end_ns": table.end_ns[chunk],
                "kernel_family": family_names[columns[chunk]],
                "bayesian_energy_mean_mj": mean[chunk],
                "bayesian_energy_lower_95_mj": lower[chunk],
                "bayesian_energy_upper_95_mj": upper[chunk],
            }
        )
        if table.event_ids is not None:
            frame.insert(0, "event_id", table.event_ids[chunk])
        frame.to_csv(
            output_path,
            mode="w" if offset == 0 else "a",
            header=offset == 0,
            index=False,
        )
    return len(table), float(mean.sum())


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


def plot_kernel_family_energy(plt, output_dir, family_rows):
    top_families = family_rows[: min(15, len(family_rows))]
    labels = [short_kernel_name(row["kernel_family"]) for row in top_families]
    medians = np.asarray(
        [row["energy_posterior_median_mj"] / 1000.0 for row in top_families]
    )
    lowers = np.asarray(
        [row["energy_posterior_lower_95_mj"] / 1000.0 for row in top_families]
    )
    uppers = np.asarray(
        [row["energy_posterior_upper_95_mj"] / 1000.0 for row in top_families]
    )
    positions = np.arange(len(top_families))[::-1]
    figure, axis = plt.subplots(figsize=(11, 7))
    axis.barh(positions, medians, color="tab:blue", alpha=0.8)
    axis.errorbar(
        medians,
        positions,
        xerr=np.vstack([medians - lowers, uppers - medians]),
        fmt="none",
        ecolor="black",
        capsize=3,
        linewidth=1,
        label="95% credible interval",
    )
    axis.set_yticks(positions, labels)
    axis.set_xlabel("Attributed energy posterior median (J)")
    axis.set_title("Bayesian energy attribution by kernel family")
    axis.grid(axis="x", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "bayesian_kernel_family_energy.png", dpi=180)
    plt.close(figure)


def build_plot_series(window_rows):
    elapsed_seconds = np.asarray(
        [
            (row["start_ns"] - window_rows[0]["start_ns"]) / NS_PER_SECOND
            for row in window_rows
        ]
    )
    measured_power = np.asarray(
        [
            row["dynamic_energy_mj"]
            / (row["duration_ns"] / NS_PER_SECOND)
            / 1000.0
            for row in window_rows
        ]
    )
    predicted_power = np.asarray(
        [row["predicted_dynamic_power_mean_w"] for row in window_rows]
    )
    predicted_power_lower = np.asarray(
        [row["predicted_dynamic_power_lower_95_w"] for row in window_rows]
    )
    predicted_power_upper = np.asarray(
        [row["predicted_dynamic_power_upper_95_w"] for row in window_rows]
    )
    predictive_power_lower = np.asarray(
        [row["predicted_dynamic_power_predictive_lower_95_w"] for row in window_rows]
    )
    predictive_power_upper = np.asarray(
        [row["predicted_dynamic_power_predictive_upper_95_w"] for row in window_rows]
    )
    window_seconds = np.asarray(
        [row["duration_ns"] / NS_PER_SECOND for row in window_rows]
    )
    measured_energy = measured_power * window_seconds
    predicted_energy = predicted_power * window_seconds
    predicted_energy_lower = predicted_power_lower * window_seconds
    predicted_energy_upper = predicted_power_upper * window_seconds
    test_start = next(
        (index for index, row in enumerate(window_rows) if row["split"] == "test"),
        len(window_rows),
    )
    return {
        "elapsed_seconds": elapsed_seconds,
        "measured_power": measured_power,
        "predicted_power": predicted_power,
        "predicted_power_lower": predicted_power_lower,
        "predicted_power_upper": predicted_power_upper,
        "predictive_power_lower": predictive_power_lower,
        "predictive_power_upper": predictive_power_upper,
        "predictive_energy_lower": predictive_power_lower * window_seconds,
        "predictive_energy_upper": predictive_power_upper * window_seconds,
        "measured_energy": measured_energy,
        "predicted_energy": predicted_energy,
        "predicted_energy_lower": predicted_energy_lower,
        "predicted_energy_upper": predicted_energy_upper,
        "test_start": test_start,
    }


def session_row_groups(window_rows):
    names = list(dict.fromkeys(row.get("session", "session") for row in window_rows))
    groups = []
    for name in names:
        rows = [
            row for row in window_rows if row.get("session", "session") == name
        ]
        rows.sort(key=lambda row: row.get("session_window_index", row["window_index"]))
        groups.append((name, rows))
    return groups


def plot_kernel_power_fit(plt, output_dir, window_rows):
    groups = session_row_groups(window_rows)
    figure, axes = plt.subplots(
        len(groups), 1, figsize=(12, max(5, 2.3 * len(groups))),
        squeeze=False, constrained_layout=True,
    )
    for group_index, (name, rows) in enumerate(groups):
        axis = axes[group_index, 0]
        series = build_plot_series(rows)
        axis.plot(
            series["elapsed_seconds"], series["measured_power"],
            label="Measured NVML dynamic power", linewidth=1.1,
        )
        axis.plot(
            series["elapsed_seconds"], series["predicted_power"],
            label="Posterior mean", linewidth=1.1,
        )
        axis.fill_between(
            series["elapsed_seconds"], series["predictive_power_lower"],
            series["predictive_power_upper"], alpha=0.12, color="tab:orange",
            linewidth=0, label="95% predictive interval",
        )
        axis.fill_between(
            series["elapsed_seconds"], series["predicted_power_lower"],
            series["predicted_power_upper"], alpha=0.3, color="tab:orange",
            linewidth=0, label="95% credible interval",
        )
        test_start = series["test_start"]
        if test_start < len(series["elapsed_seconds"]):
            axis.axvspan(
                series["elapsed_seconds"][test_start],
                series["elapsed_seconds"][-1], color="grey", alpha=0.12,
                label="Temporal holdout",
            )
        axis.set_ylabel("Power (W)")
        axis.set_title(name if len(groups) > 1 else "Kernel-family Bayesian power fit")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8, ncol=4, loc="best")
    axes[-1, 0].set_xlabel("Elapsed time (s)")
    if len(groups) > 1:
        figure.suptitle("Multi-session kernel-family Bayesian power fit")
    figure.savefig(output_dir / "bayesian_kernel_power_fit.png", dpi=180)
    plt.close(figure)


def plot_kernel_energy_test_scatter(plt, output_dir, window_rows, series):
    test_start = series["test_start"]
    actual = series["measured_energy"][test_start:]
    expected = series["predicted_energy"][test_start:]
    lower = series["predictive_energy_lower"][test_start:]
    upper = series["predictive_energy_upper"][test_start:]
    if not len(actual):
        raise ValueError("Temporal holdout contains no windows to plot")

    figure, (fit_axis, residual_axis) = plt.subplots(
        2,
        1,
        figsize=(9, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]},
    )
    test_rows = window_rows[test_start:]
    session_names = list(
        dict.fromkeys(row.get("session", "session") for row in test_rows)
    )
    colors = plt.get_cmap("tab20", len(session_names))
    for session_index, name in enumerate(session_names):
        indices = np.asarray(
            [
                index for index, row in enumerate(test_rows)
                if row.get("session", "session") == name
            ]
        )
        color = colors(session_index)
        fit_axis.vlines(
            actual[indices], lower[indices], upper[indices],
            color=color, alpha=0.18, linewidth=0.8,
        )
        fit_axis.scatter(
            actual[indices], expected[indices], color=color, alpha=0.7,
            label=name if len(session_names) > 1 else "Temporal holdout windows",
        )
        residual_axis.scatter(
            actual[indices], expected[indices] - actual[indices],
            color=color, alpha=0.7,
        )
    minimum = float(min(actual.min(), expected.min(), lower.min()))
    maximum = float(max(actual.max(), expected.max(), upper.max()))
    fit_axis.plot(
        [minimum, maximum],
        [minimum, maximum],
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="Central line: predicted = measured",
    )
    fit_axis.set_ylabel("Posterior predicted energy (J/window)")
    fit_axis.set_title(
        "Kernel-energy temporal holdout predictions "
        "(vertical bars: 95% predictive interval)"
    )
    fit_axis.grid(alpha=0.2)
    fit_axis.legend()
    residual_axis.axhline(
        0.0, color="black", linestyle="--", linewidth=1.2
    )
    residual_axis.set_xlabel("Measured dynamic energy (J/window)")
    residual_axis.set_ylabel("Distance from central line (J/window)")
    residual_axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(
        output_dir / "bayesian_kernel_energy_test_scatter.png", dpi=180
    )
    plt.close(figure)


def session_holdout_energy(window_rows, predictions):
    """Per-session totals over holdout windows: measured vs predicted energy.

    Intervals sum each posterior draw over the session's windows before taking
    quantiles, so they describe the total rather than adding per-window
    bounds. The predictive total treats window noise as independent; NVML
    samples are autocorrelated, so it can still be too narrow. Power ranges
    skip truncated windows (shorter than half the session's longest).
    """
    indices_by_session = {}
    for index, row in enumerate(window_rows):
        if row["split"] == "test":
            indices_by_session.setdefault(
                row.get("session", "session"), []
            ).append(index)

    session_rows = []
    for name, indices in indices_by_session.items():
        seconds = np.asarray(
            [window_rows[index]["duration_ns"] for index in indices]
        ) / NS_PER_SECOND
        measured_j = np.asarray(
            [window_rows[index]["dynamic_energy_mj"] for index in indices]
        ) / 1000.0
        window_draws = predictions["prediction_draws"][:, indices]
        total_draws = window_draws @ seconds
        lower, upper = np.quantile(total_draws, [0.025, 0.975])
        predictive_lower, predictive_upper = np.quantile(
            predictions["predictive_draws"][:, indices] @ seconds, [0.025, 0.975]
        )
        measured_power = measured_j / seconds
        predicted_power = window_draws.mean(axis=0)
        full = seconds >= 0.5 * seconds.max()
        measured_total = float(measured_j.sum())
        predicted_total = float(total_draws.mean())
        session_rows.append(
            {
                "session": name,
                "test_windows": len(indices),
                "test_seconds": float(seconds.sum()),
                "measured_energy_j": measured_total,
                "predicted_energy_mean_j": predicted_total,
                "predicted_energy_lower_95_j": float(lower),
                "predicted_energy_upper_95_j": float(upper),
                "predicted_energy_predictive_lower_95_j": float(predictive_lower),
                "predicted_energy_predictive_upper_95_j": float(predictive_upper),
                "energy_error_pct": (
                    100.0 * (predicted_total - measured_total) / measured_total
                    if measured_total > 0.0
                    else None
                ),
                "measured_power_mean_w": measured_total / float(seconds.sum()),
                "measured_power_min_w": float(measured_power[full].min()),
                "measured_power_max_w": float(measured_power[full].max()),
                "predicted_power_mean_w": predicted_total / float(seconds.sum()),
                "predicted_power_min_w": float(predicted_power[full].min()),
                "predicted_power_max_w": float(predicted_power[full].max()),
            }
        )

    measured = np.asarray([row["measured_energy_j"] for row in session_rows])
    predicted = np.asarray([row["predicted_energy_mean_j"] for row in session_rows])
    errors = [
        abs(row["energy_error_pct"])
        for row in session_rows
        if row["energy_error_pct"] is not None
    ]
    def coverage(prefix):
        return float(
            np.mean(
                [
                    row[f"{prefix}lower_95_j"]
                    <= row["measured_energy_j"]
                    <= row[f"{prefix}upper_95_j"]
                    for row in session_rows
                ]
            )
        )

    total_variation = float(np.sum((measured - measured.mean()) ** 2))
    session_metrics = {
        "sessions": len(session_rows),
        "energy_r_squared": (
            1.0 - float(np.sum((measured - predicted) ** 2)) / total_variation
            if len(session_rows) > 1 and total_variation > 0.0
            else None
        ),
        "mean_absolute_energy_error_pct": (
            float(np.mean(errors)) if errors else None
        ),
        "max_absolute_energy_error_pct": float(max(errors)) if errors else None,
        "total_energy_error_pct": (
            100.0 * float(predicted.sum() - measured.sum()) / float(measured.sum())
        ),
        "credible_interval_95_coverage": coverage("predicted_energy_"),
        "predictive_interval_95_coverage": coverage("predicted_energy_predictive_"),
    }
    return session_rows, session_metrics


def short_session_name(name):
    return re.sub(r"-[0-9a-f]{10,}$", "", name)


MEASURED_COLOR = "#52514e"
PREDICTED_COLOR = "#2a78d6"
REFERENCE_COLOR = "#8a8984"


def format_optional(value, pattern):
    return "n/a" if value is None else pattern.format(value)


def plot_session_energy(plt, output_dir, session_rows, session_metrics, test_metrics):
    # Numbered markers (shared with the power panel's ticks) stay legible when
    # sessions with similar totals cluster; full names would collide.
    rows = sorted(
        session_rows, key=lambda row: row["measured_power_mean_w"], reverse=True
    )
    numbers = [str(index + 1) for index in range(len(rows))]
    labels = [
        f"{number}  {short_session_name(row['session'])}"
        for number, row in zip(numbers, rows)
    ]
    measured = np.asarray([row["measured_energy_j"] for row in rows])
    predicted = np.asarray([row["predicted_energy_mean_j"] for row in rows])
    lower = np.asarray([row["predicted_energy_lower_95_j"] for row in rows])
    upper = np.asarray([row["predicted_energy_upper_95_j"] for row in rows])
    predictive_lower = np.asarray(
        [row["predicted_energy_predictive_lower_95_j"] for row in rows]
    )
    predictive_upper = np.asarray(
        [row["predicted_energy_predictive_upper_95_j"] for row in rows]
    )

    figure, (energy_axis, power_axis) = plt.subplots(
        1, 2, figsize=(15, max(6.0, 0.5 * len(rows) + 3.0)),
        gridspec_kw={"width_ratios": [1.15, 1.0]},
    )

    limit_low = 0.0
    limit_high = 1.05 * float(
        max(measured.max(), upper.max(), predictive_upper.max())
    )
    energy_axis.plot(
        [limit_low, limit_high], [limit_low, limit_high],
        color=REFERENCE_COLOR, linestyle="--", linewidth=1.2,
        label="Predicted = measured",
    )
    energy_axis.errorbar(
        measured, predicted,
        yerr=np.vstack([predicted - predictive_lower, predictive_upper - predicted]),
        fmt="none", ecolor=PREDICTED_COLOR, alpha=0.45,
        capsize=4, elinewidth=1.0, label="95% predictive interval",
    )
    energy_axis.errorbar(
        measured, predicted,
        yerr=np.vstack([predicted - lower, upper - predicted]),
        fmt="o", color=PREDICTED_COLOR, ecolor=PREDICTED_COLOR,
        markersize=7, capsize=0, elinewidth=3.0,
        markeredgecolor="white", markeredgewidth=1.0,
        label="Posterior mean ± 95% credible interval",
    )
    for index, (number, x_value, y_value) in enumerate(
        zip(numbers, measured, predicted)
    ):
        energy_axis.annotate(
            number, (x_value, y_value),
            xytext=(-12, 2) if index % 2 else (6, -10),
            textcoords="offset points", fontsize=8, fontweight="bold",
            color=MEASURED_COLOR,
        )
    energy_axis.set_xlim(limit_low, limit_high)
    energy_axis.set_ylim(limit_low, limit_high)
    energy_axis.set_aspect("equal", adjustable="box")
    energy_axis.set_xlabel("Measured dynamic energy in holdout (J)")
    energy_axis.set_ylabel("Predicted dynamic energy in holdout (J)")
    energy_axis.set_title("Total holdout energy per session")
    energy_axis.grid(alpha=0.2)
    energy_axis.legend(loc="lower right", fontsize=8)
    energy_axis.text(
        0.03, 0.97,
        "\n".join(
            [
                "Per session (energy totals):",
                "  R² = "
                + format_optional(session_metrics["energy_r_squared"], "{:.4f}"),
                "  Mean |error| = "
                + format_optional(
                    session_metrics["mean_absolute_energy_error_pct"], "{:.1f}%"
                ),
                "  All sessions: "
                + format_optional(
                    session_metrics["total_energy_error_pct"], "{:+.1f}%"
                ),
                "  95% coverage: credible "
                + f"{100.0 * session_metrics['credible_interval_95_coverage']:.0f}%"
                + ", predictive "
                + f"{100.0 * session_metrics['predictive_interval_95_coverage']:.0f}%",
                "Per window (power):",
                "  R² = " + format_optional(test_metrics["r_squared"], "{:.4f}"),
                f"  RMSE = {test_metrics['rmse_w']:.3f} W",
                "  95% coverage: credible "
                + format_optional(
                    test_metrics["credible_interval_95_coverage"], "{:.0%}"
                )
                + ", predictive "
                + format_optional(
                    test_metrics["predictive_interval_95_coverage"], "{:.0%}"
                ),
            ]
        ),
        transform=energy_axis.transAxes, va="top", fontsize=9,
        family="monospace",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#d6d5d0"},
    )

    positions = np.arange(len(rows))[::-1]
    offset = 0.17
    power_axis.hlines(
        positions + offset,
        [row["measured_power_min_w"] for row in rows],
        [row["measured_power_max_w"] for row in rows],
        color=MEASURED_COLOR, linewidth=2.0, label="Measured min–max",
    )
    power_axis.plot(
        [row["measured_power_mean_w"] for row in rows], positions + offset,
        "o", color=MEASURED_COLOR, markersize=7, markeredgecolor="white",
        label="Measured mean",
    )
    power_axis.hlines(
        positions - offset,
        [row["predicted_power_min_w"] for row in rows],
        [row["predicted_power_max_w"] for row in rows],
        color=PREDICTED_COLOR, linewidth=2.0, label="Predicted min–max",
    )
    power_axis.plot(
        [row["predicted_power_mean_w"] for row in rows], positions - offset,
        "o", color=PREDICTED_COLOR, markersize=7, markeredgecolor="white",
        label="Predicted mean",
    )
    power_axis.set_yticks(positions, labels, fontsize=8)
    power_axis.set_xlabel("Dynamic power per window in holdout (W)")
    power_axis.set_title("Power range per session")
    power_axis.grid(axis="x", alpha=0.2)
    power_axis.legend(loc="lower right", fontsize=8)

    figure.suptitle("Temporal holdout accuracy per session")
    figure.tight_layout()
    figure.savefig(output_dir / "bayesian_session_energy.png", dpi=180)
    plt.close(figure)


def create_plots(
    output_dir, window_rows, family_rows,
    session_rows=None, session_metrics=None, test_metrics=None,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("--plot requires matplotlib") from error

    for obsolete_name in (
        "bayesian_energy_partition.png",
        "bayesian_kernel_energy_temporal_holdout.png",
    ):
        (output_dir / obsolete_name).unlink(missing_ok=True)

    window_rows = [row for row in window_rows if row["split"] != "excluded"]
    plot_kernel_family_energy(plt, output_dir, family_rows)
    series = build_plot_series(window_rows)
    plot_kernel_power_fit(plt, output_dir, window_rows)
    plot_kernel_energy_test_scatter(plt, output_dir, window_rows, series)
    if session_rows:
        plot_session_energy(
            plt, output_dir, session_rows, session_metrics, test_metrics
        )


def print_coverage(test_metrics, session_metrics):
    print(
        "Test 95% coverage per window: "
        f"credible {format_optional(test_metrics['credible_interval_95_coverage'], '{:.1%}')}, "
        f"predictive {format_optional(test_metrics['predictive_interval_95_coverage'], '{:.1%}')}"
    )
    print(
        "Test 95% coverage of session energy: "
        f"credible {session_metrics['credible_interval_95_coverage']:.1%}, "
        f"predictive {session_metrics['predictive_interval_95_coverage']:.1%}"
    )


def validate_common_args(args):
    if args.window_ms <= 0.0:
        raise ValueError("--window-ms must be greater than zero")
    if args.idle_power_mw < 0.0:
        raise ValueError("--idle-power-mw cannot be negative")
    if args.prior_scale_w <= 0.0 or args.noise_prior_scale_w <= 0.0:
        raise ValueError("Prior scales must be greater than zero")
    if args.burn_in < 0 or args.posterior_draws < 100 or args.thin < 1:
        raise ValueError("Invalid MCMC settings")
    if args.min_windows_per_parameter <= 0.0:
        raise ValueError("--min-windows-per-parameter must be greater than zero")
    if not 0.0 <= args.min_window_fraction <= 1.0:
        raise ValueError("--min-window-fraction must be in [0, 1]")


def split_point(window_count, train_fraction, strict=True):
    if strict and window_count < 7:
        raise ValueError(
            f"Only {window_count} complete windows overlap kernels and "
            "telemetry; at least 7 are required for a temporal holdout"
        )
    train_count = int(window_count * train_fraction)
    return min(max(train_count, 5), window_count - 2)


def window_durations_ns(window_rows):
    return np.asarray([row["duration_ns"] for row in window_rows], dtype=np.float64)


def activity_design(activity_ns, window_rows):
    """Intercept plus each family's busy fraction of the window's real length."""
    fraction = activity_ns / window_durations_ns(window_rows)[:, None]
    return np.column_stack([np.ones(len(fraction)), fraction])


def assign_splits(
    window_rows, train_fraction, window_ns, min_window_fraction, strict=True
):
    """Label windows train/test/excluded and return the training count.

    Windows shorter than ``min_window_fraction * window_ns`` are "excluded":
    their mean power is unreliable, so they are left out of fitting and
    evaluation. Only a session's final window can be short, so training is
    still the chronological prefix ``[:train_count]``.
    """
    complete = [
        row["duration_ns"] >= min_window_fraction * window_ns
        for row in window_rows
    ]
    train_count = split_point(sum(complete), train_fraction, strict)
    if not all(complete[:train_count]):
        raise ValueError("A truncated window falls inside the training prefix")
    for index, row in enumerate(window_rows):
        if index < train_count:
            row["split"] = "train"
        else:
            row["split"] = "test" if complete[index] else "excluded"
    return train_count


def split_mask(window_rows, *splits):
    return np.asarray([row["split"] in splits for row in window_rows])


def coefficient_summary_rows(terms, coefficient_draws, column_scale):
    rows = []
    for index, term in enumerate(terms):
        values = coefficient_draws[:, index]
        lower, upper = np.quantile(values, [0.025, 0.975])
        rows.append(
            {
                "term": term,
                "posterior_power_mean_w": float(values.mean()),
                "posterior_power_lower_95_w": float(lower),
                "posterior_power_upper_95_w": float(upper),
                "activity_rms_scale": float(column_scale[index]),
            }
        )
    return rows


def main():
    args = parse_args()
    validate_common_args(args)
    if args.max_families < 2:
        raise ValueError("--max-families must be at least 2")
    if not 0.5 <= args.train_fraction < 1.0:
        raise ValueError("--train-fraction must be in [0.5, 1.0)")

    window_ns = int(args.window_ms * 1_000_000)
    trace = load_power_trace(args.telemetry)
    print("Loading kernel events...", flush=True)
    table = load_kernels(args.kernels, load_event_ids=not args.skip_event_output)
    metadata = scan_kernel_families(table, args.max_families)
    analysis_start = max(trace.first_ns, metadata["first_start"])
    analysis_end = min(trace.last_ns, metadata["last_end"])
    if analysis_end <= analysis_start:
        raise ValueError("Telemetry and kernel timelines do not overlap")

    families = metadata["families"]
    family_index = {name: index for index, name in enumerate(families)}
    print(
        f"Using {len(families)} kernel families for "
        f"{metadata['valid_events']} events.",
        flush=True,
    )
    columns = family_columns(table, metadata["selected"], family_index)
    activity_ns, union_active_ns = aggregate_family_activity(
        table, columns, len(families), analysis_start, analysis_end, window_ns
    )
    window_rows, target_power_w = build_windows(
        trace,
        activity_ns,
        union_active_ns,
        analysis_start,
        analysis_end,
        window_ns,
        args.idle_power_mw,
    )
    train_count = assign_splits(
        window_rows, args.train_fraction, window_ns, args.min_window_fraction
    )
    design = activity_design(activity_ns, window_rows)
    test_mask = split_mask(window_rows, "test")
    terms = ["unattributed_baseline"] + families
    identifiability = check_identifiability(
        design[split_mask(window_rows, "train", "test")], train_count, terms,
        args.min_windows_per_parameter, args.allow_underdetermined,
    )
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
    predictions = summarize_predictions(prediction_draws, noise_draws, args.seed + 10)
    allocation = posterior_energy_allocation(
        coefficient_draws,
        activity_ns,
        attribution_pool_mj(window_rows, args.attribution),
        window_durations_ns(window_rows),
        args.attribution,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    add_prediction_columns(window_rows, predictions)
    apply_allocation_to_rows(window_rows, allocation)
    write_csv(
        args.output_dir / "bayesian_energy_windows.csv",
        window_rows,
        list(window_rows[0]),
    )

    coefficient_rows = coefficient_summary_rows(
        terms, coefficient_draws, column_scale
    )
    write_csv(
        args.output_dir / "bayesian_kernel_coefficients.csv",
        coefficient_rows,
        list(coefficient_rows[0]),
    )

    family_rows = []
    family_total_draws = allocation["family_total_draws"]
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
                "energy_posterior_median_mj": np.median(values),
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

    train_metrics = prediction_metrics(
        target_power_w, predictions, split_mask(window_rows, "train")
    )
    test_metrics = prediction_metrics(target_power_w, predictions, test_mask)
    session_rows, session_metrics = session_holdout_energy(window_rows, predictions)
    write_csv(
        args.output_dir / "bayesian_session_energy.csv",
        session_rows,
        list(session_rows[0]),
    )
    if args.plot:
        create_plots(
            args.output_dir, window_rows, family_rows,
            session_rows, session_metrics, test_metrics,
        )

    accounting = energy_accounting(window_rows, allocation)
    event_count = metadata["valid_events"]
    if not args.skip_event_output:
        print("Writing per-event Bayesian attribution...", flush=True)
        event_count, event_total = write_event_attribution(
            table,
            columns,
            families,
            args.output_dir / "bayesian_kernel_event_energy.csv",
            activity_ns,
            allocation,
            analysis_start,
            analysis_end,
            window_ns,
        )
        accounting["event_attributed_posterior_mean_mj"] = event_total

    summary = {
        "model": "nonnegative_bayesian_kernel_family_regression",
        "inference": "Gibbs sampling with half-normal coefficient priors",
        "attribution_mode": args.attribution,
        "attribution": (
            "Each window's measured dynamic energy is split in proportion to "
            "the posterior terms b0 + sum(beta_f * activity_f); the b0 share "
            "is reported as baseline energy. Family energy is distributed to "
            "events by temporal overlap."
            if args.attribution == "model"
            else "Legacy: dynamic energy scaled by the kernel-busy fraction is "
            "split among families by beta_f * activity_f, ignoring b0. Family "
            "energy is distributed to events by temporal overlap."
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
        "test_windows": int(test_mask.sum()),
        "excluded_short_windows": int(split_mask(window_rows, "excluded").sum()),
        "min_window_fraction": args.min_window_fraction,
        "kernel_events": event_count,
        "kernel_families": len(families),
        "design_matrix_rank": identifiability["design_matrix_rank"],
        "design_parameters": int(design.shape[1]),
        "identifiability": identifiability,
        "posterior_draws": args.posterior_draws,
        "posterior_noise_mean_w": float(np.sqrt(noise_draws).mean()),
        "train_power_metrics": train_metrics,
        "test_power_metrics": test_metrics,
        "test_session_energy_metrics": session_metrics,
        **accounting,
    }
    with (args.output_dir / "bayesian_kernel_energy_summary.json").open(
        "w", encoding="utf-8"
    ) as destination:
        json.dump(summary, destination, indent=2)
        destination.write("\n")

    print("=== Bayesian kernel-energy attribution ===")
    print(f"Attribution mode: {args.attribution}")
    print(f"Kernel families: {len(families)}")
    print(f"Kernel events: {event_count}")
    print(f"Test power RMSE: {test_metrics['rmse_w']:.3f} W")
    if test_metrics["r_squared"] is not None:
        print(f"Test power R-squared: {test_metrics['r_squared']:.4f}")
    print_coverage(test_metrics, session_metrics)
    print(f"Measured dynamic energy: {accounting['measured_dynamic_energy_mj']:.3f} mJ")
    print(
        "Bayesian attributed energy: "
        f"{accounting['attributed_posterior_mean_mj']:.3f} mJ"
    )
    print(f"Baseline (b0) energy: {accounting['baseline_posterior_mean_mj']:.3f} mJ")
    print(f"Unassigned energy: {accounting['unassigned_dynamic_energy_mj']:.3f} mJ")
    print(f"Conservation error: {accounting['conservation_error_mj']:.6g} mJ")
    print_identifiability_warnings(identifiability, short_kernel_name)
    print(f"Results: {args.output_dir}")


if __name__ == "__main__":
    main()
