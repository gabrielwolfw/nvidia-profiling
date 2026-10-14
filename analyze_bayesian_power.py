#!/usr/bin/env python3
"""Fit an initial Bayesian linear GPU power model from NVML and CUPTI data."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


NS_PER_SECOND = 1_000_000_000
FEATURE_NAMES = [
    "gpu_utilization_pct",
    "memory_utilization_pct",
    "graphics_clock_mhz",
    "temperature_c",
    "kernel_active_pct",
    "log_kernel_count",
    "kernel_concurrency",
    "gpu_clock_interaction",
    "memory_clock_interaction",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate synchronized NVML/CUPTI data into time windows and fit "
            "a regularized Bayesian linear power model."
        )
    )
    parser.add_argument("--telemetry", required=True, type=Path)
    parser.add_argument("--kernels", required=True, type=Path)
    parser.add_argument("--window-ms", type=float, default=200.0)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--min-telemetry-samples", type=int, default=2)
    parser.add_argument("--prior-scale", type=float, default=1.0)
    parser.add_argument("--posterior-draws", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("bayesian-power-analysis")
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate diagnostic PNG plots (requires matplotlib).",
    )
    return parser.parse_args()


def require_fields(reader, required, label):
    fields = set(reader.fieldnames or [])
    missing = required - fields
    if missing:
        raise ValueError(f"{label} CSV is missing columns: {sorted(missing)}")


def load_telemetry(path, window_ns):
    required = {
        "timestamp_ns",
        "power_mw",
        "temperature_c",
        "gpu_utilization_pct",
        "memory_utilization_pct",
        "graphics_clock_mhz",
    }
    samples = []
    skipped = 0
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        require_fields(reader, required, "telemetry")
        for row in reader:
            try:
                sample = {
                    "timestamp_ns": int(row["timestamp_ns"]),
                    "power_w": float(row["power_mw"]) / 1000.0,
                    "temperature_c": float(row["temperature_c"]),
                    "gpu_utilization_pct": float(row["gpu_utilization_pct"]),
                    "memory_utilization_pct": float(
                        row["memory_utilization_pct"]
                    ),
                    "graphics_clock_mhz": float(row["graphics_clock_mhz"]),
                }
            except (TypeError, ValueError):
                skipped += 1
                continue
            if not all(math.isfinite(value) for value in sample.values()):
                skipped += 1
                continue
            samples.append(sample)

    if skipped:
        print(f"WARNING: Skipped {skipped} invalid telemetry rows.")
    if len(samples) < 3:
        raise ValueError("At least three valid telemetry samples are required")
    samples.sort(key=lambda sample: sample["timestamp_ns"])
    if samples[0]["timestamp_ns"] == samples[-1]["timestamp_ns"]:
        raise ValueError("Telemetry must cover more than one timestamp")

    origin_ns = samples[0]["timestamp_ns"]
    window_count = (
        (samples[-1]["timestamp_ns"] - origin_ns) // window_ns + 1
    )
    sums = {
        key: np.zeros(window_count, dtype=np.float64)
        for key in (
            "power_w",
            "temperature_c",
            "gpu_utilization_pct",
            "memory_utilization_pct",
            "graphics_clock_mhz",
        )
    }
    counts = np.zeros(window_count, dtype=np.int64)

    for sample in samples:
        index = (sample["timestamp_ns"] - origin_ns) // window_ns
        counts[index] += 1
        for key in sums:
            sums[key][index] += sample[key]

    return origin_ns, window_count, counts, sums


def aggregate_kernels(path, origin_ns, window_count, window_ns):
    required = {"start_ns", "end_ns"}
    analysis_end_ns = origin_ns + window_count * window_ns
    kernel_count = np.zeros(window_count, dtype=np.int64)
    duration_sum_ns = np.zeros(window_count, dtype=np.float64)
    active_ns = np.zeros(window_count, dtype=np.float64)
    union_end_ns = np.full(window_count, -1, dtype=np.int64)
    valid = 0
    skipped = 0
    previous_start = None

    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        require_fields(reader, required, "kernel activity")
        for row in reader:
            try:
                start_ns = int(row["start_ns"])
                end_ns = int(row["end_ns"])
            except (TypeError, ValueError):
                skipped += 1
                continue
            if end_ns <= start_ns:
                skipped += 1
                continue
            if previous_start is not None and start_ns < previous_start:
                raise ValueError(
                    "Kernel activity rows must be sorted by start_ns for "
                    "streaming aggregation"
                )
            previous_start = start_ns
            valid += 1

            if origin_ns <= start_ns < analysis_end_ns:
                start_index = (start_ns - origin_ns) // window_ns
                kernel_count[start_index] += 1

            clipped_start = max(start_ns, origin_ns)
            clipped_end = min(end_ns, analysis_end_ns)
            if clipped_end <= clipped_start:
                continue

            first_index = (clipped_start - origin_ns) // window_ns
            last_index = (clipped_end - 1 - origin_ns) // window_ns
            for index in range(first_index, last_index + 1):
                window_start = origin_ns + index * window_ns
                window_end = window_start + window_ns
                overlap_start = max(clipped_start, window_start)
                overlap_end = min(clipped_end, window_end)
                duration_sum_ns[index] += overlap_end - overlap_start

                previous_end = union_end_ns[index]
                if overlap_end > previous_end:
                    uncovered_start = max(overlap_start, previous_end)
                    active_ns[index] += overlap_end - uncovered_start
                    union_end_ns[index] = overlap_end

    if skipped:
        print(f"WARNING: Skipped {skipped} invalid kernel rows.")
    print(f"Aggregated {valid} kernel events.")
    return kernel_count, duration_sum_ns, active_ns


def build_dataset(
    origin_ns,
    window_count,
    window_ns,
    telemetry_counts,
    telemetry_sums,
    kernel_count,
    duration_sum_ns,
    active_ns,
    min_telemetry_samples,
):
    rows = []
    features = []
    targets = []

    for index in range(window_count):
        count = int(telemetry_counts[index])
        if count < min_telemetry_samples:
            continue
        averages = {
            key: float(values[index] / count)
            for key, values in telemetry_sums.items()
        }
        kernel_active_pct = min(100.0, active_ns[index] / window_ns * 100.0)
        concurrency = duration_sum_ns[index] / window_ns
        log_kernel_count = math.log1p(kernel_count[index])
        clock_ghz = averages["graphics_clock_mhz"] / 1000.0
        feature_values = [
            averages["gpu_utilization_pct"],
            averages["memory_utilization_pct"],
            averages["graphics_clock_mhz"],
            averages["temperature_c"],
            kernel_active_pct,
            log_kernel_count,
            concurrency,
            averages["gpu_utilization_pct"] * clock_ghz,
            averages["memory_utilization_pct"] * clock_ghz,
        ]
        window_start_ns = origin_ns + index * window_ns
        window_end_ns = window_start_ns + window_ns
        rows.append(
            {
                "window_index": index,
                "window_start_ns": window_start_ns,
                "window_end_ns": window_end_ns,
                "telemetry_samples": count,
                "measured_power_w": averages["power_w"],
                "measured_energy_j": averages["power_w"]
                * window_ns
                / NS_PER_SECOND,
                "temperature_c": averages["temperature_c"],
                "gpu_utilization_pct": averages["gpu_utilization_pct"],
                "memory_utilization_pct": averages[
                    "memory_utilization_pct"
                ],
                "graphics_clock_mhz": averages["graphics_clock_mhz"],
                "kernel_active_pct": kernel_active_pct,
                "kernel_count": int(kernel_count[index]),
                "log_kernel_count": log_kernel_count,
                "kernel_concurrency": concurrency,
                "gpu_clock_interaction": feature_values[-2],
                "memory_clock_interaction": feature_values[-1],
            }
        )
        features.append(feature_values)
        targets.append(averages["power_w"])

    if len(rows) < 10:
        raise ValueError(
            "At least ten complete windows are required after telemetry filtering"
        )
    return rows, np.asarray(features), np.asarray(targets)


def fit_bayesian_regression(
    features, targets, train_count, prior_scale, posterior_draws, seed
):
    train_features = features[:train_count]
    train_targets = targets[:train_count]
    feature_mean = train_features.mean(axis=0)
    feature_scale = train_features.std(axis=0, ddof=1)
    feature_scale[feature_scale < 1e-12] = 1.0
    target_mean = float(train_targets.mean())
    target_scale = float(train_targets.std(ddof=1))
    if target_scale < 1e-12:
        raise ValueError("Training power measurements have no variation")

    standardized = (features - feature_mean) / feature_scale
    design = np.column_stack([np.ones(len(features)), standardized])
    train_design = design[:train_count]
    train_target = (train_targets - target_mean) / target_scale

    parameter_count = train_design.shape[1]
    prior_mean = np.zeros(parameter_count)
    prior_variance = np.full(parameter_count, prior_scale**2)
    prior_variance[0] = 25.0
    prior_precision = np.diag(1.0 / prior_variance)

    posterior_precision = prior_precision + train_design.T @ train_design
    posterior_covariance = np.linalg.inv(posterior_precision)
    posterior_mean = posterior_covariance @ (
        train_design.T @ train_target
    )
    shape = 2.0 + train_count / 2.0
    rate = 1.0 + 0.5 * (
        train_target @ train_target
        - posterior_mean @ posterior_precision @ posterior_mean
    )
    if rate <= 0.0:
        raise ValueError("Bayesian variance posterior is not positive")

    rng = np.random.default_rng(seed)
    noise_variance = 1.0 / rng.gamma(
        shape, scale=1.0 / rate, size=posterior_draws
    )
    covariance_root = np.linalg.cholesky(posterior_covariance)
    standard_normal = rng.normal(size=(posterior_draws, parameter_count))
    coefficient_draws = posterior_mean + (
        standard_normal @ covariance_root.T
    ) * np.sqrt(noise_variance)[:, None]

    mean_draws_standardized = coefficient_draws @ design.T
    mean_draws_w = target_mean + target_scale * mean_draws_standardized
    predictive_draws_w = mean_draws_w + (
        rng.normal(size=mean_draws_w.shape)
        * np.sqrt(noise_variance)[:, None]
        * target_scale
    )

    return {
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "target_mean": target_mean,
        "target_scale": target_scale,
        "coefficient_draws": coefficient_draws,
        "mean_draws_w": mean_draws_w,
        "predictive_draws_w": predictive_draws_w,
    }


def regression_metrics(actual, predicted, lower, upper):
    residual = actual - predicted
    squared_error = float(np.sum(residual**2))
    total_variation = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "count": int(len(actual)),
        "mae_w": float(np.mean(np.abs(residual))),
        "rmse_w": float(np.sqrt(np.mean(residual**2))),
        "r_squared": (
            1.0 - squared_error / total_variation
            if total_variation > 0.0
            else None
        ),
        "predictive_interval_95_coverage": float(
            np.mean((actual >= lower) & (actual <= upper))
        ),
    }


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(args, rows, targets, train_count, fit, window_ns):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mean_draws = fit["mean_draws_w"]
    predictive_draws = fit["predictive_draws_w"]
    predicted = mean_draws.mean(axis=0)
    credible_lower, credible_upper = np.quantile(
        mean_draws, [0.025, 0.975], axis=0
    )
    predictive_lower, predictive_upper = np.quantile(
        predictive_draws, [0.025, 0.975], axis=0
    )

    prediction_rows = []
    for index, row in enumerate(rows):
        output = dict(row)
        output.update(
            {
                "split": "train" if index < train_count else "test",
                "predicted_power_w": predicted[index],
                "credible_power_lower_w": credible_lower[index],
                "credible_power_upper_w": credible_upper[index],
                "predictive_power_lower_w": predictive_lower[index],
                "predictive_power_upper_w": predictive_upper[index],
                "predicted_energy_j": predicted[index]
                * window_ns
                / NS_PER_SECOND,
                "power_residual_w": targets[index] - predicted[index],
            }
        )
        prediction_rows.append(output)

    prediction_fields = list(prediction_rows[0])
    write_csv(
        args.output_dir / "bayesian_window_predictions.csv",
        prediction_rows,
        prediction_fields,
    )

    coefficient_draws = fit["coefficient_draws"]
    coefficient_rows = []
    names = ["intercept_at_training_mean"] + FEATURE_NAMES
    for index, name in enumerate(names):
        values = coefficient_draws[:, index] * fit["target_scale"]
        if index == 0:
            values = values + fit["target_mean"]
            unit = "W at training feature means"
        else:
            unit = "W per one training standard deviation"
        lower, upper = np.quantile(values, [0.025, 0.975])
        coefficient_rows.append(
            {
                "term": name,
                "posterior_mean": float(values.mean()),
                "credible_lower_95": float(lower),
                "credible_upper_95": float(upper),
                "unit": unit,
                "training_feature_mean": (
                    "" if index == 0 else fit["feature_mean"][index - 1]
                ),
                "training_feature_stddev": (
                    "" if index == 0 else fit["feature_scale"][index - 1]
                ),
            }
        )
    write_csv(
        args.output_dir / "bayesian_coefficients.csv",
        coefficient_rows,
        list(coefficient_rows[0]),
    )

    train_metrics = regression_metrics(
        targets[:train_count],
        predicted[:train_count],
        predictive_lower[:train_count],
        predictive_upper[:train_count],
    )
    test_metrics = regression_metrics(
        targets[train_count:],
        predicted[train_count:],
        predictive_lower[train_count:],
        predictive_upper[train_count:],
    )
    baseline = np.full(len(targets) - train_count, targets[:train_count].mean())
    baseline_rmse = float(
        np.sqrt(np.mean((targets[train_count:] - baseline) ** 2))
    )
    summary = {
        "model": "conjugate_bayesian_linear_regression",
        "interpretation": (
            "Initial within-run diagnostic model; validation across independent "
            "runs is required before general use."
        ),
        "window_ms": args.window_ms,
        "complete_windows": len(rows),
        "train_windows": train_count,
        "test_windows": len(rows) - train_count,
        "train_fraction": args.train_fraction,
        "posterior_draws": args.posterior_draws,
        "prior_scale": args.prior_scale,
        "features": FEATURE_NAMES,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "test_baseline_mean_power_rmse_w": baseline_rmse,
    }
    with (args.output_dir / "bayesian_model_summary.json").open(
        "w", encoding="utf-8"
    ) as destination:
        json.dump(summary, destination, indent=2)
        destination.write("\n")

    print("=== Bayesian GPU power model ===")
    print(f"Complete windows: {len(rows)}")
    print(f"Training windows: {train_count}")
    print(f"Test windows: {len(rows) - train_count}")
    print(f"Test MAE: {test_metrics['mae_w']:.3f} W")
    print(f"Test RMSE: {test_metrics['rmse_w']:.3f} W")
    if test_metrics["r_squared"] is not None:
        print(f"Test R-squared: {test_metrics['r_squared']:.4f}")
    print(
        "Test 95% predictive coverage: "
        f"{test_metrics['predictive_interval_95_coverage'] * 100:.1f}%"
    )
    print(f"Mean-power baseline RMSE: {baseline_rmse:.3f} W")
    print(f"Results: {args.output_dir}")

    if args.plot:
        create_plots(
            args.output_dir,
            rows,
            targets,
            predicted,
            predictive_lower,
            predictive_upper,
            train_count,
        )


def create_plots(
    output_dir,
    rows,
    targets,
    predicted,
    predictive_lower,
    predictive_upper,
    train_count,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("--plot requires matplotlib") from error

    elapsed_seconds = np.asarray(
        [
            (row["window_start_ns"] - rows[0]["window_start_ns"])
            / NS_PER_SECOND
            for row in rows
        ]
    )
    figure, axis = plt.subplots(figsize=(11, 5))
    axis.plot(elapsed_seconds, targets, label="Measured", linewidth=1.2)
    axis.plot(elapsed_seconds, predicted, label="Posterior mean", linewidth=1.2)
    axis.fill_between(
        elapsed_seconds,
        predictive_lower,
        predictive_upper,
        alpha=0.2,
        label="95% predictive interval",
    )
    axis.axvline(
        elapsed_seconds[train_count],
        color="black",
        linestyle="--",
        linewidth=1,
        label="Test split",
    )
    axis.set_xlabel("Elapsed time (s)")
    axis.set_ylabel("Power (W)")
    axis.set_title("Bayesian GPU power prediction")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "bayesian_power_timeline.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(targets[train_count:], predicted[train_count:], alpha=0.65)
    limits = [
        min(targets[train_count:].min(), predicted[train_count:].min()),
        max(targets[train_count:].max(), predicted[train_count:].max()),
    ]
    axis.plot(limits, limits, color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("Measured power (W)")
    axis.set_ylabel("Predicted power (W)")
    axis.set_title("Temporal holdout predictions")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "bayesian_power_test_scatter.png", dpi=160)
    plt.close(figure)


def main():
    args = parse_args()
    if args.window_ms <= 0.0:
        raise ValueError("--window-ms must be greater than zero")
    if not 0.5 <= args.train_fraction < 1.0:
        raise ValueError("--train-fraction must be in [0.5, 1.0)")
    if args.min_telemetry_samples < 1:
        raise ValueError("--min-telemetry-samples must be positive")
    if args.prior_scale <= 0.0:
        raise ValueError("--prior-scale must be greater than zero")
    if args.posterior_draws < 100:
        raise ValueError("--posterior-draws must be at least 100")

    window_ns = int(args.window_ms * 1_000_000)
    if window_ns < 1:
        raise ValueError("--window-ms must be at least one nanosecond")

    origin_ns, window_count, telemetry_counts, telemetry_sums = load_telemetry(
        args.telemetry, window_ns
    )
    kernel_count, duration_sum_ns, active_ns = aggregate_kernels(
        args.kernels, origin_ns, window_count, window_ns
    )
    rows, features, targets = build_dataset(
        origin_ns,
        window_count,
        window_ns,
        telemetry_counts,
        telemetry_sums,
        kernel_count,
        duration_sum_ns,
        active_ns,
        args.min_telemetry_samples,
    )
    train_count = int(len(rows) * args.train_fraction)
    train_count = min(max(train_count, 5), len(rows) - 2)
    fit = fit_bayesian_regression(
        features,
        targets,
        train_count,
        args.prior_scale,
        args.posterior_draws,
        args.seed,
    )
    write_outputs(args, rows, targets, train_count, fit, window_ns)


if __name__ == "__main__":
    main()
