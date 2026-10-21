#!/usr/bin/env python3
"""Fit and visualize a Bayesian window-energy model across sessions."""

import argparse
import json
from pathlib import Path

import numpy as np

from bayesian_energy_model import (
    FEATURE_NAMES,
    NS_PER_SECOND,
    aggregate_kernels,
    build_dataset,
    fit_bayesian_regression,
    load_telemetry,
    regression_metrics,
    write_csv,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fit one Bayesian GPU energy model across independent session "
            "directories and evaluate the final temporal fraction of every "
            "session as a holdout."
        )
    )
    parser.add_argument("--sessions-root", required=True, type=Path)
    parser.add_argument("--window-ms", type=float, default=200.0)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--min-telemetry-samples", type=int, default=2)
    parser.add_argument("--prior-scale", type=float, default=1.0)
    parser.add_argument("--posterior-draws", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("bayesian-energy-multisession-analysis"),
    )
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def discover_sessions(root):
    sessions = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        telemetry = directory / "gpu_telemetry.csv"
        kernels = directory / "kernel_activity.csv"
        if telemetry.is_file() and kernels.is_file():
            sessions.append((directory.name, telemetry, kernels))
    if len(sessions) < 2:
        raise ValueError(
            "--sessions-root must contain at least two session directories "
            "with gpu_telemetry.csv and kernel_activity.csv"
        )
    return sessions


def load_session(name, telemetry, kernels, args, window_ns):
    print(f"\n=== Aggregating {name} ===")
    origin_ns, window_count, counts, sums = load_telemetry(
        telemetry, window_ns
    )
    kernel_count, duration_sum_ns, active_ns = aggregate_kernels(
        kernels, origin_ns, window_count, window_ns
    )
    rows, features, targets = build_dataset(
        origin_ns,
        window_count,
        window_ns,
        counts,
        sums,
        kernel_count,
        duration_sum_ns,
        active_ns,
        args.min_telemetry_samples,
    )
    split = int(len(rows) * args.train_fraction)
    split = min(max(split, 5), len(rows) - 2)
    for position, row in enumerate(rows):
        row["session"] = name
        row["session_window_index"] = position
        row["session_elapsed_s"] = (
            row["window_start_ns"] - origin_ns
        ) / NS_PER_SECOND
        row["split"] = "train" if position < split else "test"
    return rows, features, targets, split


def posterior_arrays(fit):
    mean_draws = fit["mean_draws_w"]
    predictive_draws = fit["predictive_draws_w"]
    return (
        mean_draws.mean(axis=0),
        *np.quantile(mean_draws, [0.025, 0.975], axis=0),
        *np.quantile(predictive_draws, [0.025, 0.975], axis=0),
    )


def coefficient_rows(fit):
    rows = []
    for index, name in enumerate(["intercept_at_training_mean"] + FEATURE_NAMES):
        values = fit["coefficient_draws"][:, index] * fit["target_scale"]
        if index == 0:
            values = values + fit["target_mean"]
            unit = "J per window at training feature means"
        else:
            unit = "J per window per one training standard deviation"
        lower, upper = np.quantile(values, [0.025, 0.975])
        rows.append(
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
    return rows


def create_plots(
    output_dir, rows, power_targets, predicted_power, power_lower,
    power_upper, energy_targets, predicted_energy, energy_lower,
    energy_upper, train_count, session_energy_rows,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("--plot requires matplotlib") from error

    test_rows = rows[train_count:]
    actual = power_targets[train_count:]
    expected = predicted_power[train_count:]
    test_lower = power_lower[train_count:]
    test_upper = power_upper[train_count:]
    names = list(dict.fromkeys(row["session"] for row in test_rows))
    colors = plt.get_cmap("tab20", len(names))

    figure, (fit_axis, residual_axis) = plt.subplots(
        2, 1, figsize=(11, 10), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]},
    )
    for session_index, name in enumerate(names):
        indices = np.asarray(
            [index for index, row in enumerate(test_rows) if row["session"] == name]
        )
        color = colors(session_index)
        fit_axis.vlines(
            actual[indices], test_lower[indices], test_upper[indices],
            color=color, alpha=0.15, linewidth=0.8,
        )
        fit_axis.scatter(
            actual[indices], expected[indices], s=22, alpha=0.75,
            color=color, label=name,
        )
        residual_axis.scatter(
            actual[indices], expected[indices] - actual[indices],
            s=22, alpha=0.75, color=color,
        )

    minimum = float(min(actual.min(), expected.min(), test_lower.min()))
    maximum = float(max(actual.max(), expected.max(), test_upper.max()))
    fit_axis.plot(
        [minimum, maximum], [minimum, maximum], color="black",
        linestyle="--", linewidth=1.2, label="Central line: predicted = measured",
    )
    fit_axis.set_ylabel("Posterior predicted power (W)")
    fit_axis.set_title("Temporal holdout predictions across Thor workloads")
    fit_axis.grid(alpha=0.2)
    fit_axis.legend(fontsize=7, ncol=2, loc="best")
    residual_axis.axhline(0.0, color="black", linestyle="--", linewidth=1.2)
    residual_axis.set_xlabel("Measured power (W)")
    residual_axis.set_ylabel("Distance from central line (W)")
    residual_axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(
        output_dir / "bayesian_power_temporal_holdout.png", dpi=180
    )
    plt.close(figure)

    summaries = []
    for name in names:
        indices = np.asarray(
            [index for index, row in enumerate(test_rows) if row["session"] == name]
        )
        summaries.append(
            (
                name,
                float(actual[indices].mean()),
                float(expected[indices].mean()),
            )
        )
    figure, axis = plt.subplots(figsize=(8, 7))
    for session_index, (name, measured, modeled) in enumerate(summaries):
        axis.scatter(
            measured, modeled, s=75, color=colors(session_index), label=name
        )
    minimum = min(min(item[1], item[2]) for item in summaries)
    maximum = max(max(item[1], item[2]) for item in summaries)
    axis.plot(
        [minimum, maximum], [minimum, maximum], color="black",
        linestyle="--", linewidth=1.2,
    )
    axis.set_xlabel("Mean measured holdout power (W)")
    axis.set_ylabel("Mean predicted holdout power (W)")
    axis.set_title("Bayesian power by workload")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=7, loc="best")
    figure.tight_layout()
    figure.savefig(output_dir / "bayesian_power_by_workload.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 7))
    measured_energy = np.asarray(
        [float(row["measured_energy_j"]) for row in session_energy_rows]
    )
    session_predicted_energy = np.asarray(
        [float(row["predicted_energy_j"]) for row in session_energy_rows]
    )
    for session_index, row in enumerate(session_energy_rows):
        axis.scatter(
            measured_energy[session_index], session_predicted_energy[session_index],
            s=75, color=colors(session_index), label=row["session"],
        )
    minimum = float(min(measured_energy.min(), session_predicted_energy.min()))
    maximum = float(max(measured_energy.max(), session_predicted_energy.max()))
    axis.plot(
        [minimum, maximum], [minimum, maximum], color="black",
        linestyle="--", linewidth=1.2,
        label="Central line: predicted = measured",
    )
    axis.set_xlabel("Measured session energy (J)")
    axis.set_ylabel("Bayesian predicted session energy (J)")
    axis.set_title("Bayesian energy across Thor workloads")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=7, ncol=2, loc="best")
    figure.tight_layout()
    figure.savefig(output_dir / "bayesian_energy_by_workload.png", dpi=180)
    plt.close(figure)

    names = list(dict.fromkeys(row["session"] for row in rows))
    figure, axes = plt.subplots(
        len(names), 1, figsize=(12, 2.25 * len(names)), sharex=False,
        constrained_layout=True,
    )
    if len(names) == 1:
        axes = [axes]
    for session_index, (name, axis) in enumerate(zip(names, axes)):
        indices = sorted(
            (index for index, row in enumerate(rows) if row["session"] == name),
            key=lambda index: rows[index]["session_window_index"],
        )
        elapsed = np.asarray([rows[index]["session_elapsed_s"] for index in indices])
        measured = energy_targets[indices]
        modeled = predicted_energy[indices]
        interval_lower = energy_lower[indices]
        interval_upper = energy_upper[indices]
        test_positions = [
            position for position, index in enumerate(indices)
            if rows[index]["split"] == "test"
        ]
        axis.plot(elapsed, measured, color="black", linewidth=1.0, label="Measured")
        axis.plot(
            elapsed, modeled, color=colors(session_index), linewidth=1.1,
            label="Posterior mean",
        )
        axis.fill_between(
            elapsed, interval_lower, interval_upper,
            color=colors(session_index), alpha=0.18,
            label="95% predictive interval",
        )
        if test_positions:
            test_start = elapsed[test_positions[0]]
            axis.axvspan(
                test_start, elapsed[-1], color="grey", alpha=0.10,
                label="Temporal holdout",
            )
        axis.set_ylabel("Energy (J/window)")
        axis.set_title(name, loc="left", fontsize=9)
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8, ncol=4, loc="upper right")
    axes[-1].set_xlabel("Elapsed session time (s)")
    figure.suptitle(
        "Bayesian energy timeline across Thor workloads", fontsize=15
    )
    figure.savefig(output_dir / "bayesian_energy_timeline.png", dpi=180)
    plt.close(figure)

    test_energy = energy_targets[train_count:]
    test_predicted_energy = predicted_energy[train_count:]
    test_energy_lower = energy_lower[train_count:]
    test_energy_upper = energy_upper[train_count:]
    figure, (fit_axis, residual_axis) = plt.subplots(
        2, 1, figsize=(11, 10), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]},
    )
    for session_index, name in enumerate(names):
        indices = np.asarray(
            [index for index, row in enumerate(test_rows) if row["session"] == name]
        )
        color = colors(session_index)
        fit_axis.vlines(
            test_energy[indices], test_energy_lower[indices],
            test_energy_upper[indices], color=color, alpha=0.15, linewidth=0.8,
        )
        fit_axis.scatter(
            test_energy[indices], test_predicted_energy[indices],
            s=22, alpha=0.75, color=color, label=name,
        )
        residual_axis.scatter(
            test_energy[indices],
            test_predicted_energy[indices] - test_energy[indices],
            s=22, alpha=0.75, color=color,
        )
    minimum = float(
        min(test_energy.min(), test_predicted_energy.min(), test_energy_lower.min())
    )
    maximum = float(
        max(test_energy.max(), test_predicted_energy.max(), test_energy_upper.max())
    )
    fit_axis.plot(
        [minimum, maximum], [minimum, maximum], color="black",
        linestyle="--", linewidth=1.2, label="Central line: predicted = measured",
    )
    fit_axis.set_ylabel("Posterior predicted energy (J/window)")
    fit_axis.set_title("Bayesian energy temporal holdout across Thor workloads")
    fit_axis.grid(alpha=0.2)
    fit_axis.legend(fontsize=7, ncol=2, loc="best")
    residual_axis.axhline(0.0, color="black", linestyle="--", linewidth=1.2)
    residual_axis.set_xlabel("Measured energy (J/window)")
    residual_axis.set_ylabel("Distance from central line (J/window)")
    residual_axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(
        output_dir / "bayesian_energy_temporal_holdout.png", dpi=180
    )
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
    sessions = discover_sessions(args.sessions_root)
    train_parts = []
    test_parts = []
    session_counts = {}
    for name, telemetry, kernels in sessions:
        rows, features, targets, split = load_session(
            name, telemetry, kernels, args, window_ns
        )
        train_parts.append((rows[:split], features[:split], targets[:split]))
        test_parts.append((rows[split:], features[split:], targets[split:]))
        session_counts[name] = {
            "windows": len(rows), "train_windows": split,
            "test_windows": len(rows) - split,
        }

    ordered_parts = train_parts + test_parts
    rows = [row for part, _, _ in ordered_parts for row in part]
    features = np.concatenate([part for _, part, _ in ordered_parts])
    power_targets = np.concatenate([part for _, _, part in ordered_parts])
    window_seconds = window_ns / NS_PER_SECOND
    energy_targets = power_targets * window_seconds
    train_count = sum(len(part) for part, _, _ in train_parts)
    fit = fit_bayesian_regression(
        features, energy_targets, train_count, args.prior_scale,
        args.posterior_draws, args.seed,
    )
    (
        predicted_energy,
        credible_energy_lower,
        credible_energy_upper,
        energy_lower,
        energy_upper,
    ) = posterior_arrays(fit)
    predicted_power = predicted_energy / window_seconds
    credible_power_lower = credible_energy_lower / window_seconds
    credible_power_upper = credible_energy_upper / window_seconds
    power_lower = energy_lower / window_seconds
    power_upper = energy_upper / window_seconds

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_rows = []
    for index, row in enumerate(rows):
        output = dict(row)
        output.update(
            {
                "predicted_energy_j": predicted_energy[index],
                "credible_energy_lower_j": credible_energy_lower[index],
                "credible_energy_upper_j": credible_energy_upper[index],
                "predictive_energy_lower_j": energy_lower[index],
                "predictive_energy_upper_j": energy_upper[index],
                "energy_residual_j": energy_targets[index] - predicted_energy[index],
                "predicted_power_w": predicted_power[index],
                "credible_power_lower_w": credible_power_lower[index],
                "credible_power_upper_w": credible_power_upper[index],
                "predictive_power_lower_w": power_lower[index],
                "predictive_power_upper_w": power_upper[index],
                "power_residual_w": power_targets[index] - predicted_power[index],
            }
        )
        output_rows.append(output)
    write_csv(
        args.output_dir / "bayesian_multisession_predictions.csv",
        output_rows, list(output_rows[0]),
    )
    coefficients = coefficient_rows(fit)
    write_csv(
        args.output_dir / "bayesian_multisession_coefficients.csv",
        coefficients, list(coefficients[0]),
    )

    session_energy_rows = []
    for name, _, _ in sessions:
        indices = [
            index for index, row in enumerate(rows) if row["session"] == name
        ]
        measured_energy = float(sum(rows[index]["measured_energy_j"] for index in indices))
        modeled_energy = float(
            sum(predicted_energy[index] for index in indices)
        )
        session_energy_rows.append(
            {
                "session": name,
                "windows": len(indices),
                "measured_energy_j": measured_energy,
                "predicted_energy_j": modeled_energy,
                "energy_error_j": modeled_energy - measured_energy,
                "energy_error_pct": (
                    100.0 * (modeled_energy - measured_energy) / measured_energy
                    if measured_energy else ""
                ),
            }
        )
    write_csv(
        args.output_dir / "bayesian_session_energy.csv",
        session_energy_rows, list(session_energy_rows[0]),
    )

    power_metrics = regression_metrics(
        power_targets[train_count:], predicted_power[train_count:],
        power_lower[train_count:], power_upper[train_count:],
    )
    raw_energy_metrics = regression_metrics(
        energy_targets[train_count:], predicted_energy[train_count:],
        energy_lower[train_count:], energy_upper[train_count:],
    )
    energy_metrics = {
        "count": raw_energy_metrics["count"],
        "mae_j_per_window": raw_energy_metrics["mae_w"],
        "rmse_j_per_window": raw_energy_metrics["rmse_w"],
        "r_squared": raw_energy_metrics["r_squared"],
        "predictive_interval_95_coverage": raw_energy_metrics[
            "predictive_interval_95_coverage"
        ],
    }
    summary = {
        "model": "multisession_conjugate_bayesian_linear_energy_regression",
        "target": "measured_energy_j_per_fixed_window",
        "validation": "final temporal fraction held out independently per session",
        "sessions_root": str(args.sessions_root),
        "session_count": len(sessions),
        "window_ms": args.window_ms,
        "train_fraction_per_session": args.train_fraction,
        "train_windows": train_count,
        "test_windows": len(rows) - train_count,
        "posterior_draws": args.posterior_draws,
        "features": FEATURE_NAMES,
        "session_windows": session_counts,
        "temporal_holdout_energy_metrics": energy_metrics,
        "derived_temporal_holdout_power_metrics": power_metrics,
    }
    with (args.output_dir / "bayesian_multisession_summary.json").open(
        "w", encoding="utf-8"
    ) as destination:
        json.dump(summary, destination, indent=2)
        destination.write("\n")

    print("\n=== Multi-session Bayesian GPU energy model ===")
    print(f"Sessions: {len(sessions)}")
    print(f"Training windows: {train_count}")
    print(f"Temporal holdout windows: {len(rows) - train_count}")
    print(f"Holdout energy MAE: {energy_metrics['mae_j_per_window']:.4f} J/window")
    print(f"Holdout energy RMSE: {energy_metrics['rmse_j_per_window']:.4f} J/window")
    if energy_metrics["r_squared"] is not None:
        print(f"Holdout energy R-squared: {energy_metrics['r_squared']:.4f}")
    print(f"Derived holdout power MAE: {power_metrics['mae_w']:.3f} W")
    print(f"Results: {args.output_dir}")

    if args.plot:
        create_plots(
            args.output_dir, rows, power_targets, predicted_power, power_lower,
            power_upper, energy_targets, predicted_energy, energy_lower,
            energy_upper, train_count, session_energy_rows,
        )


if __name__ == "__main__":
    main()
