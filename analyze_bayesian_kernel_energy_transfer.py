#!/usr/bin/env python3
"""Evaluate Bayesian kernel-energy transfer between GPU platforms."""

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_bayesian_kernel_energy import (
    activity_design,
    add_model_check_arguments,
    attribution_pool_mj,
    energy_accounting,
    fit_nonnegative_bayesian_regression,
    plot_kernel_family_energy,
    posterior_energy_allocation,
    regression_metrics,
    scan_kernel_families,
    short_kernel_name,
    split_mask,
    validate_common_args,
    window_durations_ns,
)
from analyze_bayesian_kernel_energy_multisession import (
    build_session_windows,
    discover_session_paths,
    scan_sessions,
)
from energy_attribution.diagnostics import (
    check_identifiability,
    print_identifiability_warnings,
)
from energy_attribution.io import NS_PER_SECOND, load_kernels, write_csv
from energy_attribution.windows import OTHER_FAMILY


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train a pooled Bayesian kernel-family model on one collection of "
            "sessions, then compare zero-shot and minimally calibrated "
            "predictions on one external GPU session."
        )
    )
    parser.add_argument("--train-sessions-root", required=True, type=Path)
    parser.add_argument(
        "--external-test-session",
        required=True,
        type=Path,
        help="Directory containing gpu_telemetry.csv and kernel_activity.csv.",
    )
    parser.add_argument("--window-ms", type=float, default=200.0)
    parser.add_argument("--idle-power-mw", type=float, default=0.0)
    parser.add_argument(
        "--max-families",
        type=int,
        default=24,
        help=(
            "Maximum family columns, including __other__. Keeping this below "
            "the number of training families gives unseen kernels a learned "
            "fallback coefficient."
        ),
    )
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.2,
        help=(
            "Initial fraction of the external session used only to estimate "
            "a non-negative offset and scale."
        ),
    )
    parser.add_argument("--prior-scale-w", type=float, default=250.0)
    parser.add_argument("--noise-prior-scale-w", type=float, default=20.0)
    parser.add_argument("--burn-in", type=int, default=1000)
    parser.add_argument("--posterior-draws", type=int, default=2000)
    parser.add_argument("--thin", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    add_model_check_arguments(parser)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def validate_args(args):
    validate_common_args(args)
    if args.max_families < 3:
        raise ValueError("--max-families must be at least 3")
    if not 0.05 <= args.calibration_fraction <= 0.5:
        raise ValueError("--calibration-fraction must be in [0.05, 0.5]")


def require_session_files(directory):
    telemetry = directory / "gpu_telemetry.csv"
    kernels = directory / "kernel_activity.csv"
    missing = [str(path) for path in (telemetry, kernels) if not path.is_file()]
    if missing:
        raise ValueError("External test session is missing: " + ", ".join(missing))
    return telemetry, kernels


def training_family_basis(scan):
    """Return a train-only family basis with a learned fallback column."""
    families = list(scan["families"])
    selected = set(scan["selected"])
    if OTHER_FAMILY not in families:
        fallback_source = min(
            families, key=lambda name: scan["duration_by_name"][name]
        )
        families.remove(fallback_source)
        families.append(OTHER_FAMILY)
        selected.remove(fallback_source)
        print(
            "Reserving __other__ using the least-active training family "
            f"({fallback_source}) so unseen test kernels have a learned "
            "fallback coefficient."
        )
    return families, selected


def combine_training_sessions(session_data):
    rows = []
    activity = []
    target = []
    for session in session_data:
        rows.extend(session["rows"])
        activity.append(session["activity_ns"])
        target.append(session["target_power_w"])
    return rows, np.vstack(activity), np.concatenate(target)


def posterior_predictions(latent_draws, noise_variance_draws, seed):
    """Summarize latent means and posterior predictive intervals."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(
        0.0,
        np.sqrt(noise_variance_draws)[:, None],
        size=latent_draws.shape,
    )
    predictive = np.maximum(0.0, latent_draws + noise)
    lower, upper = np.quantile(predictive, [0.025, 0.975], axis=0)
    return latent_draws.mean(axis=0), lower, upper


def interval_coverage(actual, lower, upper):
    return float(np.mean((actual >= lower) & (actual <= upper)))


def coefficient_rows(terms, draws, column_scale=None):
    rows = []
    for index, term in enumerate(terms):
        values = draws[:, index]
        lower, upper = np.quantile(values, [0.025, 0.975])
        row = {
            "term": term,
            "posterior_mean": float(values.mean()),
            "posterior_median": float(np.median(values)),
            "posterior_lower_95": float(lower),
            "posterior_upper_95": float(upper),
        }
        if column_scale is not None:
            row["activity_rms_scale"] = float(column_scale[index])
        rows.append(row)
    return rows


def test_family_rows(metadata, selected, family_index, family_total_draws):
    rows = []
    all_names = set(metadata["duration_by_name"])
    for family, column in family_index.items():
        values = family_total_draws[:, column]
        lower, upper = np.quantile(values, [0.025, 0.975])
        source_names = all_names - selected if family == OTHER_FAMILY else {family}
        rows.append(
            {
                "kernel_family": family,
                "event_count": sum(
                    metadata["event_count_by_name"][name] for name in source_names
                ),
                "total_duration_ns": sum(
                    metadata["duration_by_name"][name] for name in source_names
                ),
                "energy_posterior_mean_mj": float(values.mean()),
                "energy_posterior_median_mj": float(np.median(values)),
                "energy_posterior_lower_95_mj": float(lower),
                "energy_posterior_upper_95_mj": float(upper),
            }
        )
    rows.sort(key=lambda row: row["energy_posterior_mean_mj"], reverse=True)
    return rows


def add_prediction_columns(
    rows,
    actual_power,
    calibration_count,
    zero_mean,
    zero_lower,
    zero_upper,
    calibrated_mean,
    calibrated_lower,
    calibrated_upper,
    allocation,
):
    attributed_energy = allocation["mean"]
    for index, row in enumerate(rows):
        duration_seconds = row["duration_ns"] / NS_PER_SECOND
        row.update(
            {
                "split": "calibration" if index < calibration_count else "test",
                "measured_dynamic_power_w": float(actual_power[index]),
                "zero_shot_power_mean_w": float(zero_mean[index]),
                "zero_shot_power_lower_95_w": float(zero_lower[index]),
                "zero_shot_power_upper_95_w": float(zero_upper[index]),
                "calibrated_power_mean_w": float(calibrated_mean[index]),
                "calibrated_power_lower_95_w": float(calibrated_lower[index]),
                "calibrated_power_upper_95_w": float(calibrated_upper[index]),
                "measured_dynamic_energy_j": float(
                    actual_power[index] * duration_seconds
                ),
                "zero_shot_energy_mean_j": float(
                    zero_mean[index] * duration_seconds
                ),
                "zero_shot_energy_lower_95_j": float(
                    zero_lower[index] * duration_seconds
                ),
                "zero_shot_energy_upper_95_j": float(
                    zero_upper[index] * duration_seconds
                ),
                "calibrated_energy_mean_j": float(
                    calibrated_mean[index] * duration_seconds
                ),
                "calibrated_energy_lower_95_j": float(
                    calibrated_lower[index] * duration_seconds
                ),
                "calibrated_energy_upper_95_j": float(
                    calibrated_upper[index] * duration_seconds
                ),
                "bayesian_attributed_energy_mean_mj": float(
                    attributed_energy[index].sum()
                ),
                "baseline_energy_mean_mj": float(
                    allocation["baseline_mean"][index]
                ),
            }
        )


def plot_power_transfer(plt, output_dir, rows, calibration_count):
    elapsed = np.asarray(
        [(row["start_ns"] - rows[0]["start_ns"]) / NS_PER_SECOND for row in rows]
    )
    figure, axis = plt.subplots(figsize=(13, 6))
    axis.plot(
        elapsed,
        [row["measured_dynamic_power_w"] for row in rows],
        color="black",
        linewidth=1.2,
        label="Measured V100",
    )
    methods = (
        ("zero_shot", "tab:blue", "Thor zero-shot"),
        ("calibrated", "tab:orange", "Thor + V100 calibration"),
    )
    for prefix, color, label in methods:
        mean = np.asarray([row[f"{prefix}_power_mean_w"] for row in rows])
        lower = np.asarray([row[f"{prefix}_power_lower_95_w"] for row in rows])
        upper = np.asarray([row[f"{prefix}_power_upper_95_w"] for row in rows])
        axis.plot(elapsed, mean, color=color, linewidth=1.1, label=label)
        axis.fill_between(elapsed, lower, upper, color=color, alpha=0.12)
    if calibration_count:
        axis.axvspan(
            elapsed[0], elapsed[calibration_count - 1], color="grey", alpha=0.15,
            label="V100 calibration prefix",
        )
    axis.axvline(
        elapsed[calibration_count], color="grey", linestyle="--", linewidth=1.0
    )
    axis.set_xlabel("Elapsed time in V100 session (s)")
    axis.set_ylabel("Dynamic power (W)")
    axis.set_title("Thor to V100 Bayesian kernel-power transfer")
    axis.grid(alpha=0.2)
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(output_dir / "bayesian_transfer_power_fit.png", dpi=180)
    plt.close(figure)


def plot_energy_scatter(plt, output_dir, rows, calibration_count, metrics):
    test_rows = rows[calibration_count:]
    actual = np.asarray([row["measured_dynamic_energy_j"] for row in test_rows])
    figure, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    methods = (
        ("zero_shot", "Zero-shot Thor to V100", "tab:blue"),
        ("calibrated", "Minimally calibrated Thor to V100", "tab:orange"),
    )
    all_values = [actual]
    for prefix, _, _ in methods:
        all_values.extend(
            np.asarray([row[f"{prefix}_energy_{field}_j"] for row in test_rows])
            for field in ("mean", "lower_95", "upper_95")
        )
    minimum = float(min(values.min() for values in all_values))
    maximum = float(max(values.max() for values in all_values))
    for axis, (prefix, title, color) in zip(axes, methods):
        predicted = np.asarray(
            [row[f"{prefix}_energy_mean_j"] for row in test_rows]
        )
        lower = np.asarray(
            [row[f"{prefix}_energy_lower_95_j"] for row in test_rows]
        )
        upper = np.asarray(
            [row[f"{prefix}_energy_upper_95_j"] for row in test_rows]
        )
        axis.vlines(actual, lower, upper, color=color, alpha=0.18, linewidth=0.8)
        axis.scatter(actual, predicted, color=color, alpha=0.7, s=22)
        axis.plot(
            [minimum, maximum], [minimum, maximum], color="black",
            linestyle="--", linewidth=1.1,
        )
        method_metrics = metrics[prefix]
        r_squared = method_metrics["r_squared"]
        r_label = "undefined" if r_squared is None else f"{r_squared:.4f}"
        axis.text(
            0.04,
            0.96,
            f"Power RMSE: {method_metrics['rmse_w']:.3f} W\nPower R²: {r_label}",
            transform=axis.transAxes,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
        axis.set_title(title)
        axis.set_xlabel("Measured V100 energy (J/window)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Predicted V100 energy (J/window)")
    figure.suptitle("External temporal holdout: energy transfer comparison")
    figure.tight_layout()
    figure.savefig(output_dir / "bayesian_transfer_energy_scatter.png", dpi=180)
    plt.close(figure)


def create_plots(output_dir, rows, calibration_count, family_rows, metrics):
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("--plot requires matplotlib") from error
    plot_power_transfer(plt, output_dir, rows, calibration_count)
    plot_energy_scatter(plt, output_dir, rows, calibration_count, metrics)
    plot_kernel_family_energy(plt, output_dir, family_rows)


def main():
    args = parse_args()
    validate_args(args)
    window_ns = int(args.window_ms * 1_000_000)

    train_scan = scan_sessions(
        discover_session_paths(args.train_sessions_root), args.max_families
    )
    families, selected = training_family_basis(train_scan)
    family_index = {family: index for index, family in enumerate(families)}
    training_sessions = [
        build_session_windows(
            name, telemetry, metadata, selected, family_index,
            window_ns, args.idle_power_mw,
        )
        for name, telemetry, _, metadata in train_scan["sessions"]
    ]
    train_rows, train_activity, train_target = combine_training_sessions(
        training_sessions
    )
    # Drop truncated final windows of each training session from the fit.
    minimum_ns = args.min_window_fraction * window_ns
    complete = window_durations_ns(train_rows) >= minimum_ns
    excluded_training_windows = int((~complete).sum())
    train_rows = [row for row, keep in zip(train_rows, complete) if keep]
    train_activity = train_activity[complete]
    train_target = train_target[complete]

    test_telemetry, test_kernels = require_session_files(
        args.external_test_session
    )
    test_metadata = scan_kernel_families(load_kernels(test_kernels), max_families=2)
    external = build_session_windows(
        args.external_test_session.name,
        test_telemetry,
        test_metadata,
        selected,
        family_index,
        window_ns,
        args.idle_power_mw,
    )
    test_rows = external["rows"]
    test_activity = external["activity_ns"]
    test_target = external["target_power_w"]
    test_complete = window_durations_ns(test_rows) >= minimum_ns
    complete_count = int(test_complete.sum())
    calibration_count = int(complete_count * args.calibration_fraction)
    calibration_count = min(max(calibration_count, 5), complete_count - 2)
    if not test_complete[:calibration_count].all():
        raise ValueError("A truncated window falls inside the calibration prefix")

    all_activity = np.vstack([train_activity, test_activity])
    all_target = np.concatenate([train_target, test_target])
    all_design = activity_design(all_activity, train_rows + test_rows)
    train_count = len(train_rows)
    identifiability = check_identifiability(
        all_design, train_count, ["unattributed_baseline"] + families,
        args.min_windows_per_parameter, args.allow_underdetermined,
    )
    print(
        f"\nSampling Thor posterior with {train_count} training windows...",
        flush=True,
    )
    base_draws, base_noise, base_predictions, column_scale = (
        fit_nonnegative_bayesian_regression(
            all_design,
            all_target,
            train_count,
            args.prior_scale_w,
            args.noise_prior_scale_w,
            args.burn_in,
            args.posterior_draws,
            args.thin,
            args.seed,
        )
    )
    zero_latent = base_predictions[:, train_count:]
    zero_mean, zero_lower, zero_upper = posterior_predictions(
        zero_latent, base_noise, args.seed + 10
    )

    calibration_design = np.column_stack(
        [np.ones(len(test_rows)), zero_mean]
    )
    print(
        "Sampling V100 offset/scale calibration with "
        f"{calibration_count} prefix windows...",
        flush=True,
    )
    calibration_draws, calibration_noise, _, calibration_scale = (
        fit_nonnegative_bayesian_regression(
            calibration_design,
            test_target,
            calibration_count,
            args.prior_scale_w,
            args.noise_prior_scale_w,
            args.burn_in,
            args.posterior_draws,
            args.thin,
            args.seed + 1,
        )
    )
    calibrated_latent = (
        calibration_draws[:, 0, None]
        + calibration_draws[:, 1, None] * zero_latent
    )
    calibrated_mean, calibrated_lower, calibrated_upper = posterior_predictions(
        calibrated_latent, calibration_noise, args.seed + 11
    )

    calibrated_family_draws = base_draws.copy()
    calibrated_family_draws[:, 0] = (
        calibration_draws[:, 0]
        + calibration_draws[:, 1] * base_draws[:, 0]
    )
    calibrated_family_draws[:, 1:] *= calibration_draws[:, 1, None]
    allocation = posterior_energy_allocation(
        calibrated_family_draws,
        test_activity,
        attribution_pool_mj(test_rows, args.attribution),
        window_durations_ns(test_rows),
        args.attribution,
    )
    family_total_draws = allocation["family_total_draws"]
    add_prediction_columns(
        test_rows,
        test_target,
        calibration_count,
        zero_mean,
        zero_lower,
        zero_upper,
        calibrated_mean,
        calibrated_lower,
        calibrated_upper,
        allocation,
    )

    for row, keep in zip(test_rows, test_complete):
        if not keep:
            row["split"] = "excluded"
    evaluation = split_mask(test_rows, "test")
    metrics = {
        "zero_shot": regression_metrics(
            test_target[evaluation], zero_mean[evaluation]
        ),
        "calibrated": regression_metrics(
            test_target[evaluation], calibrated_mean[evaluation]
        ),
    }
    coverage = {
        "zero_shot": interval_coverage(
            test_target[evaluation], zero_lower[evaluation], zero_upper[evaluation]
        ),
        "calibrated": interval_coverage(
            test_target[evaluation],
            calibrated_lower[evaluation],
            calibrated_upper[evaluation],
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "bayesian_transfer_windows.csv",
        test_rows,
        list(test_rows[0]),
    )
    base_coefficient_rows = coefficient_rows(
        ["unattributed_baseline"] + families, base_draws, column_scale
    )
    write_csv(
        args.output_dir / "thor_kernel_coefficients.csv",
        base_coefficient_rows,
        list(base_coefficient_rows[0]),
    )
    calibration_rows = coefficient_rows(
        ["v100_offset_w", "thor_prediction_scale"],
        calibration_draws,
        calibration_scale,
    )
    write_csv(
        args.output_dir / "v100_calibration_coefficients.csv",
        calibration_rows,
        list(calibration_rows[0]),
    )
    family_rows = test_family_rows(
        test_metadata, selected, family_index, family_total_draws
    )
    write_csv(
        args.output_dir / "v100_kernel_family_energy.csv",
        family_rows,
        list(family_rows[0]),
    )

    test_names = set(test_metadata["duration_by_name"])
    selected_test_names = test_names & selected
    selected_events = sum(
        test_metadata["event_count_by_name"][name] for name in selected_test_names
    )
    selected_duration = sum(
        test_metadata["duration_by_name"][name] for name in selected_test_names
    )
    total_duration = sum(test_metadata["duration_by_name"].values())
    accounting = energy_accounting(test_rows, allocation)
    summary = {
        "model": "cross_gpu_bayesian_kernel_family_transfer",
        "attribution_mode": args.attribution,
        "training_sessions_root": str(args.train_sessions_root),
        "training_sessions": [session["name"] for session in training_sessions],
        "excluded_training_sessions": train_scan["excluded_sessions"],
        "external_test_session": str(args.external_test_session),
        "validation": (
            "Thor-only training; first V100 prefix reserved for offset/scale "
            "calibration; both methods compared on the untouched V100 suffix"
        ),
        "window_ms": args.window_ms,
        "calibration_fraction": args.calibration_fraction,
        "training_windows": train_count,
        "v100_calibration_windows": calibration_count,
        "v100_evaluation_windows": int(evaluation.sum()),
        "excluded_short_windows": {
            "training": excluded_training_windows,
            "v100": int((~test_complete).sum()),
        },
        "min_window_fraction": args.min_window_fraction,
        "kernel_families": len(families),
        "training_design_rank": identifiability["design_matrix_rank"],
        "training_design_parameters": int(all_design.shape[1]),
        "identifiability": identifiability,
        "v100_explicit_family_event_coverage": (
            selected_events / test_metadata["valid_events"]
        ),
        "v100_explicit_family_duration_coverage": (
            selected_duration / total_duration
        ),
        "zero_shot_evaluation_power_metrics": metrics["zero_shot"],
        "calibrated_evaluation_power_metrics": metrics["calibrated"],
        "calibrated_rmse_reduction_fraction": (
            1.0
            - metrics["calibrated"]["rmse_w"]
            / metrics["zero_shot"]["rmse_w"]
        ),
        "zero_shot_predictive_interval_coverage": coverage["zero_shot"],
        "calibrated_predictive_interval_coverage": coverage["calibrated"],
        "calibration_posterior": {
            row["term"]: {
                key: value for key, value in row.items() if key != "term"
            }
            for row in calibration_rows
        },
        "v100_measured_dynamic_energy_mj": accounting["measured_dynamic_energy_mj"],
        "v100_assignable_dynamic_energy_mj": (
            accounting["assignable_dynamic_energy_mj"]
        ),
        "v100_attributed_posterior_mean_mj": (
            accounting["attributed_posterior_mean_mj"]
        ),
        "v100_baseline_posterior_mean_mj": accounting["baseline_posterior_mean_mj"],
        "conservation_error_mj": accounting["conservation_error_mj"],
        "important_limitation": (
            "A global offset and scale calibrate device-level power but do not "
            "prove that every kernel-family coefficient transfers unchanged."
        ),
    }
    with (args.output_dir / "bayesian_transfer_summary.json").open(
        "w", encoding="utf-8"
    ) as destination:
        json.dump(summary, destination, indent=2)
        destination.write("\n")

    if args.plot:
        create_plots(
            args.output_dir,
            [row for row in test_rows if row["split"] != "excluded"],
            calibration_count,
            family_rows,
            metrics,
        )

    print("\n=== Thor to V100 Bayesian kernel-energy transfer ===")
    print(f"Thor training windows: {train_count}")
    print(f"V100 calibration windows: {calibration_count}")
    print(f"V100 evaluation windows: {int(evaluation.sum())}")
    print(
        "V100 explicit-family duration coverage: "
        f"{100.0 * summary['v100_explicit_family_duration_coverage']:.2f}%"
    )
    for method, label in (
        ("zero_shot", "Zero-shot"),
        ("calibrated", "Calibrated"),
    ):
        result = metrics[method]
        r_squared = result["r_squared"]
        r_label = "undefined" if r_squared is None else f"{r_squared:.4f}"
        print(
            f"{label}: RMSE={result['rmse_w']:.3f} W, "
            f"R-squared={r_label}, "
            f"95% coverage={100.0 * coverage[method]:.1f}%"
        )
    print(f"Conservation error: {accounting['conservation_error_mj']:.6g} mJ")
    print_identifiability_warnings(identifiability, short_kernel_name)
    print(f"Results: {args.output_dir}")


if __name__ == "__main__":
    main()
