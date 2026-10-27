"""Cross-GPU transfer of the Bayesian kernel-energy model (computation only).

A pooled model is trained on the source GPU's sessions, then evaluated on one
external target-GPU session twice: zero-shot, and after a non-negative
offset/scale calibration fitted on the target session's first windows.
"""

import logging
from dataclasses import dataclass

import numpy as np

from energy_attribution.allocation import (
    EnergyAllocation,
    attribution_pool_mj,
    energy_accounting,
    family_energy_rows,
    posterior_energy_allocation,
)
from energy_attribution.config import CrossGpuConfig
from energy_attribution.dataset import activity_design, split_mask, window_durations_ns
from energy_attribution.diagnostics import check_identifiability
from energy_attribution.families import OTHER_FAMILY, scan_kernel_families
from energy_attribution.io import load_kernels
from energy_attribution.prediction import (
    PosteriorPredictions,
    interval_coverage,
    regression_metrics,
    summarize_predictions,
)
from energy_attribution.regression import (
    PosteriorSamples,
    coefficient_summary_rows,
    fit_nonnegative_bayesian_regression,
)
from energy_attribution.sessions import (
    SessionInput,
    build_session_windows,
    discover_session_paths,
    scan_sessions,
    session_files,
)
from energy_attribution.units import NS_PER_SECOND

logger = logging.getLogger(__name__)

ZERO_SHOT_SEED_OFFSET = 10
CALIBRATION_SEED_OFFSET = 1
CALIBRATED_SEED_OFFSET = 11
METHODS = ("zero_shot", "calibrated")


@dataclass
class CrossGpuResult:  # pylint: disable=too-many-instance-attributes
    """Everything computed for one transfer evaluation."""

    config: CrossGpuConfig
    families: list
    training_sessions: list
    train_count: int
    calibration_count: int
    test_rows: list
    base: PosteriorSamples
    calibration: PosteriorSamples
    predictions: dict[str, PosteriorPredictions]
    allocation: EnergyAllocation
    identifiability: dict
    base_coefficient_rows: list
    calibration_rows: list
    family_rows: list
    metrics: dict
    coverage: dict
    accounting: dict
    summary: dict


def training_family_basis(scan):
    """Return a train-only family basis with a learned fallback column."""
    families = list(scan.families)
    selected = set(scan.selected)
    if OTHER_FAMILY not in families:
        fallback_source = min(
            families, key=lambda name: scan.duration_by_name[name]
        )
        families.remove(fallback_source)
        families.append(OTHER_FAMILY)
        selected.remove(fallback_source)
        logger.info(
            "Reserving __other__ using the least-active training family "
            "(%s) so unseen test kernels have a learned fallback coefficient.",
            fallback_source,
        )
    return families, selected


def complete_windows(rows, config):
    return (
        window_durations_ns(rows)
        >= config.min_window_fraction * config.window_ns
    )


def training_windows(config, scan, selected, family_index):
    """Pool every source session, dropping each truncated final window.

    Returns ``(rows, activity_ns, target_power_w, session_names, excluded)``.
    """
    sessions = [
        build_session_windows(
            session,
            selected,
            family_index,
            config.window_ns,
            config.idle_power_mw,
        )
        for session in scan.sessions
    ]
    rows = [row for session in sessions for row in session.rows]
    activity = np.vstack([session.activity_ns for session in sessions])
    target = np.concatenate([session.target_power_w for session in sessions])
    complete = complete_windows(rows, config)
    return (
        [row for row, keep in zip(rows, complete) if keep],
        activity[complete],
        target[complete],
        [session.name for session in sessions],
        int((~complete).sum()),
    )


def external_session(config, selected, family_index):
    """Aggregate the target session with the source family basis."""
    telemetry, kernels = session_files(config.external_test_session)
    scan = scan_kernel_families(load_kernels(kernels), max_families=2)
    session = SessionInput(
        config.external_test_session.name, telemetry, kernels, scan
    )
    windows = build_session_windows(
        session, selected, family_index, config.window_ns, config.idle_power_mw
    )
    return scan, windows


def calibration_prefix(complete, calibration_fraction):
    complete_count = int(complete.sum())
    count = int(complete_count * calibration_fraction)
    count = min(max(count, 5), complete_count - 2)
    if not complete[:count].all():
        raise ValueError(
            "A truncated window falls inside the calibration prefix"
        )
    return count


def calibrate(config, zero_latent, zero_mean, test_target, calibration_count):
    """Fit power_target = offset + scale * zero_shot on the calibration set."""
    logger.info(
        "Sampling %s offset/scale calibration with %d prefix windows...",
        config.target_label,
        calibration_count,
    )
    calibration = fit_nonnegative_bayesian_regression(
        np.column_stack([np.ones(len(zero_mean)), zero_mean]),
        test_target,
        calibration_count,
        config.prior_scale_w,
        config.noise_prior_scale_w,
        config.burn_in,
        config.posterior_draws,
        config.thin,
        config.seed + CALIBRATION_SEED_OFFSET,
    )
    draws = calibration.coefficient_draws
    calibrated = summarize_predictions(
        draws[:, 0, None] + draws[:, 1, None] * zero_latent,
        calibration.noise_draws,
        config.seed + CALIBRATED_SEED_OFFSET,
    )
    return calibration, calibrated


def calibrated_coefficients(base_draws, calibration_draws):
    """Map source-model draws through each draw's offset and scale."""
    calibrated = base_draws.copy()
    calibrated[:, 0] = (
        calibration_draws[:, 0] + calibration_draws[:, 1] * base_draws[:, 0]
    )
    calibrated[:, 1:] *= calibration_draws[:, 1, None]
    return calibrated


def add_transfer_columns(
    rows, actual_power, calibration_count, predictions, allocation
):
    """Add measured, zero-shot and calibrated values to target-session rows."""
    for index, row in enumerate(rows):
        seconds = row["duration_ns"] / NS_PER_SECOND
        row["split"] = "calibration" if index < calibration_count else "test"
        row["measured_dynamic_power_w"] = float(actual_power[index])
        for method in METHODS:
            prediction = predictions[method]
            row[f"{method}_power_mean_w"] = float(prediction.mean[index])
            row[f"{method}_power_lower_95_w"] = float(
                prediction.predictive_lower[index]
            )
            row[f"{method}_power_upper_95_w"] = float(
                prediction.predictive_upper[index]
            )
        row["measured_dynamic_energy_j"] = float(actual_power[index] * seconds)
        for method in METHODS:
            prediction = predictions[method]
            row[f"{method}_energy_mean_j"] = float(
                prediction.mean[index] * seconds
            )
            row[f"{method}_energy_lower_95_j"] = float(
                prediction.predictive_lower[index] * seconds
            )
            row[f"{method}_energy_upper_95_j"] = float(
                prediction.predictive_upper[index] * seconds
            )
        row["bayesian_attributed_energy_mean_mj"] = float(
            allocation.mean[index].sum()
        )
        row["baseline_energy_mean_mj"] = float(allocation.baseline_mean[index])


def evaluate(test_rows, test_target, predictions):
    """Point metrics and predictive coverage on the untouched target suffix."""
    evaluation = split_mask(test_rows, "test")
    actual = test_target[evaluation]
    metrics = {
        method: regression_metrics(actual, predictions[method].mean[evaluation])
        for method in METHODS
    }
    coverage = {
        method: interval_coverage(
            actual,
            predictions[method].predictive_lower[evaluation],
            predictions[method].predictive_upper[evaluation],
        )
        for method in METHODS
    }
    return evaluation, metrics, coverage


def explicit_family_coverage(scan, selected):
    """Share of target events and busy time covered by explicit families."""
    names = set(scan.duration_by_name) & selected
    events = sum(scan.event_count_by_name[name] for name in names)
    duration = sum(scan.duration_by_name[name] for name in names)
    return (
        events / scan.valid_events,
        duration / sum(scan.duration_by_name.values()),
    )


def analyze(config: CrossGpuConfig) -> CrossGpuResult:  # pylint: disable=too-many-locals
    """Train on the source GPU, calibrate and evaluate on the target session."""
    source, target = config.source_label, config.target_label
    train_scan = scan_sessions(
        discover_session_paths(config.train_sessions_root), config.max_families
    )
    families, selected = training_family_basis(train_scan)
    family_index = {family: index for index, family in enumerate(families)}
    (
        train_rows,
        train_activity,
        train_target,
        session_names,
        excluded_training,
    ) = training_windows(config, train_scan, selected, family_index)
    test_scan, external = external_session(config, selected, family_index)
    test_rows, test_target = external.rows, external.target_power_w
    test_complete = complete_windows(test_rows, config)
    calibration_count = calibration_prefix(
        test_complete, config.calibration_fraction
    )

    all_design = activity_design(
        np.vstack([train_activity, external.activity_ns]),
        train_rows + test_rows,
    )
    train_count = len(train_rows)
    identifiability = check_identifiability(
        all_design,
        train_count,
        ["unattributed_baseline"] + families,
        config.min_windows_per_parameter,
        config.allow_underdetermined,
    )
    logger.info(
        "\nSampling %s posterior with %d training windows...",
        source,
        train_count,
    )
    base = fit_nonnegative_bayesian_regression(
        all_design,
        np.concatenate([train_target, test_target]),
        train_count,
        config.prior_scale_w,
        config.noise_prior_scale_w,
        config.burn_in,
        config.posterior_draws,
        config.thin,
        config.seed,
    )
    zero_latent = base.prediction_draws[:, train_count:]
    zero_shot = summarize_predictions(
        zero_latent, base.noise_draws, config.seed + ZERO_SHOT_SEED_OFFSET
    )
    calibration, calibrated = calibrate(
        config, zero_latent, zero_shot.mean, test_target, calibration_count
    )
    predictions = {"zero_shot": zero_shot, "calibrated": calibrated}

    allocation = posterior_energy_allocation(
        calibrated_coefficients(
            base.coefficient_draws, calibration.coefficient_draws
        ),
        external.activity_ns,
        attribution_pool_mj(test_rows, config.attribution),
        window_durations_ns(test_rows),
        config.attribution,
    )
    add_transfer_columns(
        test_rows, test_target, calibration_count, predictions, allocation
    )
    for row, keep in zip(test_rows, test_complete):
        if not keep:
            row["split"] = "excluded"
    evaluation, metrics, coverage = evaluate(
        test_rows, test_target, predictions
    )

    calibration_rows = coefficient_summary_rows(
        [f"{target.lower()}_offset_w", f"{source.lower()}_prediction_scale"],
        calibration.coefficient_draws,
        calibration.column_scale,
        prefix="posterior_",
        suffix="",
        include_median=True,
    )
    event_coverage, duration_coverage = explicit_family_coverage(
        test_scan, selected
    )
    accounting = energy_accounting(test_rows, allocation)
    summary = {
        "model": "cross_gpu_bayesian_kernel_family_transfer",
        "attribution_mode": config.attribution,
        "training_sessions_root": str(config.train_sessions_root),
        "training_sessions": session_names,
        "excluded_training_sessions": train_scan.excluded_sessions,
        "external_test_session": str(config.external_test_session),
        "validation": (
            f"{source}-only training; first {target} prefix reserved for "
            "offset/scale calibration; both methods compared on the untouched "
            f"{target} suffix"
        ),
        "window_ms": config.window_ms,
        "calibration_fraction": config.calibration_fraction,
        "training_windows": train_count,
        "v100_calibration_windows": calibration_count,
        "v100_evaluation_windows": int(evaluation.sum()),
        "excluded_short_windows": {
            "training": excluded_training,
            "v100": int((~test_complete).sum()),
        },
        "min_window_fraction": config.min_window_fraction,
        "kernel_families": len(families),
        "training_design_rank": identifiability["design_matrix_rank"],
        "training_design_parameters": int(all_design.shape[1]),
        "identifiability": identifiability,
        "v100_explicit_family_event_coverage": event_coverage,
        "v100_explicit_family_duration_coverage": duration_coverage,
        "zero_shot_evaluation_power_metrics": metrics["zero_shot"],
        "calibrated_evaluation_power_metrics": metrics["calibrated"],
        "calibrated_rmse_reduction_fraction": (
            1.0
            - metrics["calibrated"]["rmse_w"] / metrics["zero_shot"]["rmse_w"]
        ),
        "zero_shot_predictive_interval_coverage": coverage["zero_shot"],
        "calibrated_predictive_interval_coverage": coverage["calibrated"],
        "calibration_posterior": {
            row["term"]: {
                key: value for key, value in row.items() if key != "term"
            }
            for row in calibration_rows
        },
        "v100_measured_dynamic_energy_mj": accounting[
            "measured_dynamic_energy_mj"
        ],
        "v100_assignable_dynamic_energy_mj": accounting[
            "assignable_dynamic_energy_mj"
        ],
        "v100_attributed_posterior_mean_mj": accounting[
            "attributed_posterior_mean_mj"
        ],
        "v100_baseline_posterior_mean_mj": accounting[
            "baseline_posterior_mean_mj"
        ],
        "conservation_error_mj": accounting["conservation_error_mj"],
        "important_limitation": (
            "A global offset and scale calibrate device-level power but do not "
            "prove that every kernel-family coefficient transfers unchanged."
        ),
    }
    return CrossGpuResult(
        config=config,
        families=families,
        training_sessions=session_names,
        train_count=train_count,
        calibration_count=calibration_count,
        test_rows=test_rows,
        base=base,
        calibration=calibration,
        predictions=predictions,
        allocation=allocation,
        identifiability=identifiability,
        base_coefficient_rows=coefficient_summary_rows(
            ["unattributed_baseline"] + families,
            base.coefficient_draws,
            base.column_scale,
            prefix="posterior_",
            suffix="",
            include_median=True,
        ),
        calibration_rows=calibration_rows,
        family_rows=family_energy_rows(
            family_index,
            test_scan.duration_by_name,
            test_scan.event_count_by_name,
            selected,
            allocation.family_total_draws,
        ),
        metrics=metrics,
        coverage=coverage,
        accounting=accounting,
        summary=summary,
    )
