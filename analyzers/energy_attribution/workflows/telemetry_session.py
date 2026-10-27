"""One-session telemetry-feature power model (computation only)."""

from dataclasses import dataclass

import numpy as np

from energy_attribution.config import TelemetrySessionConfig
from energy_attribution.dataset import split_point
from energy_attribution.telemetry_model import (
    FEATURE_NAMES,
    aggregate_kernels,
    build_dataset,
    coefficient_rows,
    fit_bayesian_regression,
    interval_regression_metrics,
    load_telemetry,
    posterior_intervals,
)
from energy_attribution.units import NS_PER_SECOND


@dataclass
class TelemetrySessionResult:  # pylint: disable=too-many-instance-attributes
    """Window predictions, coefficients and holdout metrics of one session."""

    config: TelemetrySessionConfig
    prediction_rows: list
    coefficient_rows: list
    train_count: int
    targets: np.ndarray
    predicted: np.ndarray
    predictive_lower: np.ndarray
    predictive_upper: np.ndarray
    train_metrics: dict
    test_metrics: dict
    baseline_rmse_w: float
    summary: dict


def load_windows(telemetry, kernels, window_ns, min_telemetry_samples):
    """Aggregate one session into feature rows, features and mean power."""
    origin_ns, window_count, counts, sums = load_telemetry(telemetry, window_ns)
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
        min_telemetry_samples,
    )
    return origin_ns, rows, features, targets


def analyze(config: TelemetrySessionConfig) -> TelemetrySessionResult:
    """Fit the telemetry-feature model to one session; writes nothing."""
    window_ns = config.window_ns
    _, rows, features, targets = load_windows(
        config.telemetry,
        config.kernels,
        window_ns,
        config.min_telemetry_samples,
    )
    train_count = split_point(len(rows), config.train_fraction, strict=False)
    fit = fit_bayesian_regression(
        features,
        targets,
        train_count,
        config.prior_scale,
        config.posterior_draws,
        config.seed,
    )
    (
        predicted,
        credible_lower,
        credible_upper,
        predictive_lower,
        predictive_upper,
    ) = posterior_intervals(fit)

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

    train, test = slice(0, train_count), slice(train_count, None)
    train_metrics = interval_regression_metrics(
        targets[train],
        predicted[train],
        predictive_lower[train],
        predictive_upper[train],
    )
    test_metrics = interval_regression_metrics(
        targets[test],
        predicted[test],
        predictive_lower[test],
        predictive_upper[test],
    )
    baseline = np.full(len(targets) - train_count, targets[train].mean())
    baseline_rmse = float(np.sqrt(np.mean((targets[test] - baseline) ** 2)))
    summary = {
        "model": "conjugate_bayesian_linear_regression",
        "interpretation": (
            "Initial within-run diagnostic model; validation across "
            "independent "
            "runs is required before general use."
        ),
        "window_ms": config.window_ms,
        "complete_windows": len(rows),
        "train_windows": train_count,
        "test_windows": len(rows) - train_count,
        "train_fraction": config.train_fraction,
        "posterior_draws": config.posterior_draws,
        "prior_scale": config.prior_scale,
        "features": FEATURE_NAMES,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "test_baseline_mean_power_rmse_w": baseline_rmse,
    }
    return TelemetrySessionResult(
        config=config,
        prediction_rows=prediction_rows,
        coefficient_rows=coefficient_rows(
            fit,
            "W at training feature means",
            "W per one training standard deviation",
        ),
        train_count=train_count,
        targets=targets,
        predicted=predicted,
        predictive_lower=predictive_lower,
        predictive_upper=predictive_upper,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        baseline_rmse_w=baseline_rmse,
        summary=summary,
    )
