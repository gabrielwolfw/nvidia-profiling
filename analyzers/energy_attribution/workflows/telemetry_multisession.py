"""Pooled telemetry-feature energy model across sessions (computation only)."""

import logging
from dataclasses import dataclass

import numpy as np

from energy_attribution.config import TelemetryMultiSessionConfig
from energy_attribution.dataset import split_point
from energy_attribution.sessions import discover_session_paths
from energy_attribution.telemetry_model import (
    FEATURE_NAMES,
    coefficient_rows,
    fit_bayesian_regression,
    interval_regression_metrics,
    posterior_intervals,
)
from energy_attribution.units import NS_PER_SECOND
from energy_attribution.workflows.telemetry_session import load_windows

logger = logging.getLogger(__name__)


@dataclass
class TelemetryMultiSessionResult:  # pylint: disable=too-many-instance-attributes
    """Pooled predictions (train rows first), coefficients and session totals."""

    config: TelemetryMultiSessionConfig
    rows: list
    prediction_rows: list
    coefficient_rows: list
    session_energy_rows: list
    train_count: int
    power_targets: np.ndarray
    energy_targets: np.ndarray
    predicted_energy: np.ndarray
    energy_lower: np.ndarray
    energy_upper: np.ndarray
    energy_metrics: dict
    power_metrics: dict
    summary: dict

    @property
    def window_seconds(self):
        return self.config.window_ns / NS_PER_SECOND


def load_session(session, config):
    """Aggregate one session and label its rows; returns (rows, X, y, split)."""
    logger.info("\n=== Aggregating %s ===", session.name)
    origin_ns, rows, features, targets = load_windows(
        session.telemetry, session.kernels, config.window_ns,
        config.min_telemetry_samples,
    )
    split = split_point(len(rows), config.train_fraction, strict=False)
    for position, row in enumerate(rows):
        row["session"] = session.name
        row["session_window_index"] = position
        row["session_elapsed_s"] = (row["window_start_ns"] - origin_ns) / NS_PER_SECOND
        row["split"] = "train" if position < split else "test"
    return rows, features, targets, split


def session_energy(rows, sessions, predicted_energy):
    """Measured and predicted total energy of each session."""
    energy_rows = []
    for session in sessions:
        indices = [index for index, row in enumerate(rows) if row["session"] == session.name]
        measured = float(sum(rows[index]["measured_energy_j"] for index in indices))
        modeled = float(sum(predicted_energy[index] for index in indices))
        energy_rows.append(
            {
                "session": session.name,
                "windows": len(indices),
                "measured_energy_j": measured,
                "predicted_energy_j": modeled,
                "energy_error_j": modeled - measured,
                "energy_error_pct": (
                    100.0 * (modeled - measured) / measured if measured else ""
                ),
            }
        )
    return energy_rows


def analyze(  # pylint: disable=too-many-locals
    config: TelemetryMultiSessionConfig,
) -> TelemetryMultiSessionResult:
    """Fit one energy-per-window model across sessions; writes nothing."""
    sessions = discover_session_paths(config.sessions_root)
    if len(sessions) < 2:
        raise ValueError(
            "--sessions-root must contain at least two session directories "
            "with gpu_telemetry.csv and kernel_activity.csv"
        )
    train_parts, test_parts, session_counts = [], [], {}
    for session in sessions:
        rows, features, targets, split = load_session(session, config)
        train_parts.append((rows[:split], features[:split], targets[:split]))
        test_parts.append((rows[split:], features[split:], targets[split:]))
        session_counts[session.name] = {
            "windows": len(rows), "train_windows": split,
            "test_windows": len(rows) - split,
        }

    ordered_parts = train_parts + test_parts
    rows = [row for part, _, _ in ordered_parts for row in part]
    features = np.concatenate([part for _, part, _ in ordered_parts])
    power_targets = np.concatenate([part for _, _, part in ordered_parts])
    window_seconds = config.window_ns / NS_PER_SECOND
    energy_targets = power_targets * window_seconds
    train_count = sum(len(part) for part, _, _ in train_parts)
    fit = fit_bayesian_regression(
        features, energy_targets, train_count, config.prior_scale,
        config.posterior_draws, config.seed,
    )
    (
        predicted_energy, credible_lower, credible_upper, energy_lower, energy_upper,
    ) = posterior_intervals(fit)
    predicted_power = predicted_energy / window_seconds
    power_lower = energy_lower / window_seconds
    power_upper = energy_upper / window_seconds

    prediction_rows = []
    for index, row in enumerate(rows):
        output = dict(row)
        output.update(
            {
                "predicted_energy_j": predicted_energy[index],
                "credible_energy_lower_j": credible_lower[index],
                "credible_energy_upper_j": credible_upper[index],
                "predictive_energy_lower_j": energy_lower[index],
                "predictive_energy_upper_j": energy_upper[index],
                "energy_residual_j": energy_targets[index] - predicted_energy[index],
                "predicted_power_w": predicted_power[index],
                "credible_power_lower_w": credible_lower[index] / window_seconds,
                "credible_power_upper_w": credible_upper[index] / window_seconds,
                "predictive_power_lower_w": power_lower[index],
                "predictive_power_upper_w": power_upper[index],
                "power_residual_w": power_targets[index] - predicted_power[index],
            }
        )
        prediction_rows.append(output)

    test = slice(train_count, None)
    power_metrics = interval_regression_metrics(
        power_targets[test], predicted_power[test], power_lower[test], power_upper[test]
    )
    raw_energy = interval_regression_metrics(
        energy_targets[test], predicted_energy[test], energy_lower[test], energy_upper[test]
    )
    energy_metrics = {
        "count": raw_energy["count"],
        "mae_j_per_window": raw_energy["mae_w"],
        "rmse_j_per_window": raw_energy["rmse_w"],
        "r_squared": raw_energy["r_squared"],
        "predictive_interval_95_coverage": raw_energy["predictive_interval_95_coverage"],
    }
    summary = {
        "model": "multisession_conjugate_bayesian_linear_energy_regression",
        "target": "measured_energy_j_per_fixed_window",
        "validation": "final temporal fraction held out independently per session",
        "sessions_root": str(config.sessions_root),
        "session_count": len(sessions),
        "window_ms": config.window_ms,
        "train_fraction_per_session": config.train_fraction,
        "train_windows": train_count,
        "test_windows": len(rows) - train_count,
        "posterior_draws": config.posterior_draws,
        "features": FEATURE_NAMES,
        "session_windows": session_counts,
        "temporal_holdout_energy_metrics": energy_metrics,
        "derived_temporal_holdout_power_metrics": power_metrics,
    }
    return TelemetryMultiSessionResult(
        config=config,
        rows=rows,
        prediction_rows=prediction_rows,
        coefficient_rows=coefficient_rows(
            fit, "J per window at training feature means",
            "J per window per one training standard deviation",
        ),
        session_energy_rows=session_energy(rows, sessions, predicted_energy),
        train_count=train_count,
        power_targets=power_targets,
        energy_targets=energy_targets,
        predicted_energy=predicted_energy,
        energy_lower=energy_lower,
        energy_upper=energy_upper,
        energy_metrics=energy_metrics,
        power_metrics=power_metrics,
        summary=summary,
    )
