#!/usr/bin/env python3
"""Backward-compatible facade for the packaged telemetry-energy model.

Running this file fits the one-session telemetry model
(``energy_attribution.commands.telemetry_energy``). The re-exports keep
existing imports working; new code should import :mod:`energy_attribution`.
"""

from energy_attribution.commands.telemetry_energy import main
from energy_attribution.io import write_csv
from energy_attribution.telemetry_model import (
    FEATURE_NAMES,
    aggregate_kernels,
    build_dataset,
    fit_bayesian_regression,
    interval_regression_metrics as regression_metrics,
    load_telemetry,
)
from energy_attribution.units import NS_PER_SECOND

__all__ = [
    "FEATURE_NAMES",
    "NS_PER_SECOND",
    "aggregate_kernels",
    "build_dataset",
    "fit_bayesian_regression",
    "load_telemetry",
    "main",
    "regression_metrics",
    "write_csv",
]


if __name__ == "__main__":
    main()
