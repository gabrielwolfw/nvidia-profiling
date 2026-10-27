"""Pooled Bayesian kernel-energy attribution across sessions."""

import logging
from dataclasses import dataclass

import numpy as np

from energy_attribution.allocation import (
    EnergyAllocation,
    energy_accounting,
    family_energy_rows,
)
from energy_attribution.config import MultiSessionConfig
from energy_attribution.dataset import WindowDataset, assign_splits
from energy_attribution.prediction import PosteriorPredictions
from energy_attribution.regression import PosteriorSamples, coefficient_summary_rows
from energy_attribution.sessions import (
    build_session_windows,
    discover_session_paths,
    scan_sessions,
)
from energy_attribution.workflows.single_session import (
    allocate_energy,
    fit_posterior,
    holdout_metrics,
)

logger = logging.getLogger(__name__)


@dataclass
class MultiSessionResult:  # pylint: disable=too-many-instance-attributes
    """Pooled model results; ``summary`` is the JSON report."""

    config: MultiSessionConfig
    dataset: WindowDataset
    families: list
    included_sessions: list
    excluded_sessions: list
    posterior: PosteriorSamples
    predictions: PosteriorPredictions
    allocation: EnergyAllocation
    identifiability: dict
    coefficient_rows: list
    family_rows: list
    session_rows: list
    session_metrics: dict
    train_metrics: dict
    test_metrics: dict
    accounting: dict
    summary: dict


def pooled_dataset(sessions):
    """Stack every session's training windows first, then every holdout.

    Each ``SessionWindows.train_count`` marks that session's training prefix;
    the pooled rows ``[:train_count]`` are therefore all training windows.
    """
    rows = []
    activity_parts = []
    target_parts = []
    for split_name in ("train", "test"):
        for session in sessions:
            if split_name == "train":
                selection = slice(0, session.train_count)
            else:
                selection = slice(session.train_count, None)
            rows.extend(session.rows[selection])
            activity_parts.append(session.activity_ns[selection])
            target_parts.append(session.target_power_w[selection])
    return WindowDataset(
        rows,
        np.vstack(activity_parts),
        np.concatenate(target_parts),
        sum(session.train_count for session in sessions),
    )


def analyze(config: MultiSessionConfig) -> MultiSessionResult:
    """Fit one pooled model across sessions without writing any file."""
    scan = scan_sessions(
        discover_session_paths(config.sessions_root), config.max_families
    )
    families = scan.families
    family_index = {family: index for index, family in enumerate(families)}
    sessions = []
    for session in scan.sessions:
        windows = build_session_windows(
            session,
            scan.selected,
            family_index,
            config.window_ns,
            config.idle_power_mw,
        )
        windows.train_count = assign_splits(
            windows.rows,
            config.train_fraction,
            config.window_ns,
            config.min_window_fraction,
            strict=False,
        )
        sessions.append(windows)
    dataset = pooled_dataset(sessions)

    terms = ["unattributed_baseline"] + families
    identifiability, posterior, predictions = fit_posterior(
        config, dataset, terms, label="pooled posterior"
    )
    allocation = allocate_energy(config, dataset, posterior, predictions)
    train_metrics, test_metrics, session_rows, session_metrics = (
        holdout_metrics(dataset, predictions)
    )
    accounting = energy_accounting(dataset.rows, allocation)
    summary = {
        "model": "pooled_nonnegative_bayesian_kernel_family_regression",
        "attribution_mode": config.attribution,
        "validation": "final temporal fraction held out independently per "
        "session",
        "sessions_root": str(config.sessions_root),
        "included_sessions": [session.name for session in sessions],
        "excluded_sessions": scan.excluded_sessions,
        "window_ms": config.window_ms,
        "train_windows": dataset.train_count,
        "test_windows": int(dataset.mask("test").sum()),
        "excluded_short_windows": int(dataset.mask("excluded").sum()),
        "min_window_fraction": config.min_window_fraction,
        "kernel_events": int(sum(scan.event_count_by_name.values())),
        "kernel_families": len(families),
        "design_matrix_rank": identifiability["design_matrix_rank"],
        "design_parameters": len(terms),
        "identifiability": identifiability,
        "posterior_draws": config.posterior_draws,
        "posterior_noise_mean_w": float(np.sqrt(posterior.noise_draws).mean()),
        "train_power_metrics": train_metrics,
        "test_power_metrics": test_metrics,
        "test_session_energy_metrics": session_metrics,
        **accounting,
        "important_limitation": (
            "Attribution is probabilistic; concurrent kernel families may not "
            "be statistically identifiable."
        ),
    }
    return MultiSessionResult(
        config=config,
        dataset=dataset,
        families=families,
        included_sessions=[session.name for session in sessions],
        excluded_sessions=scan.excluded_sessions,
        posterior=posterior,
        predictions=predictions,
        allocation=allocation,
        identifiability=identifiability,
        coefficient_rows=coefficient_summary_rows(
            terms, posterior.coefficient_draws, posterior.column_scale
        ),
        family_rows=family_energy_rows(
            family_index,
            scan.duration_by_name,
            scan.event_count_by_name,
            scan.selected,
            allocation.family_total_draws,
        ),
        session_rows=session_rows,
        session_metrics=session_metrics,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        accounting=accounting,
        summary=summary,
    )
