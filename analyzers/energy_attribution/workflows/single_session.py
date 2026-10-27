"""Single-session Bayesian kernel-energy attribution (computation only).

:func:`analyze` reads the two input CSVs and returns a
:class:`SingleSessionResult`; it writes nothing. Output files are produced by
:func:`energy_attribution.reporting.outputs.write_single_session`.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from energy_attribution.allocation import (
    EnergyAllocation,
    EventAttribution,
    apply_allocation_to_rows,
    attribute_events,
    attribution_pool_mj,
    energy_accounting,
    family_energy_rows,
    posterior_energy_allocation,
)
from energy_attribution.config import SingleSessionConfig
from energy_attribution.dataset import WindowDataset, assign_splits
from energy_attribution.diagnostics import check_identifiability
from energy_attribution.families import family_columns, scan_kernel_families
from energy_attribution.io import load_kernels, load_power_trace
from energy_attribution.prediction import (
    PosteriorPredictions,
    add_prediction_columns,
    prediction_metrics,
    session_holdout_energy,
    summarize_predictions,
)
from energy_attribution.regression import (
    PosteriorSamples,
    coefficient_summary_rows,
    fit_nonnegative_bayesian_regression,
)
from energy_attribution.windows import aggregate_family_activity, build_windows

logger = logging.getLogger(__name__)

PREDICTIVE_SEED_OFFSET = 10


@dataclass
class SingleSessionResult:  # pylint: disable=too-many-instance-attributes
    """Everything computed for one session; ``summary`` is the JSON report."""

    config: SingleSessionConfig
    dataset: WindowDataset
    families: list
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
    events: Optional[EventAttribution]
    summary: dict


def fit_posterior(config, dataset, terms, label="posterior"):
    """Check identifiability, then sample the non-negative regression.

    Shared by the single- and multi-session workflows. Returns
    ``(identifiability, posterior, predictions)``.
    """
    identifiability = check_identifiability(
        dataset.design[dataset.mask("train", "test")],
        dataset.train_count,
        terms,
        config.min_windows_per_parameter,
        config.allow_underdetermined,
    )
    logger.info(
        "Sampling %s with %d training windows...", label, dataset.train_count
    )
    posterior = fit_nonnegative_bayesian_regression(
        dataset.design,
        dataset.target_power_w,
        dataset.train_count,
        config.prior_scale_w,
        config.noise_prior_scale_w,
        config.burn_in,
        config.posterior_draws,
        config.thin,
        config.seed,
    )
    predictions = summarize_predictions(
        posterior.prediction_draws,
        posterior.noise_draws,
        config.seed + PREDICTIVE_SEED_OFFSET,
    )
    return identifiability, posterior, predictions


def allocate_energy(config, dataset, posterior, predictions):
    """Split window energy and annotate the window rows with the results."""
    allocation = posterior_energy_allocation(
        posterior.coefficient_draws,
        dataset.activity_ns,
        attribution_pool_mj(dataset.rows, config.attribution),
        dataset.durations_ns,
        config.attribution,
    )
    add_prediction_columns(dataset.rows, predictions)
    apply_allocation_to_rows(dataset.rows, allocation)
    return allocation


def holdout_metrics(dataset, predictions):
    """Return (train, test, session rows, session metrics) of a fit."""
    target = dataset.target_power_w
    return (
        prediction_metrics(target, predictions, dataset.mask("train")),
        prediction_metrics(target, predictions, dataset.mask("test")),
        *session_holdout_energy(dataset.rows, predictions),
    )


def attribution_description(attribution):
    if attribution == "model":
        return (
            "Each window's measured dynamic energy is split in proportion to "
            "the posterior terms b0 + sum(beta_f * activity_f); the b0 share "
            "is reported as baseline energy. Family energy is distributed to "
            "events by temporal overlap."
        )
    return (
        "Legacy: dynamic energy scaled by the kernel-busy fraction is "
        "split among families by beta_f * activity_f, ignoring b0. Family "
        "energy is distributed to events by temporal overlap."
    )


def build_summary(config, dataset, result_parts):
    """Assemble the machine-readable JSON summary of one session."""
    parts = result_parts
    return {
        "model": "nonnegative_bayesian_kernel_family_regression",
        "inference": "Gibbs sampling with half-normal coefficient priors",
        "attribution_mode": config.attribution,
        "attribution": attribution_description(config.attribution),
        "important_limitation": (
            "No direct per-kernel energy ground truth is available. Credible "
            "intervals quantify model uncertainty, not physical meter accuracy."
        ),
        "event_interval_note": (
            "Per-event bounds sum marginal 95% family-window intervals and are "
            "an approximation when an event crosses more than one window."
        ),
        "window_ms": config.window_ms,
        "idle_power_mw": config.idle_power_mw,
        "windows": len(dataset.rows),
        "train_windows": dataset.train_count,
        "test_windows": int(dataset.mask("test").sum()),
        "excluded_short_windows": int(dataset.mask("excluded").sum()),
        "min_window_fraction": config.min_window_fraction,
        "kernel_events": parts["event_count"],
        "kernel_families": len(parts["families"]),
        "design_matrix_rank": parts["identifiability"]["design_matrix_rank"],
        "design_parameters": len(parts["families"]) + 1,
        "identifiability": parts["identifiability"],
        "posterior_draws": config.posterior_draws,
        "posterior_noise_mean_w": float(
            np.sqrt(parts["posterior"].noise_draws).mean()
        ),
        "train_power_metrics": parts["train_metrics"],
        "test_power_metrics": parts["test_metrics"],
        "test_session_energy_metrics": parts["session_metrics"],
        **parts["accounting"],
    }


def analyze(config: SingleSessionConfig) -> SingleSessionResult:
    """Fit one session and attribute its energy without writing any file."""
    window_ns = config.window_ns
    trace = load_power_trace(config.telemetry)
    logger.info("Loading kernel events...")
    table = load_kernels(
        config.kernels, load_event_ids=not config.skip_event_output
    )
    scan = scan_kernel_families(table, config.max_families)
    analysis_start = max(trace.first_ns, scan.first_start)
    analysis_end = min(trace.last_ns, scan.last_end)
    if analysis_end <= analysis_start:
        raise ValueError("Telemetry and kernel timelines do not overlap")

    families = scan.families
    family_index = {name: index for index, name in enumerate(families)}
    logger.info(
        "Using %d kernel families for %d events.",
        len(families),
        scan.valid_events,
    )
    columns = family_columns(table, scan.selected, family_index)
    activity_ns, union_active_ns = aggregate_family_activity(
        table, columns, len(families), analysis_start, analysis_end, window_ns
    )
    rows, target_power_w = build_windows(
        trace,
        union_active_ns,
        analysis_start,
        analysis_end,
        window_ns,
        config.idle_power_mw,
    )
    train_count = assign_splits(
        rows, config.train_fraction, window_ns, config.min_window_fraction
    )
    dataset = WindowDataset(rows, activity_ns, target_power_w, train_count)

    terms = ["unattributed_baseline"] + families
    identifiability, posterior, predictions = fit_posterior(
        config, dataset, terms
    )
    allocation = allocate_energy(config, dataset, posterior, predictions)
    train_metrics, test_metrics, session_rows, session_metrics = (
        holdout_metrics(dataset, predictions)
    )
    accounting = energy_accounting(rows, allocation)
    events = None
    if not config.skip_event_output:
        events = attribute_events(
            table,
            columns,
            families,
            activity_ns,
            allocation,
            analysis_start,
            analysis_end,
            window_ns,
        )
        accounting["event_attributed_posterior_mean_mj"] = float(
            events.mean.sum()
        )

    parts = {
        "families": families,
        "event_count": scan.valid_events,
        "identifiability": identifiability,
        "posterior": posterior,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "session_metrics": session_metrics,
        "accounting": accounting,
    }
    return SingleSessionResult(
        config=config,
        dataset=dataset,
        families=families,
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
        events=events,
        summary=build_summary(config, dataset, parts),
    )
