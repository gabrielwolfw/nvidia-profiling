"""Split measured window energy among kernel families, the baseline and events."""

from dataclasses import dataclass

from typing import Optional

import numpy as np

from energy_attribution.config import ATTRIBUTION_MODES
from energy_attribution.families import source_names
from energy_attribution.regression import posterior_summary
from energy_attribution.windows import distribute_to_events


@dataclass
class EnergyAllocation:
    """Posterior energy split (mJ).

    ``mean``/``lower``/``upper`` are (windows, families); the ``*_draws``
    arrays hold per-draw totals over all windows for credible intervals.
    """

    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    family_total_draws: np.ndarray
    baseline_mean: np.ndarray
    baseline_total_draws: np.ndarray


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

    return EnergyAllocation(
        mean=means,
        lower=lowers,
        upper=uppers,
        family_total_draws=family_total_draws,
        baseline_mean=baseline_mean,
        baseline_total_draws=baseline_total_draws,
    )


def apply_allocation_to_rows(window_rows, allocation):
    for row, family_energy, baseline in zip(
        window_rows, allocation.mean, allocation.baseline_mean
    ):
        row["bayesian_attributed_energy_mean_mj"] = float(family_energy.sum())
        row["baseline_energy_mean_mj"] = float(baseline)


def energy_accounting(window_rows, allocation):
    """Energy totals; dynamic = attributed + baseline + unassigned."""
    total_dynamic = sum(row["dynamic_energy_mj"] for row in window_rows)
    total_assignable = sum(row["assignable_energy_mj"] for row in window_rows)
    attributed = float(allocation.mean.sum())
    baseline_draws = allocation.baseline_total_draws
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


@dataclass
class EventAttribution:  # pylint: disable=too-many-instance-attributes
    """Posterior energy (mJ) of every kernel event, in trace order.

    ``family_column`` indexes ``families``; ``event_ids`` is ``None`` when the
    trace has no event_id column.
    """

    start_ns: np.ndarray
    end_ns: np.ndarray
    event_ids: Optional[np.ndarray]
    family_column: np.ndarray
    families: list
    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


def attribute_events(
    table, columns, families, activity_ns, allocation,
    analysis_start, analysis_end, window_ns,
):
    """Distribute each family's window energy to its events by overlap."""
    mean, lower, upper = distribute_to_events(
        table, columns, activity_ns,
        [allocation.mean, allocation.lower, allocation.upper],
        analysis_start, analysis_end, window_ns,
    )
    return EventAttribution(
        start_ns=table.start_ns,
        end_ns=table.end_ns,
        event_ids=table.event_ids,
        family_column=columns,
        families=list(families),
        mean=mean,
        lower=lower,
        upper=upper,
    )


def family_energy_rows(
    family_index, duration_by_name, event_count_by_name, selected,
    family_total_draws,
):
    """Posterior energy totals per family, largest mean first.

    Event counts and durations of ``__other__`` sum every kernel name outside
    ``selected``; names absent from the statistics count as zero.
    """
    rows = []
    for family, column in family_index.items():
        names = source_names(family, duration_by_name, selected)
        mean, median, lower, upper = posterior_summary(family_total_draws[:, column])
        rows.append(
            {
                "kernel_family": family,
                "event_count": sum(event_count_by_name[name] for name in names),
                "total_duration_ns": sum(duration_by_name[name] for name in names),
                "energy_posterior_mean_mj": mean,
                "energy_posterior_median_mj": median,
                "energy_posterior_lower_95_mj": lower,
                "energy_posterior_upper_95_mj": upper,
            }
        )
    rows.sort(key=lambda row: row["energy_posterior_mean_mj"], reverse=True)
    return rows

