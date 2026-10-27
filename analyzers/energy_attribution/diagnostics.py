"""Identifiability checks for kernel-family regression designs."""

import logging

import numpy as np

from energy_attribution.config import DEFAULT_MIN_WINDOWS_PER_PARAMETER

logger = logging.getLogger(__name__)

ALIAS_RELATIVE_RESIDUAL = 1e-6


def aliased_columns(train_design, tolerance=ALIAS_RELATIVE_RESIDUAL):
    """Columns that are (numerically) a linear combination of the others.

    Their coefficients are not identified by the data: the likelihood is flat
    along a direction that trades them off against other columns, so their
    posterior is shaped by the prior and the non-negativity constraint.
    """
    column_scale = np.sqrt(np.mean(train_design * train_design, axis=0))
    active = column_scale > 1e-12
    scaled = train_design[:, active] / column_scale[active]
    active_columns = np.flatnonzero(active)
    aliased = []
    for position, column in enumerate(active_columns):
        target = scaled[:, position]
        others = np.delete(scaled, position, axis=1)
        if not others.shape[1]:
            continue
        solution = np.linalg.lstsq(others, target, rcond=None)[0]
        residual = target - others @ solution
        if float(residual @ residual) <= tolerance * float(target @ target):
            aliased.append(int(column))
    return aliased


def check_identifiability(
    design,
    train_count,
    terms,
    min_windows_per_parameter=DEFAULT_MIN_WINDOWS_PER_PARAMETER,
    allow_underdetermined=False,
):
    """Validate the training design and describe poorly identified terms.

    Raises ``ValueError`` when there are fewer than
    ``min_windows_per_parameter`` training windows per parameter, unless
    ``allow_underdetermined`` is set. Returns a JSON-serializable report.
    """
    parameters = design.shape[1]
    required = int(np.ceil(min_windows_per_parameter * parameters))
    train = design[:train_count]
    test = design[train_count:]
    report = {
        "train_windows": int(train_count),
        "parameters": int(parameters),
        "min_windows_per_parameter": float(min_windows_per_parameter),
        "required_train_windows": required,
        "design_matrix_rank": int(np.linalg.matrix_rank(train)),
    }

    if train_count < required:
        suggestion = max(2, int(train_count // min_windows_per_parameter) - 1)
        message = (
            f"Only {train_count} training windows for {parameters} parameters "
            f"(need at least {required} at {min_windows_per_parameter:g} "
            "windows "
            f"per parameter). Reduce --max-families (e.g. to {suggestion}), "
            "use shorter --window-ms, pool more sessions, or pass "
            "--allow-underdetermined to fit anyway."
        )
        if not allow_underdetermined:
            raise ValueError(message)
        logger.warning("%s", message)
    report["underdetermined"] = bool(train_count < required)

    train_active = np.any(train > 0.0, axis=0)
    test_active = np.any(test > 0.0, axis=0)
    test_activity = test.sum(axis=0)
    total_test_activity = float(test_activity[1:].sum())
    unseen = [
        index
        for index in range(1, parameters)
        if not train_active[index] and test_active[index]
    ]
    report["families_without_training_activity"] = [
        terms[index]
        for index in range(1, parameters)
        if not train_active[index]
    ]
    report["test_families_unseen_in_training"] = [
        terms[index] for index in unseen
    ]
    report["test_activity_share_unseen_in_training"] = (
        float(test_activity[unseen].sum()) / total_test_activity
        if total_test_activity > 0.0
        else 0.0
    )
    report["aliased_terms"] = [terms[index] for index in aliased_columns(train)]
    return report


def log_identifiability_warnings(report, label=str):
    """Log rank deficiency, aliased terms and holdout-only families."""
    if report["design_matrix_rank"] < report["parameters"]:
        logger.warning(
            "Kernel-family activity is not fully identifiable "
            "(rank %s of %s); "
            "interpret overlapping family intervals conservatively.",
            report["design_matrix_rank"],
            report["parameters"],
        )
    if report["aliased_terms"]:
        aliased_terms = report["aliased_terms"]
        logger.warning(
            "%s terms are linear "
            "combinations of others in training and are prior-dominated: %s%s",
            len(aliased_terms),
            ", ".join(label(term) for term in aliased_terms[:8]),
            " ..." if len(aliased_terms) > 8 else "",
        )
    if report["test_families_unseen_in_training"]:
        logger.warning(
            "%s families are active in the holdout but never in training "
            "(%.1f%% of holdout family activity); "
            "their coefficients come from the prior.",
            len(report["test_families_unseen_in_training"]),
            100.0 * report["test_activity_share_unseen_in_training"],
        )
