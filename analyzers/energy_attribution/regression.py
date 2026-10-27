"""Non-negative Bayesian linear regression sampled with a Gibbs sampler."""

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class PosteriorSamples:
    """Posterior draws of a non-negative linear power model.

    ``coefficient_draws`` is (draws, parameters) in watts per unit of the
    design column, ``noise_draws`` the matching observation variance (W^2)
    and ``prediction_draws`` the latent mean power (draws, windows) for every
    row of the design, train and holdout alike.
    """

    coefficient_draws: np.ndarray
    noise_draws: np.ndarray
    prediction_draws: np.ndarray
    column_scale: np.ndarray


def sample_lower_truncated_normal(mean, standard_deviation, rng):
    """Draw from N(mean, sd) truncated to [0, infinity)."""
    if standard_deviation <= 0.0:
        return max(0.0, mean)
    lower = -mean / standard_deviation
    if lower <= 0.0:
        while True:
            value = rng.normal()
            if value >= lower:
                return mean + standard_deviation * value
    alpha = 0.5 * (lower + math.sqrt(lower * lower + 4.0))
    while True:
        value = lower + rng.exponential(1.0 / alpha)
        if rng.random() <= math.exp(-0.5 * (value - alpha) ** 2):
            return mean + standard_deviation * value


def fit_nonnegative_bayesian_regression(
    design,
    target,
    train_count,
    prior_scale_w,
    noise_prior_scale_w,
    burn_in,
    posterior_draws,
    thin,
    seed,
):
    train_design = design[:train_count]
    train_target = target[:train_count]
    column_scale = np.sqrt(np.mean(train_design * train_design, axis=0))
    column_scale[column_scale < 1e-12] = 1.0
    scaled_design = design / column_scale
    train_scaled = scaled_design[:train_count]

    beta = np.maximum(
        0.0, np.linalg.lstsq(train_scaled, train_target, rcond=None)[0]
    )
    residual = train_target - train_scaled @ beta
    sigma2 = max(float(np.mean(residual * residual)), 1e-6)
    prior_variance = prior_scale_w**2
    noise_shape = 2.0
    noise_rate = noise_prior_scale_w**2
    rng = np.random.default_rng(seed)
    draws = np.empty((posterior_draws, design.shape[1]), dtype=np.float64)
    noise_draws = np.empty(posterior_draws, dtype=np.float64)
    total_iterations = burn_in + posterior_draws * thin
    saved = 0

    for iteration in range(total_iterations):
        for column in range(train_scaled.shape[1]):
            values = train_scaled[:, column]
            residual += values * beta[column]
            precision = float(values @ values) / sigma2 + 1.0 / prior_variance
            variance = 1.0 / precision
            mean = variance * float(values @ residual) / sigma2
            beta[column] = sample_lower_truncated_normal(
                mean, math.sqrt(variance), rng
            )
            residual -= values * beta[column]

        posterior_shape = noise_shape + train_count / 2.0
        posterior_rate = noise_rate + 0.5 * float(residual @ residual)
        sigma2 = 1.0 / rng.gamma(posterior_shape, 1.0 / posterior_rate)

        if iteration >= burn_in and (iteration - burn_in) % thin == 0:
            draws[saved] = beta / column_scale
            noise_draws[saved] = sigma2
            saved += 1

    return PosteriorSamples(draws, noise_draws, draws @ design.T, column_scale)


def posterior_summary(values):
    """Return (mean, median, lower 2.5%, upper 97.5%) of 1-D draws as floats."""
    lower, upper = np.quantile(values, [0.025, 0.975])
    return float(values.mean()), float(np.median(values)), float(lower), float(upper)


def coefficient_summary_rows(
    terms, coefficient_draws, column_scale, *,
    prefix="posterior_power_", suffix="_w", include_median=False,
):
    """One CSV row per model term with posterior mean and 95% interval.

    Column names are ``{prefix}{mean|median|lower_95|upper_95}{suffix}`` plus
    ``activity_rms_scale``; the kernel and transfer CSVs differ only there.
    """
    rows = []
    for index, term in enumerate(terms):
        mean, median, lower, upper = posterior_summary(coefficient_draws[:, index])
        row = {"term": term, f"{prefix}mean{suffix}": mean}
        if include_median:
            row[f"{prefix}median{suffix}"] = median
        row[f"{prefix}lower_95{suffix}"] = lower
        row[f"{prefix}upper_95{suffix}"] = upper
        row["activity_rms_scale"] = float(column_scale[index])
        rows.append(row)
    return rows

