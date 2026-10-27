"""Matplotlib diagnostics for the telemetry-feature energy models."""

import numpy as np

from energy_attribution.units import NS_PER_SECOND


def create_session_plots(
    output_dir,
    rows,
    targets,
    predicted,
    predictive_lower,
    predictive_upper,
    train_count,
):
    try:
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    except ImportError as error:
        raise RuntimeError("--plot requires matplotlib") from error

    elapsed_seconds = np.asarray(
        [
            (row["window_start_ns"] - rows[0]["window_start_ns"])
            / NS_PER_SECOND
            for row in rows
        ]
    )
    figure, axis = plt.subplots(figsize=(11, 5))
    axis.plot(elapsed_seconds, targets, label="Measured", linewidth=1.2)
    axis.plot(elapsed_seconds, predicted, label="Posterior mean", linewidth=1.2)
    axis.fill_between(
        elapsed_seconds,
        predictive_lower,
        predictive_upper,
        alpha=0.2,
        label="95% predictive interval",
    )
    axis.axvline(
        elapsed_seconds[train_count],
        color="black",
        linestyle="--",
        linewidth=1,
        label="Test split",
    )
    axis.set_xlabel("Elapsed time (s)")
    axis.set_ylabel("Power (W)")
    axis.set_title("Bayesian GPU power prediction")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "bayesian_power_timeline.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(targets[train_count:], predicted[train_count:], alpha=0.65)
    limits = [
        min(targets[train_count:].min(), predicted[train_count:].min()),
        max(targets[train_count:].max(), predicted[train_count:].max()),
    ]
    axis.plot(limits, limits, color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("Measured power (W)")
    axis.set_ylabel("Predicted power (W)")
    axis.set_title("Temporal holdout predictions")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "bayesian_power_test_scatter.png", dpi=160)
    plt.close(figure)


def _session_indices(rows, name):
    return np.asarray(
        [index for index, row in enumerate(rows) if row["session"] == name]
    )


def _identity_line(axis, minimum, maximum, **kwargs):
    axis.plot(
        [minimum, maximum],
        [minimum, maximum],
        color="black",
        linestyle="--",
        linewidth=1.2,
        **kwargs,
    )


def _plot_holdout_fit(plt, path, holdout, groups, colors, labels):
    """Predicted-versus-measured scatter with a residual panel below.

    Args:
        plt: The ``matplotlib.pyplot`` module.
        path: Output PNG path.
        holdout: Test-split (measured, predicted, lower, upper) arrays.
        groups: (session name, test-row indices) pairs, in color order.
        colors: Colormap indexed by group position.
        labels: Dict with ``ylabel``, ``title``, ``xlabel`` and ``residual``.
    """
    actual, expected, lower, upper = holdout
    figure, (fit_axis, residual_axis) = plt.subplots(
        2,
        1,
        figsize=(11, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]},
    )
    for session_index, (name, indices) in enumerate(groups):
        color = colors(session_index)
        fit_axis.vlines(
            actual[indices],
            lower[indices],
            upper[indices],
            color=color,
            alpha=0.15,
            linewidth=0.8,
        )
        fit_axis.scatter(
            actual[indices],
            expected[indices],
            s=22,
            alpha=0.75,
            color=color,
            label=name,
        )
        residual_axis.scatter(
            actual[indices],
            expected[indices] - actual[indices],
            s=22,
            alpha=0.75,
            color=color,
        )
    _identity_line(
        fit_axis,
        float(min(actual.min(), expected.min(), lower.min())),
        float(max(actual.max(), expected.max(), upper.max())),
        label="Central line: predicted = measured",
    )
    fit_axis.set_ylabel(labels["ylabel"])
    fit_axis.set_title(labels["title"])
    fit_axis.grid(alpha=0.2)
    fit_axis.legend(fontsize=7, ncol=2, loc="best")
    residual_axis.axhline(0.0, color="black", linestyle="--", linewidth=1.2)
    residual_axis.set_xlabel(labels["xlabel"])
    residual_axis.set_ylabel(labels["residual"])
    residual_axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_power_by_workload(plt, output_dir, groups, holdout, colors):
    """Mean measured versus predicted holdout power for each workload."""
    actual, expected = holdout[0], holdout[1]
    summaries = [
        (name, float(actual[indices].mean()), float(expected[indices].mean()))
        for name, indices in groups
    ]
    figure, axis = plt.subplots(figsize=(8, 7))
    for session_index, (name, measured, modeled) in enumerate(summaries):
        axis.scatter(
            measured, modeled, s=75, color=colors(session_index), label=name
        )
    _identity_line(
        axis,
        min(min(item[1], item[2]) for item in summaries),
        max(max(item[1], item[2]) for item in summaries),
    )
    axis.set_xlabel("Mean measured holdout power (W)")
    axis.set_ylabel("Mean predicted holdout power (W)")
    axis.set_title("Bayesian power by workload")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=7, loc="best")
    figure.tight_layout()
    figure.savefig(output_dir / "bayesian_power_by_workload.png", dpi=180)
    plt.close(figure)


def _plot_session_energy(
    plt, output_dir, session_energy_rows, colors, platform_label
):
    """Measured versus predicted total energy for each session."""
    measured_energy = np.asarray(
        [float(row["measured_energy_j"]) for row in session_energy_rows]
    )
    predicted_energy = np.asarray(
        [float(row["predicted_energy_j"]) for row in session_energy_rows]
    )
    figure, axis = plt.subplots(figsize=(8, 7))
    for session_index, row in enumerate(session_energy_rows):
        axis.scatter(
            measured_energy[session_index],
            predicted_energy[session_index],
            s=75,
            color=colors(session_index),
            label=row["session"],
        )
    _identity_line(
        axis,
        float(min(measured_energy.min(), predicted_energy.min())),
        float(max(measured_energy.max(), predicted_energy.max())),
        label="Central line: predicted = measured",
    )
    axis.set_xlabel("Measured session energy (J)")
    axis.set_ylabel("Bayesian predicted session energy (J)")
    axis.set_title(f"Bayesian energy across {platform_label} workloads")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=7, ncol=2, loc="best")
    figure.tight_layout()
    figure.savefig(output_dir / "bayesian_energy_by_workload.png", dpi=180)
    plt.close(figure)


def _plot_session_timeline(axis, rows, name, energy, color):
    """Draws one session's measured and predicted energy per window."""
    measured, modeled, lower, upper = energy
    indices = sorted(
        (index for index, row in enumerate(rows) if row["session"] == name),
        key=lambda index: rows[index]["session_window_index"],
    )
    elapsed = np.asarray(
        [rows[index]["session_elapsed_s"] for index in indices]
    )
    test_positions = [
        position
        for position, index in enumerate(indices)
        if rows[index]["split"] == "test"
    ]
    axis.plot(
        elapsed,
        measured[indices],
        color="black",
        linewidth=1.0,
        label="Measured",
    )
    axis.plot(
        elapsed,
        modeled[indices],
        color=color,
        linewidth=1.1,
        label="Posterior mean",
    )
    axis.fill_between(
        elapsed,
        lower[indices],
        upper[indices],
        color=color,
        alpha=0.18,
        label="95% predictive interval",
    )
    if test_positions:
        axis.axvspan(
            elapsed[test_positions[0]],
            elapsed[-1],
            color="grey",
            alpha=0.10,
            label="Temporal holdout",
        )
    axis.set_ylabel("Energy (J/window)")
    axis.set_title(name, loc="left", fontsize=9)
    axis.grid(alpha=0.2)


def _plot_energy_timeline(plt, output_dir, rows, names, energy, context):
    """Stacks one energy timeline panel per session."""
    colors, platform_label = context
    figure, axes = plt.subplots(
        len(names),
        1,
        figsize=(12, 2.25 * len(names)),
        sharex=False,
        constrained_layout=True,
    )
    if len(names) == 1:
        axes = [axes]
    for session_index, (name, axis) in enumerate(zip(names, axes)):
        _plot_session_timeline(axis, rows, name, energy, colors(session_index))
    axes[0].legend(fontsize=8, ncol=4, loc="upper right")
    axes[-1].set_xlabel("Elapsed session time (s)")
    figure.suptitle(
        f"Bayesian energy timeline across {platform_label} workloads",
        fontsize=15,
    )
    figure.savefig(output_dir / "bayesian_energy_timeline.png", dpi=180)
    plt.close(figure)


def create_multisession_plots(
    output_dir,
    rows,
    power_targets,
    predicted_power,
    power_lower,
    power_upper,
    energy_targets,
    predicted_energy,
    energy_lower,
    energy_upper,
    train_count,
    session_energy_rows,
    platform_label="Thor",
):  # pylint: disable=too-many-arguments,too-many-locals
    try:
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    except ImportError as error:
        raise RuntimeError("--plot requires matplotlib") from error

    test_rows = rows[train_count:]
    test_names = list(dict.fromkeys(row["session"] for row in test_rows))
    all_names = list(dict.fromkeys(row["session"] for row in rows))
    colors = plt.get_cmap("tab20", len(test_names))
    power = (power_targets, predicted_power, power_lower, power_upper)
    energy = (energy_targets, predicted_energy, energy_lower, energy_upper)
    power_holdout = tuple(values[train_count:] for values in power)
    energy_holdout = tuple(values[train_count:] for values in energy)
    test_groups = [
        (name, _session_indices(test_rows, name)) for name in test_names
    ]

    _plot_holdout_fit(
        plt,
        output_dir / "bayesian_power_temporal_holdout.png",
        power_holdout,
        test_groups,
        colors,
        {
            "ylabel": "Posterior predicted power (W)",
            "title": (
                f"Temporal holdout predictions across {platform_label} "
                "workloads"
            ),
            "xlabel": "Measured power (W)",
            "residual": "Distance from central line (W)",
        },
    )
    _plot_power_by_workload(plt, output_dir, test_groups, power_holdout, colors)
    _plot_session_energy(
        plt, output_dir, session_energy_rows, colors, platform_label
    )
    _plot_energy_timeline(
        plt, output_dir, rows, all_names, energy, (colors, platform_label)
    )
    _plot_holdout_fit(
        plt,
        output_dir / "bayesian_energy_temporal_holdout.png",
        energy_holdout,
        [(name, _session_indices(test_rows, name)) for name in all_names],
        colors,
        {
            "ylabel": "Posterior predicted energy (J/window)",
            "title": (
                f"Bayesian energy temporal holdout across {platform_label} "
                "workloads"
            ),
            "xlabel": "Measured energy (J/window)",
            "residual": "Distance from central line (J/window)",
        },
    )
