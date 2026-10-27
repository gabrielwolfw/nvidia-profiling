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
        import matplotlib.pyplot as plt
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



def create_multisession_plots(
    output_dir, rows, power_targets, predicted_power, power_lower,
    power_upper, energy_targets, predicted_energy, energy_lower,
    energy_upper, train_count, session_energy_rows, platform_label="Thor",
):  # pylint: disable=too-many-arguments,too-many-locals,too-many-statements
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("--plot requires matplotlib") from error

    test_rows = rows[train_count:]
    actual = power_targets[train_count:]
    expected = predicted_power[train_count:]
    test_lower = power_lower[train_count:]
    test_upper = power_upper[train_count:]
    names = list(dict.fromkeys(row["session"] for row in test_rows))
    colors = plt.get_cmap("tab20", len(names))

    figure, (fit_axis, residual_axis) = plt.subplots(
        2, 1, figsize=(11, 10), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]},
    )
    for session_index, name in enumerate(names):
        indices = np.asarray(
            [index for index, row in enumerate(test_rows) if row["session"] == name]
        )
        color = colors(session_index)
        fit_axis.vlines(
            actual[indices], test_lower[indices], test_upper[indices],
            color=color, alpha=0.15, linewidth=0.8,
        )
        fit_axis.scatter(
            actual[indices], expected[indices], s=22, alpha=0.75,
            color=color, label=name,
        )
        residual_axis.scatter(
            actual[indices], expected[indices] - actual[indices],
            s=22, alpha=0.75, color=color,
        )

    minimum = float(min(actual.min(), expected.min(), test_lower.min()))
    maximum = float(max(actual.max(), expected.max(), test_upper.max()))
    fit_axis.plot(
        [minimum, maximum], [minimum, maximum], color="black",
        linestyle="--", linewidth=1.2, label="Central line: predicted = measured",
    )
    fit_axis.set_ylabel("Posterior predicted power (W)")
    fit_axis.set_title(f"Temporal holdout predictions across {platform_label} workloads")
    fit_axis.grid(alpha=0.2)
    fit_axis.legend(fontsize=7, ncol=2, loc="best")
    residual_axis.axhline(0.0, color="black", linestyle="--", linewidth=1.2)
    residual_axis.set_xlabel("Measured power (W)")
    residual_axis.set_ylabel("Distance from central line (W)")
    residual_axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(
        output_dir / "bayesian_power_temporal_holdout.png", dpi=180
    )
    plt.close(figure)

    summaries = []
    for name in names:
        indices = np.asarray(
            [index for index, row in enumerate(test_rows) if row["session"] == name]
        )
        summaries.append(
            (
                name,
                float(actual[indices].mean()),
                float(expected[indices].mean()),
            )
        )
    figure, axis = plt.subplots(figsize=(8, 7))
    for session_index, (name, measured, modeled) in enumerate(summaries):
        axis.scatter(
            measured, modeled, s=75, color=colors(session_index), label=name
        )
    minimum = min(min(item[1], item[2]) for item in summaries)
    maximum = max(max(item[1], item[2]) for item in summaries)
    axis.plot(
        [minimum, maximum], [minimum, maximum], color="black",
        linestyle="--", linewidth=1.2,
    )
    axis.set_xlabel("Mean measured holdout power (W)")
    axis.set_ylabel("Mean predicted holdout power (W)")
    axis.set_title("Bayesian power by workload")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=7, loc="best")
    figure.tight_layout()
    figure.savefig(output_dir / "bayesian_power_by_workload.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 7))
    measured_energy = np.asarray(
        [float(row["measured_energy_j"]) for row in session_energy_rows]
    )
    session_predicted_energy = np.asarray(
        [float(row["predicted_energy_j"]) for row in session_energy_rows]
    )
    for session_index, row in enumerate(session_energy_rows):
        axis.scatter(
            measured_energy[session_index], session_predicted_energy[session_index],
            s=75, color=colors(session_index), label=row["session"],
        )
    minimum = float(min(measured_energy.min(), session_predicted_energy.min()))
    maximum = float(max(measured_energy.max(), session_predicted_energy.max()))
    axis.plot(
        [minimum, maximum], [minimum, maximum], color="black",
        linestyle="--", linewidth=1.2,
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

    names = list(dict.fromkeys(row["session"] for row in rows))
    figure, axes = plt.subplots(
        len(names), 1, figsize=(12, 2.25 * len(names)), sharex=False,
        constrained_layout=True,
    )
    if len(names) == 1:
        axes = [axes]
    for session_index, (name, axis) in enumerate(zip(names, axes)):
        indices = sorted(
            (index for index, row in enumerate(rows) if row["session"] == name),
            key=lambda index: rows[index]["session_window_index"],
        )
        elapsed = np.asarray([rows[index]["session_elapsed_s"] for index in indices])
        measured = energy_targets[indices]
        modeled = predicted_energy[indices]
        interval_lower = energy_lower[indices]
        interval_upper = energy_upper[indices]
        test_positions = [
            position for position, index in enumerate(indices)
            if rows[index]["split"] == "test"
        ]
        axis.plot(elapsed, measured, color="black", linewidth=1.0, label="Measured")
        axis.plot(
            elapsed, modeled, color=colors(session_index), linewidth=1.1,
            label="Posterior mean",
        )
        axis.fill_between(
            elapsed, interval_lower, interval_upper,
            color=colors(session_index), alpha=0.18,
            label="95% predictive interval",
        )
        if test_positions:
            test_start = elapsed[test_positions[0]]
            axis.axvspan(
                test_start, elapsed[-1], color="grey", alpha=0.10,
                label="Temporal holdout",
            )
        axis.set_ylabel("Energy (J/window)")
        axis.set_title(name, loc="left", fontsize=9)
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8, ncol=4, loc="upper right")
    axes[-1].set_xlabel("Elapsed session time (s)")
    figure.suptitle(
        f"Bayesian energy timeline across {platform_label} workloads", fontsize=15
    )
    figure.savefig(output_dir / "bayesian_energy_timeline.png", dpi=180)
    plt.close(figure)

    test_energy = energy_targets[train_count:]
    test_predicted_energy = predicted_energy[train_count:]
    test_energy_lower = energy_lower[train_count:]
    test_energy_upper = energy_upper[train_count:]
    figure, (fit_axis, residual_axis) = plt.subplots(
        2, 1, figsize=(11, 10), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]},
    )
    for session_index, name in enumerate(names):
        indices = np.asarray(
            [index for index, row in enumerate(test_rows) if row["session"] == name]
        )
        color = colors(session_index)
        fit_axis.vlines(
            test_energy[indices], test_energy_lower[indices],
            test_energy_upper[indices], color=color, alpha=0.15, linewidth=0.8,
        )
        fit_axis.scatter(
            test_energy[indices], test_predicted_energy[indices],
            s=22, alpha=0.75, color=color, label=name,
        )
        residual_axis.scatter(
            test_energy[indices],
            test_predicted_energy[indices] - test_energy[indices],
            s=22, alpha=0.75, color=color,
        )
    minimum = float(
        min(test_energy.min(), test_predicted_energy.min(), test_energy_lower.min())
    )
    maximum = float(
        max(test_energy.max(), test_predicted_energy.max(), test_energy_upper.max())
    )
    fit_axis.plot(
        [minimum, maximum], [minimum, maximum], color="black",
        linestyle="--", linewidth=1.2, label="Central line: predicted = measured",
    )
    fit_axis.set_ylabel("Posterior predicted energy (J/window)")
    fit_axis.set_title(f"Bayesian energy temporal holdout across {platform_label} workloads")
    fit_axis.grid(alpha=0.2)
    fit_axis.legend(fontsize=7, ncol=2, loc="best")
    residual_axis.axhline(0.0, color="black", linestyle="--", linewidth=1.2)
    residual_axis.set_xlabel("Measured energy (J/window)")
    residual_axis.set_ylabel("Distance from central line (J/window)")
    residual_axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(
        output_dir / "bayesian_energy_temporal_holdout.png", dpi=180
    )
    plt.close(figure)

