"""Matplotlib diagnostics for PM Sampling coverage of kernel intervals."""

from energy_attribution.pm_association import COMPLETE_COVERAGE_PCT

PLOT_FILES = ("pm_coverage_timeline.png", "pm_sampling_gaps.png")


def _plot_coverage(plt, output_dir, origin_ns, kernels, output_rows):
    kernel_time_s = [
        (kernel["start_ns"] - origin_ns) / 1_000_000_000 for kernel in kernels
    ]
    coverage_pct = [float(row["pm_coverage_pct"]) for row in output_rows]
    figure, axis = plt.subplots(figsize=(12, 4.8))
    axis.scatter(kernel_time_s, coverage_pct, s=7, alpha=0.65, linewidths=0)
    axis.axhline(
        COMPLETE_COVERAGE_PCT,
        color="tab:green",
        linestyle="--",
        linewidth=1,
        label="99.9% coverage target",
    )
    axis.set_title("PM coverage by kernel")
    axis.set_xlabel("Kernel start relative to first kernel (s)")
    axis.set_ylabel("PM coverage (%)")
    axis.set_ylim(-2, 102)
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / PLOT_FILES[0], dpi=160)
    plt.close(figure)


def _plot_gaps(plt, output_dir, origin_ns, gap_events, gap_threshold_ns):
    figure, axis = plt.subplots(figsize=(12, 4.8))
    threshold_ms = gap_threshold_ns / 1_000_000
    if gap_events:
        gap_time_s = [
            ((start_ns + end_ns) / 2 - origin_ns) / 1_000_000_000
            for start_ns, end_ns, _ in gap_events
        ]
        gap_duration_ms = [
            duration_ns / 1_000_000 for _, _, duration_ns in gap_events
        ]
        axis.scatter(gap_time_s, gap_duration_ms, s=20, color="tab:red")
        axis.set_yscale("log")
    else:
        axis.text(
            0.5,
            0.5,
            "No PM gaps above the reporting threshold",
            transform=axis.transAxes,
            ha="center",
            va="center",
        )
    axis.axhline(
        threshold_ms,
        color="tab:orange",
        linestyle="--",
        linewidth=1,
        label=f"Reporting threshold ({threshold_ms:.3f} ms)",
    )
    axis.set_title("PM sampling gaps")
    axis.set_xlabel("Gap midpoint relative to first kernel (s)")
    axis.set_ylabel("Gap duration (ms)")
    axis.grid(True, alpha=0.25, which="both")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / PLOT_FILES[1], dpi=160)
    plt.close(figure)


def generate_plots(
    kernels, output_rows, gap_events, gap_threshold_ns, output_dir
):
    try:
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    except ImportError as error:
        raise RuntimeError(
            "--plot requires matplotlib; install it with: "
            "python3 -m pip install matplotlib"
        ) from error

    origin_ns = kernels[0]["start_ns"]
    _plot_coverage(plt, output_dir, origin_ns, kernels, output_rows)
    _plot_gaps(plt, output_dir, origin_ns, gap_events, gap_threshold_ns)
