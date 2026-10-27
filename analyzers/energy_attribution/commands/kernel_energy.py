"""``nvidia-kernel-energy``: attribute one session's energy to CUDA kernels."""

from energy_attribution.api import run_single_session
from energy_attribution.cli import run_command
from energy_attribution.config import SingleSessionConfig
from energy_attribution.reporting.text import print_single_session_report

DESCRIPTION = (
    "Fit a non-negative Bayesian kernel-family power model and attribute "
    "measured NVML energy to CUPTI kernel events."
)


def main(argv=None):
    return run_command(
        SingleSessionConfig,
        DESCRIPTION,
        run_single_session,
        print_single_session_report,
        argv,
    )


if __name__ == "__main__":
    main()
