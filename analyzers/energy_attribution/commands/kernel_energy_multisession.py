"""``nvidia-kernel-energy-multisession``: one pooled model across sessions."""

from energy_attribution.api import run_multisession
from energy_attribution.cli import run_command
from energy_attribution.config import MultiSessionConfig
from energy_attribution.reporting.text import print_multisession_report

DESCRIPTION = (
    "Fit one non-negative Bayesian kernel-family model across all compatible "
    "sessions below a results directory."
)


def main(argv=None):
    return run_command(
        MultiSessionConfig,
        DESCRIPTION,
        run_multisession,
        print_multisession_report,
        argv,
    )


if __name__ == "__main__":
    main()
