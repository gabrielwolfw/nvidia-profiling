"""``nvidia-energy``: one telemetry-feature energy model across sessions."""

from energy_attribution.api import run_telemetry_multisession
from energy_attribution.cli import run_command
from energy_attribution.config import TelemetryMultiSessionConfig
from energy_attribution.reporting.text import print_telemetry_multisession_report

DESCRIPTION = (
    "Fit one Bayesian GPU energy model across independent session directories "
    "and evaluate the final temporal fraction of every session as a holdout."
)


def main(argv=None):
    return run_command(
        TelemetryMultiSessionConfig, DESCRIPTION, run_telemetry_multisession,
        print_telemetry_multisession_report, argv,
    )


if __name__ == "__main__":
    main()
