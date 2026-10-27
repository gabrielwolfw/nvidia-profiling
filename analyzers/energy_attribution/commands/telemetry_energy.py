"""``nvidia-energy-session``: telemetry-feature power model for one session."""

from energy_attribution.api import run_telemetry_session
from energy_attribution.cli import run_command
from energy_attribution.config import TelemetrySessionConfig
from energy_attribution.reporting.text import print_telemetry_session_report

DESCRIPTION = (
    "Aggregate synchronized NVML/CUPTI data into time windows and fit a "
    "regularized Bayesian linear power model."
)


def main(argv=None):
    return run_command(
        TelemetrySessionConfig, DESCRIPTION, run_telemetry_session,
        print_telemetry_session_report, argv,
    )


if __name__ == "__main__":
    main()
