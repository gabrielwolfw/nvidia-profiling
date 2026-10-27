"""``nvidia-kernel-energy-transfer``: cross-GPU transfer with calibration."""

from energy_attribution.api import run_cross_gpu
from energy_attribution.cli import run_command
from energy_attribution.config import CrossGpuConfig
from energy_attribution.reporting.text import print_cross_gpu_report

DESCRIPTION = (
    "Train a pooled Bayesian kernel-family model on one collection of "
    "sessions, then compare zero-shot and minimally calibrated predictions "
    "on one external GPU session."
)


def main(argv=None):
    return run_command(
        CrossGpuConfig, DESCRIPTION, run_cross_gpu, print_cross_gpu_report, argv,
    )


if __name__ == "__main__":
    main()
