#!/usr/bin/env python3
"""Compatibility entry point; implementation lives in the analyzer package."""

from energy_attribution.commands import weighted_kernel_energy


def main() -> None:
    """Runs the analyzer with command-line arguments."""
    weighted_kernel_energy.main()


if __name__ == "__main__":
    main()
