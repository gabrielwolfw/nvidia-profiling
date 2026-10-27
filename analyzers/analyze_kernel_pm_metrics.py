#!/usr/bin/env python3
"""Compatibility entry point; implementation lives in the analyzer package."""

from energy_attribution.commands import kernel_pm_metrics


def main() -> None:
    """Runs the analyzer with command-line arguments."""
    kernel_pm_metrics.main()


if __name__ == "__main__":
    main()
