#!/usr/bin/env python3
"""Compatibility entry point; implementation lives in the analyzer package."""

from energy_attribution.commands import pm_sampling_plot


def main() -> None:
    """Runs the analyzer with command-line arguments."""
    pm_sampling_plot.main()


if __name__ == "__main__":
    main()
