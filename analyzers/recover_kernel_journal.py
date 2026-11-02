#!/usr/bin/env python3
"""Compatibility entry point; implementation lives in the analyzer package."""

from energy_attribution.commands import recover_kernel_journal


def main() -> None:
    """Runs the recovery with command-line arguments."""
    recover_kernel_journal.main()


if __name__ == "__main__":
    main()
