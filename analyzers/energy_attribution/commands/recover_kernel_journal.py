"""Recovers kernel_activity.csv from the journal of an interrupted capture."""

from __future__ import annotations

import argparse
import pathlib
import sys

from energy_attribution import kernel_journal


def parse_args() -> argparse.Namespace:
    """Parses the command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Write kernel_activity.csv from the kernel_activity.journal "
            "left by a profiled process that did not exit normally."
        )
    )
    parser.add_argument(
        "journal",
        type=pathlib.Path,
        help="kernel_activity.journal, or the capture directory holding it",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        help="CSV to write (default: kernel_activity.csv next to the journal)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace the output CSV if it already exists",
    )
    return parser.parse_args()


def main() -> None:
    """Runs the recovery with command-line arguments."""
    args = parse_args()
    journal_path = args.journal
    if journal_path.is_dir():
        journal_path = journal_path / "kernel_activity.journal"
    output_path = args.output or journal_path.with_name("kernel_activity.csv")
    if output_path.exists() and not args.force:
        sys.exit(f"{output_path} already exists; pass --force to replace it")

    try:
        journal = kernel_journal.read_kernel_journal(journal_path)
    except (OSError, ValueError) as error:
        sys.exit(f"Cannot read the kernel journal: {error}")

    rows = kernel_journal.write_kernel_csv(journal, output_path)
    print(f"Kernel rows: {rows}")
    print(f"Kernel names: {len(journal.names)}")
    if journal.truncated:
        print(
            "The journal ends inside a chunk; the incomplete tail was skipped."
        )
    print(f"Kernel CSV: {output_path}")


if __name__ == "__main__":
    main()
