"""Kernel-family basis: rank kernels, fold rare ones into __other__, encode events."""

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np

from energy_attribution.io import KernelTable

logger = logging.getLogger(__name__)

OTHER_FAMILY = "__other__"


@dataclass
class KernelScan:  # pylint: disable=too-many-instance-attributes
    """Per-kernel-name statistics of one trace and its chosen family basis.

    ``table`` holds the loaded events until they are aggregated into windows;
    workflows release it (set it to ``None``) to bound memory across sessions.
    """

    table: Optional[KernelTable]
    families: list
    selected: set
    duration_by_name: Counter
    event_count_by_name: Counter
    first_start: int
    last_end: int
    valid_events: int


def source_names(family, all_names, selected):
    """Kernel names whose events are counted under ``family``."""
    return set(all_names) - set(selected) if family == OTHER_FAMILY else {family}


def select_families(duration_by_name, max_families):
    """Rank kernel names by total duration; fold the tail into __other__."""
    ranked = [name for name, _ in duration_by_name.most_common()]
    if len(ranked) > max_families:
        selected = ranked[: max_families - 1]
        return selected + [OTHER_FAMILY], set(selected)
    return ranked, set(ranked)


def family_columns(table, selected, family_index):
    """Map every event in ``table`` to its design-matrix family column."""
    lookup = np.empty(len(table.names), dtype=np.int64)
    for code, name in enumerate(table.names):
        family = name if name in selected else OTHER_FAMILY
        if family not in family_index:
            raise ValueError(
                f"Kernel {name!r} is outside the family basis and no "
                f"{OTHER_FAMILY} column exists"
            )
        lookup[code] = family_index[family]
    return lookup[table.name_codes]


def scan_kernel_families(table, max_families):
    """Summarize a loaded KernelTable and choose its family basis."""
    if not len(table):
        raise ValueError("No valid kernel events were found")
    if table.skipped:
        logger.warning("Skipped %d invalid kernel rows.", table.skipped)
    duration_by_name, event_count_by_name = table.name_statistics()
    families, selected = select_families(duration_by_name, max_families)
    return KernelScan(
        table=table,
        families=families,
        selected=selected,
        duration_by_name=duration_by_name,
        event_count_by_name=event_count_by_name,
        first_start=table.first_start,
        last_end=table.last_end,
        valid_events=len(table),
    )

