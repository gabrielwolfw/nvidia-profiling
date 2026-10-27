"""Discover profiling sessions and aggregate each one into windows."""

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from energy_attribution.families import (
    KernelScan,
    family_columns,
    scan_kernel_families,
    select_families,
)
from energy_attribution.io import load_kernels, load_power_trace
from energy_attribution.windows import aggregate_family_activity, build_windows

logger = logging.getLogger(__name__)

TELEMETRY_FILE = "gpu_telemetry.csv"
KERNELS_FILE = "kernel_activity.csv"


@dataclass
class SessionInput:
    """One session directory and, once scanned, its kernel statistics."""

    name: str
    telemetry: Path
    kernels: Path
    scan: KernelScan = None


@dataclass
class SessionScan:
    """All usable sessions below a root and their shared family basis."""

    sessions: list
    excluded_sessions: list
    families: list
    selected: set
    duration_by_name: Counter = field(default_factory=Counter)
    event_count_by_name: Counter = field(default_factory=Counter)


@dataclass
class SessionWindows:
    """One session aggregated into windows (rows carry session labels)."""

    name: str
    rows: list
    activity_ns: np.ndarray
    target_power_w: np.ndarray
    train_count: int = 0


def session_files(directory):
    """Return the telemetry and kernel CSV paths of one session directory."""
    telemetry = directory / TELEMETRY_FILE
    kernels = directory / KERNELS_FILE
    missing = [str(path) for path in (telemetry, kernels) if not path.is_file()]
    if missing:
        raise ValueError("Session is missing: " + ", ".join(missing))
    return telemetry, kernels


def discover_session_paths(root):
    """Sessions are the subdirectories of ``root`` holding both CSV files."""
    sessions = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        telemetry = directory / TELEMETRY_FILE
        kernels = directory / KERNELS_FILE
        if telemetry.is_file() and kernels.is_file():
            sessions.append(SessionInput(directory.name, telemetry, kernels))
    if not sessions:
        raise ValueError("No complete profiling sessions were found")
    return sessions


def scan_sessions(session_inputs, max_families):
    """Load every session's kernels and pick one family basis for all of them.

    Sessions without valid kernel events are excluded with their reason.
    """
    active_sessions = []
    excluded_sessions = []
    duration_by_name = Counter()
    event_count_by_name = Counter()
    for session in session_inputs:
        try:
            logger.info("Loading kernel events for %s...", session.name)
            session.scan = scan_kernel_families(
                load_kernels(session.kernels), max_families=2
            )
        except ValueError as error:
            excluded_sessions.append({"session": session.name, "reason": str(error)})
            logger.info("Skipping %s: %s", session.name, error)
            continue
        duration_by_name.update(session.scan.duration_by_name)
        event_count_by_name.update(session.scan.event_count_by_name)
        active_sessions.append(session)

    if not active_sessions:
        raise ValueError("No session contains valid CUDA kernel events")
    families, selected = select_families(duration_by_name, max_families)
    return SessionScan(
        sessions=active_sessions,
        excluded_sessions=excluded_sessions,
        families=families,
        selected=selected,
        duration_by_name=duration_by_name,
        event_count_by_name=event_count_by_name,
    )


def build_session_windows(session, selected, family_index, window_ns, idle_power_mw):
    """Aggregate one scanned session; releases its KernelTable afterwards."""
    name, scan = session.name, session.scan
    logger.info("\n=== Aggregating %s ===", name)
    trace = load_power_trace(session.telemetry)
    analysis_start = max(trace.first_ns, scan.first_start)
    analysis_end = min(trace.last_ns, scan.last_end)
    if analysis_end <= analysis_start:
        raise ValueError(f"Telemetry and kernels do not overlap in {name}")
    table, scan.table = scan.table, None
    activity_ns, union_active_ns = aggregate_family_activity(
        table,
        family_columns(table, selected, family_index),
        len(family_index),
        analysis_start,
        analysis_end,
        window_ns,
    )
    del table
    rows, target_power_w = build_windows(
        trace, union_active_ns, analysis_start, analysis_end, window_ns,
        idle_power_mw,
    )
    for index, row in enumerate(rows):
        row["session"] = name
        row["session_window_index"] = index
    return SessionWindows(name, rows, activity_ns, target_power_w)
