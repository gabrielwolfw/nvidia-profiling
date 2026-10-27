"""Stream NVML telemetry, CUPTI kernels and PM weights from capture CSVs."""

import csv
import math


def require_columns(reader, required_fields, description):
    """Raises ValueError when ``reader`` lacks any of ``required_fields``."""
    missing_fields = required_fields - set(reader.fieldnames or [])
    if missing_fields:
        raise ValueError(
            f"{description} is missing columns: {sorted(missing_fields)}"
        )


def deduplicate_timestamps(samples):
    """Keeps the last sample of each timestamp in sorted ``samples``."""
    deduplicated = []
    for timestamp, power in samples:
        if deduplicated and timestamp == deduplicated[-1][0]:
            deduplicated[-1] = (timestamp, power)
        else:
            deduplicated.append((timestamp, power))
    return deduplicated


def load_telemetry(path):
    samples = []
    skipped = 0
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        require_columns(reader, {"timestamp_ns", "power_mw"}, "Telemetry CSV")
        for row in reader:
            try:
                samples.append(
                    (int(row["timestamp_ns"]), float(row["power_mw"]))
                )
            except (TypeError, ValueError):
                skipped += 1
    if skipped:
        print(f"WARNING: Skipped {skipped} invalid telemetry rows.")
    samples.sort()
    deduplicated = deduplicate_timestamps(samples)
    if len(deduplicated) < 2:
        raise ValueError(
            "At least two distinct telemetry timestamps are required"
        )
    return deduplicated


def load_pm_kernel_metrics(path, weight_metric, min_coverage_pct):
    """Load valid PM weights by event ID without loading CSV rows twice.

    The PM file is already an offline timestamp join. A weight is used only
    when a kernel has sufficient PM coverage and a positive finite value for
    the selected PM metric. Event IDs are unique within one CUPTI trace.
    """
    pm_by_event_id = {}
    row_count = 0
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        require_columns(
            reader,
            {"event_id", "pm_coverage_pct", weight_metric},
            "PM kernel metrics CSV",
        )
        for row in reader:
            row_count += 1
            event_id = row.get("event_id")
            if event_id in pm_by_event_id:
                raise ValueError(f"Duplicate PM metrics event_id: {event_id}")
            try:
                coverage_pct = float(row["pm_coverage_pct"])
                weight = float(row[weight_metric])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(coverage_pct) or not math.isfinite(weight):
                continue
            eligible_weight = (
                weight
                if coverage_pct >= min_coverage_pct and weight > 0.0
                else None
            )
            pm_by_event_id[event_id] = (coverage_pct, eligible_weight)
    if row_count == 0:
        raise ValueError("PM kernel metrics CSV is empty")
    return pm_by_event_id


def parse_kernel_row(row):
    try:
        start = int(row["start_ns"])
        end = int(row["end_ns"])
    except (KeyError, TypeError, ValueError):
        return None
    if end <= start:
        return None
    return start, end


def scan_kernel_metadata(path, pm_by_event_id):
    """Validates kernel order and counts kernels with PM weights."""
    metadata = {
        "count": 0,
        "first_start": None,
        "last_end": None,
        "pm_matched": 0,
        "pm_eligible": 0,
    }
    skipped = 0
    previous_start = None
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        require_columns(
            reader, {"event_id", "start_ns", "end_ns"}, "Kernel activity CSV"
        )
        for row in reader:
            interval = parse_kernel_row(row)
            if interval is None:
                skipped += 1
                continue
            start, end = interval
            if previous_start is not None and start < previous_start:
                raise ValueError(
                    "Kernel activity rows must be sorted by start_ns for "
                    "streaming analysis"
                )
            previous_start = start
            _record_kernel(metadata, start, end)
            _record_pm_match(metadata, pm_by_event_id.get(row.get("event_id")))
    if skipped:
        print(f"WARNING: Skipped {skipped} invalid kernel rows.")
    return metadata


def _record_kernel(metadata, start, end):
    metadata["count"] += 1
    if metadata["first_start"] is None:
        metadata["first_start"] = start
    last_end = metadata["last_end"]
    metadata["last_end"] = end if last_end is None else max(last_end, end)


def _record_pm_match(metadata, pm_values):
    if pm_values is not None:
        metadata["pm_matched"] += 1
        if pm_values[1] is not None:
            metadata["pm_eligible"] += 1


def iter_kernels(path, pm_by_event_id):
    """Yields (index, start, end, PM weight) for each valid kernel row."""
    valid_index = 0
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        for row in reader:
            interval = parse_kernel_row(row)
            if interval is None:
                continue
            start, end = interval
            pm_values = pm_by_event_id.get(row.get("event_id"))
            weight = pm_values[1] if pm_values is not None else None
            yield valid_index, start, end, weight
            valid_index += 1
