"""Columnar loaders for CUPTI kernel traces and NVML power telemetry."""

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


NS_PER_SECOND = 1_000_000_000
KERNEL_COLUMNS = ("start_ns", "end_ns", "kernel_name")
KERNEL_CHUNK_ROWS = 200_000


def require_fields(fieldnames, required, label):
    missing = set(required) - set(fieldnames or [])
    if missing:
        raise ValueError(f"{label} CSV is missing columns: {sorted(missing)}")


def read_csv_header(path):
    with path.open(newline="", encoding="utf-8") as source:
        return next(csv.reader(source), [])


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@dataclass
class PowerTrace:
    """Sorted, timestamp-deduplicated NVML power samples."""

    timestamps_ns: np.ndarray
    power_mw: np.ndarray

    @property
    def first_ns(self):
        return int(self.timestamps_ns[0])

    @property
    def last_ns(self):
        return int(self.timestamps_ns[-1])


def load_power_trace(path):
    samples = []
    skipped = 0
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        require_fields(reader.fieldnames, {"timestamp_ns", "power_mw"}, "Telemetry")
        for row in reader:
            try:
                samples.append((int(row["timestamp_ns"]), float(row["power_mw"])))
            except (TypeError, ValueError):
                skipped += 1
    if skipped:
        print(f"WARNING: Skipped {skipped} invalid telemetry rows.")
    samples.sort()
    deduplicated = []
    for timestamp, power in samples:
        if deduplicated and timestamp == deduplicated[-1][0]:
            deduplicated[-1] = (timestamp, power)
        else:
            deduplicated.append((timestamp, power))
    if len(deduplicated) < 2:
        raise ValueError("At least two distinct telemetry timestamps are required")
    return PowerTrace(
        np.fromiter((sample[0] for sample in deduplicated), dtype=np.int64),
        np.fromiter((sample[1] for sample in deduplicated), dtype=np.float64),
    )


@dataclass
class KernelTable:
    """Valid kernel events in file order, with names stored as integer codes.

    Rows are kept only when start/end parse as integers, end > start and the
    kernel name is non-empty, matching the original streaming parsers.
    """

    path: Path
    start_ns: np.ndarray
    end_ns: np.ndarray
    name_codes: np.ndarray
    names: list
    event_ids: np.ndarray = None
    skipped: int = 0

    def __len__(self):
        return len(self.start_ns)

    @property
    def first_start(self):
        return int(self.start_ns[0])

    @property
    def last_end(self):
        return int(self.end_ns.max())

    def name_statistics(self):
        """Return (duration_by_name, event_count_by_name) Counters.

        Names are inserted in first-appearance order so ``most_common`` breaks
        ties exactly like the original row-by-row Counter updates.
        """
        durations = np.zeros(len(self.names), dtype=np.int64)
        np.add.at(durations, self.name_codes, self.end_ns - self.start_ns)
        counts = np.bincount(self.name_codes, minlength=len(self.names))
        return (
            Counter(dict(zip(self.names, durations.tolist()))),
            Counter(dict(zip(self.names, counts.tolist()))),
        )


def _parse_int_column(values):
    """Parse a string column to int64, returning (values, valid_mask)."""
    try:
        return values.astype(np.int64).to_numpy(), None
    except (TypeError, ValueError, OverflowError):
        parsed = np.zeros(len(values), dtype=np.int64)
        valid = np.zeros(len(values), dtype=bool)
        for index, value in enumerate(values.tolist()):
            try:
                parsed[index] = int(value)
                valid[index] = True
            except (TypeError, ValueError, OverflowError):
                pass
        return parsed, valid


def load_kernels(path, load_event_ids=False, chunk_rows=KERNEL_CHUNK_ROWS):
    """Read a CUPTI kernel CSV once into compact numpy columns."""
    header = read_csv_header(path)
    require_fields(header, KERNEL_COLUMNS, "kernels")
    columns = list(KERNEL_COLUMNS)
    has_event_ids = load_event_ids and "event_id" in header
    if has_event_ids:
        columns.append("event_id")

    code_by_name = {}
    names = []
    starts, ends, codes, event_ids = [], [], [], []
    skipped = 0
    reader = pd.read_csv(
        path,
        usecols=columns,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        chunksize=chunk_rows,
        encoding="utf-8",
    )
    for chunk in reader:
        start, start_valid = _parse_int_column(chunk["start_ns"])
        end, end_valid = _parse_int_column(chunk["end_ns"])
        # Truncated rows (e.g. a partially flushed last line) yield NaN.
        name_values = chunk["kernel_name"].fillna("").to_numpy()
        valid = (end > start) & (name_values != "")
        for mask in (start_valid, end_valid):
            if mask is not None:
                valid &= mask
        skipped += int(len(valid) - valid.sum())

        chunk_codes, chunk_names = pd.factorize(name_values[valid])
        global_codes = np.empty(len(chunk_names), dtype=np.int32)
        for local_code, name in enumerate(chunk_names):
            code = code_by_name.get(name)
            if code is None:
                code = code_by_name[name] = len(names)
                names.append(name)
            global_codes[local_code] = code

        starts.append(start[valid])
        ends.append(end[valid])
        codes.append(global_codes[chunk_codes])
        if has_event_ids:
            ids, ids_valid = _parse_int_column(chunk["event_id"])
            if ids_valid is not None:
                ids[~ids_valid] = -1
            event_ids.append(ids[valid])

    def concatenate(parts, dtype):
        return np.concatenate(parts) if parts else np.empty(0, dtype=dtype)

    table = KernelTable(
        path=path,
        start_ns=concatenate(starts, np.int64),
        end_ns=concatenate(ends, np.int64),
        name_codes=concatenate(codes, np.int32),
        names=names,
        event_ids=concatenate(event_ids, np.int64) if has_event_ids else None,
        skipped=skipped,
    )
    if len(table) > 1 and np.any(np.diff(table.start_ns) < 0):
        raise ValueError("Kernel rows must be sorted by start_ns for streaming analysis")
    return table
