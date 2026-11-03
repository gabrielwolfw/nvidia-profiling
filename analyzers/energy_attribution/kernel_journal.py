"""Reads the binary kernel journal that the preload collector writes.

The collector appends CUPTI kernel records to ``kernel_activity.journal``
while the profiled process runs and replaces it with ``kernel_activity.csv``
at a normal exit. After a crash only the journal remains; this module reads
it and writes the CSV that the analyzers expect.

The layout matches ``include/efimon/gpu/nvidia-gpu/kernel-journal.hpp``: an
8-byte magic, a uint32 version and an int32 PID, followed by name chunks
(``N``, uint32 id, uint32 length, bytes) and run chunks (``R``, uint32 count,
then 36-byte records). All integers are little-endian.
"""

from __future__ import annotations

import csv
import dataclasses
import pathlib
import struct

import numpy as np

MAGIC = b"NVPMKJ01"
FORMAT_VERSION = 1
RECORD_DTYPE = np.dtype(
    [
        ("start_ns", "<u8"),
        ("end_ns", "<u8"),
        ("device_id", "<u4"),
        ("context_id", "<u4"),
        ("stream_id", "<u4"),
        ("correlation_id", "<u4"),
        ("name_id", "<u4"),
    ]
)
KERNEL_CSV_COLUMNS = (
    "event_id",
    "pid",
    "device_id",
    "context_id",
    "stream_id",
    "correlation_id",
    "start_ns",
    "end_ns",
    "duration_ns",
    "kernel_name",
)

_HEADER = struct.Struct("<8sIi")
_NAME_HEADER = struct.Struct("<II")
_RUN_HEADER = struct.Struct("<I")


@dataclasses.dataclass(frozen=True)
class KernelJournal:
    """Contents of a kernel journal.

    Attributes:
        pid: Process ID of the profiled process.
        names: Kernel names by name id.
        records: Kernel records in file order, with fields of RECORD_DTYPE.
        truncated: Whether the file ends inside a chunk, as after a crash.
    """

    pid: int
    names: dict[int, str]
    records: np.ndarray
    truncated: bool


def read_kernel_journal(path: pathlib.Path) -> KernelJournal:
    """Reads every complete chunk of a kernel journal.

    Args:
        path: Journal file written by the preload collector.

    Returns:
        The journal contents. A chunk cut short by a crash is ignored and
        reported through ``truncated``.

    Raises:
        ValueError: If the file is not a kernel journal of a known version, or
            if a chunk has an unknown tag.
    """
    data = path.read_bytes()
    if len(data) < _HEADER.size:
        raise ValueError(f"{path} is too short to be a kernel journal")
    magic, version, pid = _HEADER.unpack_from(data)
    if magic != MAGIC:
        raise ValueError(f"{path} is not a kernel journal")
    if version != FORMAT_VERSION:
        raise ValueError(f"{path} has unsupported journal version {version}")

    names = {}
    runs = []
    offset = _HEADER.size
    truncated = False
    while offset < len(data):
        tag = data[offset : offset + 1]
        body = offset + 1
        if tag == b"N":
            if body + _NAME_HEADER.size > len(data):
                truncated = True
                break
            name_id, length = _NAME_HEADER.unpack_from(data, body)
            name_start = body + _NAME_HEADER.size
            if name_start + length > len(data):
                truncated = True
                break
            names[name_id] = data[name_start : name_start + length].decode(
                "utf-8", errors="replace"
            )
            offset = name_start + length
        elif tag == b"R":
            if body + _RUN_HEADER.size > len(data):
                truncated = True
                break
            (count,) = _RUN_HEADER.unpack_from(data, body)
            records_start = body + _RUN_HEADER.size
            records_end = records_start + count * RECORD_DTYPE.itemsize
            if records_end > len(data):
                truncated = True
                break
            runs.append(
                np.frombuffer(
                    data, dtype=RECORD_DTYPE, count=count, offset=records_start
                )
            )
            offset = records_end
        else:
            raise ValueError(f"{path} has an unknown chunk tag at {offset}")

    records = np.concatenate(runs) if runs else np.empty(0, dtype=RECORD_DTYPE)
    return KernelJournal(
        pid=pid, names=names, records=records, truncated=truncated
    )


def write_kernel_csv(journal: KernelJournal, output_path: pathlib.Path) -> int:
    """Writes the journal as a kernel_activity.csv sorted by start time.

    Rows with the same start are ordered by correlation id, as the collector
    orders them, and kernel names are
    quoted as the collector quotes them.

    Args:
        journal: Contents returned by read_kernel_journal.
        output_path: CSV file to create or replace.

    Returns:
        The number of kernel rows written.
    """
    order = np.lexsort(
        (journal.records["correlation_id"], journal.records["start_ns"])
    )
    records = journal.records[order]
    with output_path.open("w", newline="", encoding="utf-8") as output:
        # The collector leaves the header unquoted and quotes every name.
        output.write(",".join(KERNEL_CSV_COLUMNS) + "\n")
        writer = csv.writer(
            output, quoting=csv.QUOTE_NONNUMERIC, lineterminator="\n"
        )
        for event_id, record in enumerate(records):
            start_ns = int(record["start_ns"])
            end_ns = int(record["end_ns"])
            writer.writerow(
                (
                    event_id,
                    journal.pid,
                    int(record["device_id"]),
                    int(record["context_id"]),
                    int(record["stream_id"]),
                    int(record["correlation_id"]),
                    start_ns,
                    end_ns,
                    end_ns - start_ns,
                    journal.names.get(int(record["name_id"]), "unknown"),
                )
            )
    return len(records)
