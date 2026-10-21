"""Vectorized aggregation of kernel activity and NVML energy into time windows."""

import numpy as np

from energy_attribution.io import NS_PER_SECOND


OTHER_FAMILY = "__other__"
EVENT_CHUNK = 1_000_000


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


def window_count(analysis_start, analysis_end, window_ns):
    return -(-(analysis_end - analysis_start) // window_ns)


def window_bounds(analysis_start, analysis_end, window_ns):
    starts = analysis_start + window_ns * np.arange(
        window_count(analysis_start, analysis_end, window_ns), dtype=np.int64
    )
    return starts, np.minimum(starts + window_ns, analysis_end)


def split_by_windows(starts, ends, analysis_start, analysis_end, window_ns):
    """Clip intervals to the analysis range and cut them at window edges.

    Returns ``(owner, window, overlap_ns)`` with one entry per interval/window
    piece, ordered by interval and then by window. ``owner`` indexes into the
    input arrays. Intervals entirely outside the range produce no pieces.
    """
    clipped_start = np.maximum(starts, analysis_start)
    clipped_end = np.minimum(ends, analysis_end)
    inside = np.flatnonzero(clipped_end > clipped_start)
    clipped_start = clipped_start[inside]
    clipped_end = clipped_end[inside]
    first = (clipped_start - analysis_start) // window_ns
    last = (clipped_end - 1 - analysis_start) // window_ns
    spans = last - first + 1
    if np.all(spans == 1):
        window_start = analysis_start + first * window_ns
        window_end = np.minimum(window_start + window_ns, analysis_end)
        overlap = (
            np.minimum(clipped_end, window_end)
            - np.maximum(clipped_start, window_start)
        )
        return inside, first, overlap

    piece_offsets = np.arange(int(spans.sum())) - np.repeat(
        np.cumsum(spans) - spans, spans
    )
    window = np.repeat(first, spans) + piece_offsets
    window_start = analysis_start + window * window_ns
    window_end = np.minimum(window_start + window_ns, analysis_end)
    overlap = (
        np.minimum(np.repeat(clipped_end, spans), window_end)
        - np.maximum(np.repeat(clipped_start, spans), window_start)
    )
    return np.repeat(inside, spans), window, overlap


def uncovered_pieces(starts, ends, covered_until):
    """Parts of start-sorted intervals not covered by earlier intervals.

    ``covered_until`` is the running maximum end of all preceding intervals.
    The returned pieces are disjoint and their union equals the union of the
    inputs, so summing their window overlaps gives kernel-union busy time.
    Returns ``(piece_start, piece_end, new_covered_until)``.
    """
    running_end = np.maximum.accumulate(ends)
    previous_end = np.empty_like(running_end)
    previous_end[0] = covered_until
    previous_end[1:] = np.maximum(running_end[:-1], covered_until)
    extends = ends > previous_end
    return (
        np.maximum(starts[extends], previous_end[extends]),
        ends[extends],
        max(covered_until, int(running_end[-1])),
    )


def aggregate_family_activity(
    table, columns, family_count, analysis_start, analysis_end, window_ns
):
    """Return per-window family busy time and kernel-union busy time (ns).

    Events are processed in chunks to bound memory; all sums are of integer
    nanoseconds, so results are exact and independent of chunking.
    """
    windows = window_count(analysis_start, analysis_end, window_ns)
    cells = windows * family_count
    activity_ns = np.zeros(cells, dtype=np.float64)
    union_active_ns = np.zeros(windows, dtype=np.float64)
    covered_until = np.iinfo(np.int64).min
    for offset in range(0, len(table), EVENT_CHUNK):
        chunk = slice(offset, offset + EVENT_CHUNK)
        owner, window, overlap = split_by_windows(
            table.start_ns[chunk], table.end_ns[chunk],
            analysis_start, analysis_end, window_ns,
        )
        activity_ns += np.bincount(
            window * family_count + columns[chunk][owner],
            weights=overlap,
            minlength=cells,
        )

        clipped_start = np.maximum(table.start_ns[chunk], analysis_start)
        clipped_end = np.minimum(table.end_ns[chunk], analysis_end)
        inside = clipped_end > clipped_start
        if not inside.any():
            continue
        piece_start, piece_end, covered_until = uncovered_pieces(
            clipped_start[inside], clipped_end[inside], covered_until
        )
        _, window, overlap = split_by_windows(
            piece_start, piece_end, analysis_start, analysis_end, window_ns
        )
        union_active_ns += np.bincount(window, weights=overlap, minlength=windows)
    return activity_ns.reshape(windows, family_count), union_active_ns


def distribute_to_events(
    table, columns, activity_ns, window_family_values,
    analysis_start, analysis_end, window_ns,
):
    """Spread (window, family) quantities to events by temporal overlap.

    Each event receives ``value[w, family] * overlap / activity_ns[w, family]``
    summed over the windows it touches. Returns one array per input matrix.
    """
    outputs = [np.zeros(len(table), dtype=np.float64) for _ in window_family_values]
    for offset in range(0, len(table), EVENT_CHUNK):
        chunk = slice(offset, offset + EVENT_CHUNK)
        owner, window, overlap = split_by_windows(
            table.start_ns[chunk], table.end_ns[chunk],
            analysis_start, analysis_end, window_ns,
        )
        family = columns[chunk][owner]
        family_duration = activity_ns[window, family]
        usable = family_duration > 0.0
        owner, window, family = owner[usable], window[usable], family[usable]
        fraction = overlap[usable] / family_duration[usable]
        size = min(EVENT_CHUNK, len(table) - offset)
        for output, values in zip(outputs, window_family_values):
            output[chunk] = np.bincount(
                owner, weights=values[window, family] * fraction, minlength=size
            )
    return outputs


def integrate_windows(trace, window_starts, window_ends, idle_power_mw):
    """Trapezoidal device and dynamic energy (mJ) for every window.

    Uses the same per-segment arithmetic and summation order as the original
    O(windows x samples) scan, but visits only segments overlapping a window.
    """
    timestamps = trace.timestamps_ns.tolist()
    power = trace.power_mw.tolist()
    first_segment = np.maximum(
        np.searchsorted(trace.timestamps_ns, window_starts, side="right") - 1, 0
    ).tolist()
    end_segment = np.minimum(
        np.searchsorted(trace.timestamps_ns, window_ends, side="left"),
        len(timestamps) - 1,
    ).tolist()
    device = np.zeros(len(window_starts), dtype=np.float64)
    dynamic = np.zeros(len(window_starts), dtype=np.float64)
    for window, (start, end) in enumerate(
        zip(window_starts.tolist(), window_ends.tolist())
    ):
        device_energy = 0.0
        dynamic_energy = 0.0
        for segment in range(first_segment[window], end_segment[window]):
            left_t, right_t = timestamps[segment], timestamps[segment + 1]
            overlap_start = max(start, left_t)
            overlap_end = min(end, right_t)
            if overlap_start >= overlap_end:
                continue
            left_p, right_p = power[segment], power[segment + 1]
            span = right_t - left_t
            start_device = left_p + (overlap_start - left_t) / span * (right_p - left_p)
            end_device = left_p + (overlap_end - left_t) / span * (right_p - left_p)
            seconds = (overlap_end - overlap_start) / NS_PER_SECOND
            device_energy += 0.5 * (start_device + end_device) * seconds
            dynamic_energy += 0.5 * (
                max(0.0, start_device - idle_power_mw)
                + max(0.0, end_device - idle_power_mw)
            ) * seconds
        device[window] = device_energy
        dynamic[window] = dynamic_energy
    return device, dynamic


def build_windows(
    trace, activity_ns, union_active_ns, analysis_start, analysis_end,
    window_ns, idle_power_mw,
):
    """Return per-window diagnostic rows and the dynamic-power target (W)."""
    window_starts, window_ends = window_bounds(
        analysis_start, analysis_end, window_ns
    )
    device_mj, dynamic_mj = integrate_windows(
        trace, window_starts, window_ends, idle_power_mw
    )
    rows = []
    for index, (start_ns, end_ns) in enumerate(
        zip(window_starts.tolist(), window_ends.tolist())
    ):
        duration_ns = end_ns - start_ns
        active_fraction = min(1.0, union_active_ns[index] / duration_ns)
        assignable_energy_mj = dynamic_mj[index] * active_fraction
        rows.append(
            {
                "window_index": index,
                "start_ns": start_ns,
                "end_ns": end_ns,
                "duration_ns": duration_ns,
                "kernel_active_fraction": active_fraction,
                "device_energy_mj": float(device_mj[index]),
                "dynamic_energy_mj": float(dynamic_mj[index]),
                "assignable_energy_mj": float(assignable_energy_mj),
                "unassigned_energy_mj": float(
                    dynamic_mj[index] - assignable_energy_mj
                ),
            }
        )
    durations_s = (window_ends - window_starts) / NS_PER_SECOND
    return rows, dynamic_mj / durations_s / 1000.0
