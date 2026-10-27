"""Split measured NVML energy among concurrent kernels in one streaming sweep.

Kernels are read once, in start order, so captures with millions of events
fit in memory. Concurrent kernels share each energy segment by PM weight when
all of them have one, and equally otherwise.
"""

from array import array
from dataclasses import dataclass
import heapq
import math

from energy_attribution.streaming_io import iter_kernels
from energy_attribution.units import NS_PER_SECOND

PROGRESS_INTERVAL = 1_000_000


def interpolated_power(timestamp, left, right, idle_power_mw):
    left_t, left_p = left
    right_t, right_p = right
    fraction = (timestamp - left_t) / (right_t - left_t)
    device_power = left_p + fraction * (right_p - left_p)
    return device_power, max(0.0, device_power - idle_power_mw)


def interval_energy_mj(start, end, left, right, idle_power_mw):
    start_device, start_dynamic = interpolated_power(
        start, left, right, idle_power_mw
    )
    end_device, end_dynamic = interpolated_power(
        end, left, right, idle_power_mw
    )
    duration_seconds = (end - start) / NS_PER_SECOND
    return (
        0.5 * (start_device + end_device) * duration_seconds,
        0.5 * (start_dynamic + end_dynamic) * duration_seconds,
    )


def integrate_range(samples, start, end, idle_power_mw):
    device_energy = 0.0
    dynamic_energy = 0.0
    for left, right in zip(samples, samples[1:]):
        overlap_start = max(start, left[0])
        overlap_end = min(end, right[0])
        if overlap_start >= overlap_end:
            continue
        device, dynamic = interval_energy_mj(
            overlap_start, overlap_end, left, right, idle_power_mw
        )
        device_energy += device
        dynamic_energy += dynamic
    return device_energy, dynamic_energy


def window_total(start, end, window_ns):
    return (end - start + window_ns - 1) // window_ns


class WindowActivity:
    """Union of kernel-busy time inside each fixed-length analysis window."""

    def __init__(self, start, end, window_ns):
        self.start = start
        self.end = end
        self.window_ns = window_ns
        count = window_total(start, end, window_ns)
        self.active_ns = array("d", [0.0]) * count
        self._union_end = array("q", [-1]) * count

    def add_kernel(self, kernel_start, kernel_end):
        """Adds the part of a kernel interval not already counted as busy."""
        clipped_start = max(kernel_start, self.start)
        clipped_end = min(kernel_end, self.end)
        if clipped_start >= clipped_end:
            return
        first_window = (clipped_start - self.start) // self.window_ns
        last_window = (clipped_end - 1 - self.start) // self.window_ns
        for window_index in range(first_window, last_window + 1):
            window_start = self.start + window_index * self.window_ns
            window_end = min(window_start + self.window_ns, self.end)
            overlap_start = max(clipped_start, window_start)
            overlap_end = min(clipped_end, window_end)
            previous_end = self._union_end[window_index]
            if overlap_end > previous_end:
                uncovered_start = max(overlap_start, previous_end)
                self.active_ns[window_index] += overlap_end - uncovered_start
                self._union_end[window_index] = overlap_end


class KernelEnergyLedger:
    """Per-kernel energy shares and the totals accumulated by the sweep."""

    def __init__(self):
        self.assigned = 0.0
        self.unassigned = 0.0
        self.pm_weighted = 0.0
        self.fallback = 0.0
        self.estimated_energy = array("d")
        self.pm_weighted_energy = array("d")
        self.fallback_energy = array("d")

    def add_kernel(self, index):
        if index != len(self.estimated_energy):
            raise RuntimeError("Kernel stream index is not contiguous")
        self.estimated_energy.append(0.0)
        self.pm_weighted_energy.append(0.0)
        self.fallback_energy.append(0.0)

    def distribute(self, active, dynamic_energy):
        """Splits one segment's energy among the ``active`` kernel weights."""
        if not active:
            self.unassigned += dynamic_energy
            return
        use_pm_weights = all(weight is not None for weight in active.values())
        total_weight = sum(active.values()) if use_pm_weights else 0.0
        if total_weight > 0.0:
            for index, weight in active.items():
                share = dynamic_energy * weight / total_weight
                self.estimated_energy[index] += share
                self.pm_weighted_energy[index] += share
            self.pm_weighted += dynamic_energy
        else:
            share = dynamic_energy / len(active)
            for index in active:
                self.estimated_energy[index] += share
                self.fallback_energy[index] += share
            self.fallback += dynamic_energy
        self.assigned += dynamic_energy

    def results(self, window_active_ns):
        return {
            "assigned": self.assigned,
            "unassigned": self.unassigned,
            "pm_weighted": self.pm_weighted,
            "fallback": self.fallback,
            "estimated_energy": self.estimated_energy,
            "pm_weighted_energy": self.pm_weighted_energy,
            "fallback_energy": self.fallback_energy,
            "window_active_ns": window_active_ns,
        }


def _tracked_kernels(kernel_path, pm_by_event_id, ledger, activity):
    """Yields kernels in start order, registering each with the sweep."""
    for kernel in iter_kernels(kernel_path, pm_by_event_id):
        index, kernel_start, kernel_end, _ = kernel
        ledger.add_kernel(index)
        if (index + 1) % PROGRESS_INTERVAL == 0:
            print(
                f"Attributed timeline for {index + 1} kernel events...",
                flush=True,
            )
        activity.add_kernel(kernel_start, kernel_end)
        yield kernel


def _advance_active_set(cursor, next_kernel, kernels, active, end_heap):
    """Retires kernels ended by ``cursor`` and admits those started by it."""
    while end_heap and end_heap[0][0] <= cursor:
        _, index = heapq.heappop(end_heap)
        active.pop(index, None)
    while next_kernel is not None and next_kernel[1] <= cursor:
        index, _, kernel_end, weight = next_kernel
        if kernel_end > cursor:
            active[index] = weight
            heapq.heappush(end_heap, (kernel_end, index))
        next_kernel = next(kernels, None)
    return next_kernel


def _next_boundary(cursor, segment_end, next_kernel, end_heap):
    """Returns the next time at which the set of active kernels changes."""
    next_start = (
        next_kernel[1]
        if next_kernel is not None and next_kernel[1] < segment_end
        else segment_end
    )
    next_end = end_heap[0][0] if end_heap else segment_end
    boundary = min(segment_end, next_start, next_end)
    if boundary <= cursor:
        raise RuntimeError("Energy attribution sweep did not advance")
    return boundary


def attribute_energy_streaming(
    samples,
    kernel_path,
    pm_by_event_id,
    start,
    end,
    idle_power_mw,
    window_ns,
):
    """Attribute energy with a chronological sweep over kernel boundaries."""
    ledger = KernelEnergyLedger()
    activity = WindowActivity(start, end, window_ns)
    kernels = _tracked_kernels(kernel_path, pm_by_event_id, ledger, activity)
    next_kernel = next(kernels, None)
    active = {}
    end_heap = []

    for left, right in zip(samples, samples[1:]):
        segment_start = max(start, left[0])
        segment_end = min(end, right[0])
        cursor = segment_start
        while cursor < segment_end:
            next_kernel = _advance_active_set(
                cursor, next_kernel, kernels, active, end_heap
            )
            boundary = _next_boundary(
                cursor, segment_end, next_kernel, end_heap
            )
            _, dynamic_energy = interval_energy_mj(
                cursor, boundary, left, right, idle_power_mw
            )
            ledger.distribute(active, dynamic_energy)
            cursor = boundary

    return ledger.results(activity.active_ns)


def empty_attribution(dynamic_energy, start, end, window_ns):
    """Results for a capture without kernels: all energy is unassigned."""
    ledger = KernelEnergyLedger()
    ledger.unassigned = dynamic_energy
    return ledger.results(
        array("d", [0.0]) * window_total(start, end, window_ns)
    )


def pearson(values_x, values_y):
    finite_pairs = [
        (x, y)
        for x, y in zip(values_x, values_y)
        if math.isfinite(x) and math.isfinite(y)
    ]
    values_x = [pair[0] for pair in finite_pairs]
    values_y = [pair[1] for pair in finite_pairs]
    if len(values_x) < 2:
        return math.nan
    mean_x = sum(values_x) / len(values_x)
    mean_y = sum(values_y) / len(values_y)
    centered_x = [value - mean_x for value in values_x]
    centered_y = [value - mean_y for value in values_y]
    denominator = math.sqrt(
        sum(value * value for value in centered_x)
        * sum(value * value for value in centered_y)
    )
    if denominator == 0:
        return math.nan
    return sum(x * y for x, y in zip(centered_x, centered_y)) / denominator


def build_windows(
    samples, window_active_ns, start, end, idle_power_mw, window_ns
):
    windows = []
    window_start = start
    index = 0
    while window_start < end:
        window_end = min(window_start + window_ns, end)
        duration_ns = window_end - window_start
        active_ns = window_active_ns[index]
        device_energy, dynamic_energy = integrate_range(
            samples, window_start, window_end, idle_power_mw
        )
        duration_seconds = duration_ns / NS_PER_SECOND
        windows.append(
            {
                "window": index,
                "start_ns": window_start,
                "end_ns": window_end,
                "duration_ms": duration_ns / 1_000_000,
                "kernel_active_ms": active_ns / 1_000_000,
                "kernel_active_pct": 100.0 * active_ns / duration_ns,
                "device_energy_mj": device_energy,
                "dynamic_energy_mj": dynamic_energy,
                "avg_device_power_mw": device_energy / duration_seconds,
                "avg_dynamic_power_mw": dynamic_energy / duration_seconds,
            }
        )
        index += 1
        window_start = window_end
    return windows


@dataclass
class WeightedEnergyReport:
    """Energy totals, per-window activity and attribution for one capture."""

    analysis_start: int
    analysis_end: int
    device_energy: float
    dynamic_energy: float
    attribution: dict
    windows: list
    kernel_metadata: dict

    @property
    def conservation_error(self):
        return (
            self.dynamic_energy
            - self.attribution["assigned"]
            - self.attribution["unassigned"]
        )

    def correlations(self):
        """Pearson r of kernel activity against window energy and power."""
        utilization = [row["kernel_active_pct"] for row in self.windows]
        energy = [row["dynamic_energy_mj"] for row in self.windows]
        power = [row["avg_dynamic_power_mw"] for row in self.windows]
        return pearson(utilization, energy), pearson(utilization, power)
