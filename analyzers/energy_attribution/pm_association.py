"""Time-weighted association of CUPTI PM samples with kernel intervals.

PM samples and kernels are both swept in start order; each sample adds its
metric values to every overlapping kernel, weighted by the overlap length.
"""

import csv
from dataclasses import dataclass, field
import math
import statistics

# PM CSV columns that describe the sample rather than a metric.
PM_METADATA_COLUMNS = {"sample", "start_ns", "end_ns", "duration_ns"}
COMPLETE_COVERAGE_PCT = 99.9


def read_csv_header(path):
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.reader(source)
        return next(reader, [])


def load_kernels(path):
    """Returns valid kernel rows sorted by time, the header and skip count."""
    kernels = []
    skipped = 0
    fieldnames = read_csv_header(path)
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            interval = _interval(row)
            if interval is None:
                skipped += 1
                continue
            row["start_ns"], row["end_ns"] = interval
            kernels.append(row)
    kernels.sort(key=lambda row: (row["start_ns"], row["end_ns"]))
    return kernels, fieldnames, skipped


def pm_metric_columns(path):
    """Validates the PM CSV header and returns its metric columns."""
    pm_fields = read_csv_header(path)
    missing_fields = {"start_ns", "end_ns"} - set(pm_fields)
    if missing_fields:
        raise ValueError(f"PM CSV is missing columns: {sorted(missing_fields)}")
    pm_metrics = [
        field_name
        for field_name in pm_fields
        if field_name not in PM_METADATA_COLUMNS
    ]
    if not pm_metrics:
        raise ValueError("PM CSV contains no metric columns")
    return pm_metrics


def read_pm_duration_stats(path):
    durations = []
    raw_samples = 0
    invalid_samples = 0
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            raw_samples += 1
            interval = _interval(row)
            if interval is None:
                invalid_samples += 1
                continue
            durations.append(interval[1] - interval[0])
    if not durations:
        raise ValueError("No valid PM sample intervals were found")
    return raw_samples, invalid_samples, statistics.median(durations)


def _interval(row):
    try:
        start_ns = int(row["start_ns"])
        end_ns = int(row["end_ns"])
    except (KeyError, TypeError, ValueError):
        return None
    if end_ns <= start_ns:
        return None
    return start_ns, end_ns


@dataclass
class SampleStats:
    """PM sample counts and gaps observed during association."""

    valid: int = 0
    rejected_duration: int = 0
    rejected_values: int = 0
    gap_events: list = field(default_factory=list)


def _new_accumulator(pm_metrics):
    return {
        "sample_count": 0,
        "covered_ns": 0,
        "last_covered_end_ns": 0,
        "weighted_duration_ns": 0,
        "metric_sums": {metric: 0.0 for metric in pm_metrics},
    }


def _metric_values(sample, pm_metrics):
    """Returns the sample's finite metric values, or None if any is invalid."""
    try:
        values = {metric: float(sample[metric]) for metric in pm_metrics}
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in values.values()):
        return None
    return values


def _track_gap(stats, previous_end_ns, sample_start, sample_end, max_gap_ns):
    """Records a gap before this sample and returns the new covered end."""
    if previous_end_ns is not None and sample_start > previous_end_ns:
        gap_duration_ns = sample_start - previous_end_ns
        if gap_duration_ns > max_gap_ns:
            stats.gap_events.append(
                (previous_end_ns, sample_start, gap_duration_ns)
            )
    if previous_end_ns is None:
        return sample_end
    return max(previous_end_ns, sample_end)


def _accumulate_overlap(accumulator, overlap_start, overlap_end, values):
    overlap_ns = overlap_end - overlap_start
    accumulator["sample_count"] += 1
    accumulator["weighted_duration_ns"] += overlap_ns
    for metric, value in values.items():
        accumulator["metric_sums"][metric] += value * overlap_ns
    uncovered_start = max(overlap_start, accumulator["last_covered_end_ns"])
    if overlap_end > uncovered_start:
        accumulator["covered_ns"] += overlap_end - uncovered_start
    accumulator["last_covered_end_ns"] = max(
        accumulator["last_covered_end_ns"], overlap_end
    )


def _add_sample(kernels, accumulators, first_index, interval, values):
    """Adds one sample to every kernel it overlaps, from ``first_index``."""
    sample_start, sample_end = interval
    candidate_index = first_index
    while (
        candidate_index < len(kernels)
        and kernels[candidate_index]["start_ns"] < sample_end
    ):
        kernel = kernels[candidate_index]
        overlap_start = max(sample_start, kernel["start_ns"])
        overlap_end = min(sample_end, kernel["end_ns"])
        if overlap_end - overlap_start > 0:
            _accumulate_overlap(
                accumulators[candidate_index],
                overlap_start,
                overlap_end,
                values,
            )
        candidate_index += 1


def associate_pm_samples(path, kernels, pm_metrics, duration_limits_ns):
    """Accumulates PM samples per kernel in one pass over the PM CSV.

    Args:
        path: PM Sampling CSV path.
        kernels: Kernel rows sorted by ``(start_ns, end_ns)``.
        pm_metrics: Metric columns to accumulate.
        duration_limits_ns: Accepted (minimum, maximum) sample duration.

    Returns:
        One accumulator per kernel and the sample statistics.
    """
    min_duration_ns, max_duration_ns = duration_limits_ns
    accumulators = [_new_accumulator(pm_metrics) for _ in kernels]
    stats = SampleStats()
    previous_valid_end_ns = None
    kernel_index = 0
    with path.open(newline="", encoding="utf-8") as source:
        for sample in csv.DictReader(source):
            try:
                interval = (int(sample["start_ns"]), int(sample["end_ns"]))
            except (KeyError, TypeError, ValueError):
                continue
            sample_start, sample_end = interval
            sample_duration = sample_end - sample_start
            if not min_duration_ns <= sample_duration <= max_duration_ns:
                stats.rejected_duration += 1
                continue
            previous_valid_end_ns = _track_gap(
                stats, previous_valid_end_ns, *interval, max_duration_ns
            )
            while (
                kernel_index < len(kernels)
                and kernels[kernel_index]["end_ns"] <= sample_start
            ):
                kernel_index += 1
            if (
                kernel_index == len(kernels)
                or kernels[kernel_index]["start_ns"] >= sample_end
            ):
                stats.valid += 1
                continue
            values = _metric_values(sample, pm_metrics)
            if values is None:
                stats.rejected_values += 1
                continue
            stats.valid += 1
            _add_sample(kernels, accumulators, kernel_index, interval, values)
    return accumulators, stats


def build_kernel_rows(kernels, accumulators, pm_metrics):
    """Returns output rows plus counts of covered and fully covered kernels."""
    output_rows = []
    covered_kernels = 0
    complete_coverage_kernels = 0
    for kernel, accumulator in zip(kernels, accumulators):
        duration_ns = kernel["end_ns"] - kernel["start_ns"]
        coverage_pct = 100.0 * accumulator["covered_ns"] / duration_ns
        row = dict(kernel)
        row["pm_sample_count"] = accumulator["sample_count"]
        row["pm_covered_ns"] = accumulator["covered_ns"]
        row["pm_coverage_pct"] = min(100.0, coverage_pct)
        weighted_duration_ns = accumulator["weighted_duration_ns"]
        if weighted_duration_ns > 0:
            covered_kernels += 1
            if coverage_pct >= COMPLETE_COVERAGE_PCT:
                complete_coverage_kernels += 1
            for metric in pm_metrics:
                row[metric] = (
                    accumulator["metric_sums"][metric] / weighted_duration_ns
                )
        else:
            for metric in pm_metrics:
                row[metric] = ""
        output_rows.append(row)
    return output_rows, covered_kernels, complete_coverage_kernels
