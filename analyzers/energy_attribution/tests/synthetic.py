"""Deterministic synthetic NVML/CUPTI sessions for end-to-end regression tests.

Power follows a known kernel-family model, so the Bayesian scripts have a
real signal to fit while files stay small (~1-2k kernels per session).
"""

import csv

import numpy as np

BASE_TIMESTAMP_NS = 1_700_000_000_000_000_000
TELEMETRY_PERIOD_NS = 50_000_000
SEGMENT_NS = 1_000_000_000
FAMILY_POWER_MW = {
    "kernel_gemm": 90_000.0,
    "kernel_attention": 60_000.0,
    "kernel_norm": 25_000.0,
    "kernel_norm_fused": 25_000.0,
    "kernel_copy": 12_000.0,
}
# kernel_norm_fused always follows kernel_norm, making the pair collinear.
PRIMARY_FAMILIES = [
    "kernel_gemm",
    "kernel_attention",
    "kernel_norm",
    "kernel_copy",
]
RARE_NAMES = [f"kernel_rare_{index}" for index in range(6)]
RARE_POWER_MW = 40_000.0


def generate_kernels(rng, duration_ns, family_weights):
    """Sequential kernels on one stream plus a sparse overlapping stream."""
    events = []
    time_ns = BASE_TIMESTAMP_NS + 120_000_000
    end_ns = BASE_TIMESTAMP_NS + duration_ns - 30_000_000
    segment_duty = {}
    while time_ns < end_ns:
        segment = (time_ns - BASE_TIMESTAMP_NS) // SEGMENT_NS
        if segment not in segment_duty:
            segment_duty[segment] = rng.uniform(0.2, 0.95)
        weights = np.asarray(family_weights) * rng.uniform(
            0.5, 1.5, len(family_weights)
        )
        family = PRIMARY_FAMILIES[
            rng.choice(len(PRIMARY_FAMILIES), p=weights / weights.sum())
        ]
        if rng.random() < 0.03:
            family = RARE_NAMES[rng.integers(len(RARE_NAMES))]
        length = int(rng.integers(2_000_000, 12_000_000))
        events.append((time_ns, time_ns + length, family, 7))
        time_ns += length
        if family == "kernel_norm":
            fused = int(rng.integers(500_000, 2_000_000))
            events.append((time_ns, time_ns + fused, "kernel_norm_fused", 7))
            time_ns += fused
        duty = segment_duty[segment]
        time_ns += int(length * (1.0 - duty) / duty)
        if rng.random() < 0.08:
            start = time_ns - int(rng.integers(1_000_000, length))
            events.append(
                (
                    start,
                    start + int(rng.integers(1_000_000, 4_000_000)),
                    "kernel_copy",
                    9,
                )
            )
    events.sort(key=lambda event: event[0])
    return events


def busy_fraction_by_sample(events, sample_times):
    """Fraction of each [t - period, t) interval covered by every name."""
    names = list(FAMILY_POWER_MW) + RARE_NAMES
    starts = np.asarray([event[0] for event in events], dtype=np.int64)
    ends = np.asarray([event[1] for event in events], dtype=np.int64)
    labels = np.asarray([names.index(event[2]) for event in events])
    fractions = np.zeros((len(sample_times), len(names)))
    for index, sample_end in enumerate(sample_times):
        sample_start = sample_end - TELEMETRY_PERIOD_NS
        overlap = np.clip(
            np.minimum(ends, sample_end) - np.maximum(starts, sample_start),
            0,
            None,
        )
        fractions[index] = np.bincount(
            labels, weights=overlap, minlength=len(names)
        )
    return names, fractions / TELEMETRY_PERIOD_NS


def write_session(
    directory, seed, duration_ns, family_weights, scale=1.0, offset_mw=0.0
):
    """Write gpu_telemetry.csv and kernel_activity.csv into ``directory``."""
    rng = np.random.default_rng(seed)
    directory.mkdir(parents=True, exist_ok=True)
    events = generate_kernels(rng, duration_ns, family_weights)
    sample_times = BASE_TIMESTAMP_NS + TELEMETRY_PERIOD_NS * np.arange(
        1, duration_ns // TELEMETRY_PERIOD_NS + 1, dtype=np.int64
    )
    names, fractions = busy_fraction_by_sample(events, sample_times)
    coefficients = np.asarray(
        [FAMILY_POWER_MW.get(name, RARE_POWER_MW) for name in names]
    )
    power_mw = offset_mw + scale * (
        20_000.0
        + fractions @ coefficients
        + rng.normal(0.0, 600.0, len(sample_times))
    )

    # Derived NVML-style counters for the telemetry-feature model. They use no
    # random draws, so the kernel-model fixtures are unaffected by them.
    busy = np.clip(fractions.sum(axis=1), 0.0, 1.0)
    memory_busy = np.clip(
        fractions[:, names.index("kernel_copy")]
        + fractions[:, names.index("kernel_norm")],
        0.0,
        1.0,
    )
    gpu_utilization = np.round(100.0 * busy)
    memory_utilization = np.round(70.0 * memory_busy)
    graphics_clock = np.where(busy > 0.05, 1_530.0, 405.0)
    temperature = np.round(
        38.0 + 12.0 * np.cumsum(busy) / len(busy) + 6.0 * busy
    )
    with (directory / "gpu_telemetry.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "timestamp_ns",
                "power_mw",
                "temperature_c",
                "gpu_utilization_pct",
                "memory_utilization_pct",
                "graphics_clock_mhz",
            ]
        )
        for row in zip(
            sample_times.tolist(),
            power_mw.tolist(),
            temperature.tolist(),
            gpu_utilization.tolist(),
            memory_utilization.tolist(),
            graphics_clock.tolist(),
        ):
            timestamp, power, *counters = row
            writer.writerow(
                [
                    timestamp,
                    int(round(power)),
                    *(int(value) for value in counters),
                ]
            )
    with (directory / "kernel_activity.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "event_id",
                "stream_id",
                "start_ns",
                "end_ns",
                "duration_ns",
                "kernel_name",
            ]
        )
        for event_id, (start, end, name, stream) in enumerate(events):
            writer.writerow([event_id, stream, start, end, end - start, name])


TRAINING_SESSIONS = {
    "s1-compute": (11, 14_130_000_000, [6.0, 2.0, 1.0, 1.0]),
    "s2-memory": (12, 13_070_000_000, [1.0, 1.0, 2.0, 6.0]),
    "s3-mixed": (13, 15_910_000_000, [3.0, 3.0, 2.0, 2.0]),
}
EXTERNAL_SESSION = ("x1-external", 21, 9_460_000_000, [3.0, 2.0, 2.0, 3.0])


def write_fixture(root):
    """Create ``root/train/<session>`` and ``root/external`` session folders."""
    for name, (seed, duration_ns, weights) in TRAINING_SESSIONS.items():
        write_session(root / "train" / name, seed, duration_ns, weights)
    name, seed, duration_ns, weights = EXTERNAL_SESSION
    write_session(
        root / "external" / name,
        seed,
        duration_ns,
        weights,
        scale=2.5,
        offset_mw=15_000.0,
    )
    return root
