#!/usr/bin/env python3
"""Pooled Bayesian kernel-energy attribution across profiling sessions."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from analyze_bayesian_kernel_energy import (
    add_model_check_arguments,
    apply_allocation_to_rows,
    attribution_pool_mj,
    coefficient_summary_rows,
    create_plots,
    energy_accounting,
    fit_nonnegative_bayesian_regression,
    posterior_energy_allocation,
    regression_metrics,
    scan_kernel_families,
    session_holdout_energy,
    short_kernel_name,
    validate_common_args,
)
from energy_attribution.diagnostics import (
    check_identifiability,
    print_identifiability_warnings,
)
from energy_attribution.io import load_kernels, load_power_trace, write_csv
from energy_attribution.windows import (
    OTHER_FAMILY,
    aggregate_family_activity,
    build_windows,
    family_columns,
    select_families,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fit one non-negative Bayesian kernel-family model across all "
            "compatible sessions below a results directory."
        )
    )
    parser.add_argument("--sessions-root", required=True, type=Path)
    parser.add_argument("--window-ms", type=float, default=200.0)
    parser.add_argument("--idle-power-mw", type=float, default=0.0)
    parser.add_argument("--max-families", type=int, default=64)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--prior-scale-w", type=float, default=250.0)
    parser.add_argument("--noise-prior-scale-w", type=float, default=20.0)
    parser.add_argument("--burn-in", type=int, default=1000)
    parser.add_argument("--posterior-draws", type=int, default=2000)
    parser.add_argument("--thin", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    add_model_check_arguments(parser)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for pooled model outputs and plots.",
    )
    return parser.parse_args()


def validate_args(args):
    validate_common_args(args)
    if args.max_families < 2:
        raise ValueError("--max-families must be at least 2")
    if not 0.5 <= args.train_fraction < 1.0:
        raise ValueError("--train-fraction must be in [0.5, 1.0)")


def discover_session_paths(root):
    sessions = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        telemetry = directory / "gpu_telemetry.csv"
        kernels = directory / "kernel_activity.csv"
        if telemetry.is_file() and kernels.is_file():
            sessions.append((directory.name, telemetry, kernels))
    if not sessions:
        raise ValueError("No complete profiling sessions were found")
    return sessions


def scan_sessions(session_paths, max_families):
    active_sessions = []
    excluded_sessions = []
    duration_by_name = Counter()
    event_count_by_name = Counter()
    for name, telemetry, kernels in session_paths:
        try:
            print(f"Loading kernel events for {name}...", flush=True)
            metadata = scan_kernel_families(load_kernels(kernels), max_families=2)
        except ValueError as error:
            excluded_sessions.append({"session": name, "reason": str(error)})
            print(f"Skipping {name}: {error}")
            continue
        duration_by_name.update(metadata["duration_by_name"])
        event_count_by_name.update(metadata["event_count_by_name"])
        active_sessions.append((name, telemetry, kernels, metadata))

    if not active_sessions:
        raise ValueError("No session contains valid CUDA kernel events")
    families, selected = select_families(duration_by_name, max_families)
    return {
        "sessions": active_sessions,
        "excluded_sessions": excluded_sessions,
        "families": families,
        "selected": selected,
        "duration_by_name": duration_by_name,
        "event_count_by_name": event_count_by_name,
    }


def build_session_windows(
    name, telemetry_path, metadata, selected, family_index,
    window_ns, idle_power_mw,
):
    """Aggregate one session; releases its KernelTable afterwards."""
    print(f"\n=== Aggregating {name} ===", flush=True)
    trace = load_power_trace(telemetry_path)
    analysis_start = max(trace.first_ns, metadata["first_start"])
    analysis_end = min(trace.last_ns, metadata["last_end"])
    if analysis_end <= analysis_start:
        raise ValueError(f"Telemetry and kernels do not overlap in {name}")
    table = metadata.pop("table")
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
        trace, activity_ns, union_active_ns, analysis_start, analysis_end,
        window_ns, idle_power_mw,
    )
    for index, row in enumerate(rows):
        row["session"] = name
        row["session_window_index"] = index
    return {
        "name": name,
        "rows": rows,
        "activity_ns": activity_ns,
        "target_power_w": target_power_w,
    }


def aggregate_session(
    name, telemetry_path, metadata, selected, family_index,
    window_ns, idle_power_mw, train_fraction,
):
    result = build_session_windows(
        name, telemetry_path, metadata, selected, family_index,
        window_ns, idle_power_mw,
    )
    rows = result["rows"]
    split = int(len(rows) * train_fraction)
    split = min(max(split, 5), len(rows) - 2)
    for index, row in enumerate(rows):
        row["split"] = "train" if index < split else "test"
    result["split"] = split
    return result


def combine_train_test(session_data):
    rows = []
    activity_parts = []
    target_parts = []
    for split_name in ("train", "test"):
        for session in session_data:
            split = session["split"]
            selection = slice(0, split) if split_name == "train" else slice(split, None)
            rows.extend(session["rows"][selection])
            activity_parts.append(session["activity_ns"][selection])
            target_parts.append(session["target_power_w"][selection])
    train_count = sum(session["split"] for session in session_data)
    return (
        rows,
        np.vstack(activity_parts),
        np.concatenate(target_parts),
        train_count,
    )


def build_family_rows(scan, family_index, family_total_draws):
    rows = []
    for family, column in family_index.items():
        values = family_total_draws[:, column]
        lower, upper = np.quantile(values, [0.025, 0.975])
        if family == OTHER_FAMILY:
            source_names = scan["duration_by_name"].keys() - scan["selected"]
            events = sum(scan["event_count_by_name"][name] for name in source_names)
            duration = sum(scan["duration_by_name"][name] for name in source_names)
        else:
            events = scan["event_count_by_name"][family]
            duration = scan["duration_by_name"][family]
        rows.append(
            {
                "kernel_family": family,
                "event_count": events,
                "total_duration_ns": duration,
                "energy_posterior_mean_mj": float(values.mean()),
                "energy_posterior_median_mj": float(np.median(values)),
                "energy_posterior_lower_95_mj": float(lower),
                "energy_posterior_upper_95_mj": float(upper),
            }
        )
    rows.sort(key=lambda row: row["energy_posterior_mean_mj"], reverse=True)
    return rows


def main():
    args = parse_args()
    validate_args(args)
    window_ns = int(args.window_ms * 1_000_000)
    scan = scan_sessions(
        discover_session_paths(args.sessions_root), args.max_families
    )
    families = scan["families"]
    family_index = {family: index for index, family in enumerate(families)}
    session_data = [
        aggregate_session(
            name, telemetry, metadata, scan["selected"], family_index,
            window_ns, args.idle_power_mw, args.train_fraction,
        )
        for name, telemetry, _, metadata in scan["sessions"]
    ]
    rows, activity_ns, target_power_w, train_count = combine_train_test(
        session_data
    )
    design = np.column_stack(
        [np.ones(len(activity_ns)), activity_ns / window_ns]
    )
    terms = ["unattributed_baseline"] + families
    identifiability = check_identifiability(
        design, train_count, terms,
        args.min_windows_per_parameter, args.allow_underdetermined,
    )
    print(
        f"\nSampling pooled posterior with {train_count} training windows...",
        flush=True,
    )
    coefficient_draws, noise_draws, prediction_draws, column_scale = (
        fit_nonnegative_bayesian_regression(
            design, target_power_w, train_count, args.prior_scale_w,
            args.noise_prior_scale_w, args.burn_in, args.posterior_draws,
            args.thin, args.seed,
        )
    )
    predicted_mean = prediction_draws.mean(axis=0)
    predicted_lower, predicted_upper = np.quantile(
        prediction_draws, [0.025, 0.975], axis=0
    )
    allocation = posterior_energy_allocation(
        coefficient_draws,
        activity_ns,
        attribution_pool_mj(rows, args.attribution),
        window_ns,
        args.attribution,
    )
    for index, row in enumerate(rows):
        row.update(
            {
                "predicted_dynamic_power_mean_w": predicted_mean[index],
                "predicted_dynamic_power_lower_95_w": predicted_lower[index],
                "predicted_dynamic_power_upper_95_w": predicted_upper[index],
            }
        )
    apply_allocation_to_rows(rows, allocation)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "bayesian_energy_windows.csv", rows, list(rows[0])
    )
    coefficient_rows = coefficient_summary_rows(
        terms, coefficient_draws, column_scale
    )
    write_csv(
        args.output_dir / "bayesian_kernel_coefficients.csv",
        coefficient_rows, list(coefficient_rows[0]),
    )
    family_rows = build_family_rows(
        scan, family_index, allocation["family_total_draws"]
    )
    write_csv(
        args.output_dir / "bayesian_kernel_family_energy.csv",
        family_rows, list(family_rows[0]),
    )
    train_metrics = regression_metrics(
        target_power_w[:train_count], predicted_mean[:train_count]
    )
    test_metrics = regression_metrics(
        target_power_w[train_count:], predicted_mean[train_count:]
    )
    session_rows, session_metrics = session_holdout_energy(
        rows, prediction_draws
    )
    write_csv(
        args.output_dir / "bayesian_session_energy.csv",
        session_rows, list(session_rows[0]),
    )
    if args.plot:
        create_plots(
            args.output_dir, rows, family_rows,
            session_rows, session_metrics, test_metrics,
        )

    accounting = energy_accounting(rows, allocation)
    summary = {
        "model": "pooled_nonnegative_bayesian_kernel_family_regression",
        "attribution_mode": args.attribution,
        "validation": "final temporal fraction held out independently per session",
        "sessions_root": str(args.sessions_root),
        "included_sessions": [session["name"] for session in session_data],
        "excluded_sessions": scan["excluded_sessions"],
        "window_ms": args.window_ms,
        "train_windows": train_count,
        "test_windows": len(rows) - train_count,
        "kernel_events": int(sum(scan["event_count_by_name"].values())),
        "kernel_families": len(families),
        "design_matrix_rank": identifiability["design_matrix_rank"],
        "design_parameters": int(design.shape[1]),
        "identifiability": identifiability,
        "posterior_draws": args.posterior_draws,
        "posterior_noise_mean_w": float(np.sqrt(noise_draws).mean()),
        "train_power_metrics": train_metrics,
        "test_power_metrics": test_metrics,
        "test_session_energy_metrics": session_metrics,
        **accounting,
        "important_limitation": (
            "Attribution is probabilistic; concurrent kernel families may not "
            "be statistically identifiable."
        ),
    }
    with (args.output_dir / "bayesian_kernel_energy_summary.json").open(
        "w", encoding="utf-8"
    ) as destination:
        json.dump(summary, destination, indent=2)
        destination.write("\n")

    print("\n=== Multi-session Bayesian kernel-energy attribution ===")
    print(f"Included sessions: {len(session_data)}")
    print(f"Excluded sessions: {len(scan['excluded_sessions'])}")
    print(f"Kernel families: {len(families)}")
    print(f"Test power RMSE: {test_metrics['rmse_w']:.3f} W")
    if test_metrics["r_squared"] is not None:
        print(f"Test power R-squared: {test_metrics['r_squared']:.4f}")
    print(f"Baseline (b0) energy: {accounting['baseline_posterior_mean_mj']:.3f} mJ")
    print(f"Conservation error: {accounting['conservation_error_mj']:.6g} mJ")
    print_identifiability_warnings(identifiability, short_kernel_name)
    print(f"Results: {args.output_dir}")


if __name__ == "__main__":
    main()
