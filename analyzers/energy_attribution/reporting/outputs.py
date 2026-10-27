"""Write analysis results to CSV, JSON and PNG files.

File names and CSV columns are a stable contract checked by the golden
regression tests; the ``thor``/``v100`` transfer file names are historical.
"""

import json
import logging

import numpy as np
import pandas as pd

from energy_attribution.io import write_csv
from energy_attribution.reporting.plots import create_plots, create_transfer_plots
from energy_attribution.reporting.telemetry_plots import (
    create_multisession_plots,
    create_session_plots,
)
from energy_attribution.windows import EVENT_CHUNK

logger = logging.getLogger(__name__)


def write_rows(path, rows):
    """Write dictionaries as CSV, using the first row's keys as columns."""
    write_csv(path, rows, list(rows[0]))


def write_summary(path, summary):
    with path.open("w", encoding="utf-8") as destination:
        json.dump(summary, destination, indent=2)
        destination.write("\n")


def write_event_csv(path, events):
    """Per-event energy in chunks; join to the CUPTI trace via event_id."""
    family_names = np.asarray(events.families, dtype=object)
    for offset in range(0, len(events.start_ns), EVENT_CHUNK):
        chunk = slice(offset, offset + EVENT_CHUNK)
        frame = pd.DataFrame(
            {
                "start_ns": events.start_ns[chunk],
                "end_ns": events.end_ns[chunk],
                "kernel_family": family_names[events.family_column[chunk]],
                "bayesian_energy_mean_mj": events.mean[chunk],
                "bayesian_energy_lower_95_mj": events.lower[chunk],
                "bayesian_energy_upper_95_mj": events.upper[chunk],
            }
        )
        if events.event_ids is not None:
            frame.insert(0, "event_id", events.event_ids[chunk])
        frame.to_csv(
            path, mode="w" if offset == 0 else "a", header=offset == 0, index=False
        )


def _write_kernel_model(result, output_dir):
    """Outputs shared by the single- and multi-session kernel models."""
    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(output_dir / "bayesian_energy_windows.csv", result.dataset.rows)
    write_rows(output_dir / "bayesian_kernel_coefficients.csv", result.coefficient_rows)
    write_rows(output_dir / "bayesian_kernel_family_energy.csv", result.family_rows)
    write_rows(output_dir / "bayesian_session_energy.csv", result.session_rows)
    if result.config.plot:
        create_plots(
            output_dir, result.dataset.rows, result.family_rows,
            result.session_rows, result.session_metrics, result.test_metrics,
        )


def write_single_session(result):
    output_dir = result.config.output_dir
    _write_kernel_model(result, output_dir)
    if result.events is not None:
        logger.info("Writing per-event Bayesian attribution...")
        write_event_csv(output_dir / "bayesian_kernel_event_energy.csv", result.events)
    write_summary(output_dir / "bayesian_kernel_energy_summary.json", result.summary)


def write_multisession(result):
    output_dir = result.config.output_dir
    _write_kernel_model(result, output_dir)
    write_summary(output_dir / "bayesian_kernel_energy_summary.json", result.summary)


def write_cross_gpu(result):
    config = result.config
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(output_dir / "bayesian_transfer_windows.csv", result.test_rows)
    write_rows(output_dir / "thor_kernel_coefficients.csv", result.base_coefficient_rows)
    write_rows(output_dir / "v100_calibration_coefficients.csv", result.calibration_rows)
    write_rows(output_dir / "v100_kernel_family_energy.csv", result.family_rows)
    write_summary(output_dir / "bayesian_transfer_summary.json", result.summary)
    if config.plot:
        create_transfer_plots(
            output_dir,
            [row for row in result.test_rows if row["split"] != "excluded"],
            result.calibration_count,
            result.family_rows,
            result.metrics,
            (config.source_label, config.target_label),
        )


def write_telemetry_session(result):
    output_dir = result.config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(output_dir / "bayesian_window_predictions.csv", result.prediction_rows)
    write_rows(output_dir / "bayesian_coefficients.csv", result.coefficient_rows)
    write_summary(output_dir / "bayesian_model_summary.json", result.summary)
    if result.config.plot:
        create_session_plots(
            output_dir, result.prediction_rows, result.targets, result.predicted,
            result.predictive_lower, result.predictive_upper, result.train_count,
        )


def write_telemetry_multisession(result):
    output_dir = result.config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(
        output_dir / "bayesian_multisession_predictions.csv", result.prediction_rows
    )
    write_rows(
        output_dir / "bayesian_multisession_coefficients.csv", result.coefficient_rows
    )
    write_rows(output_dir / "bayesian_session_energy.csv", result.session_energy_rows)
    write_summary(output_dir / "bayesian_multisession_summary.json", result.summary)
    if result.config.plot:
        seconds = result.window_seconds
        create_multisession_plots(
            output_dir, result.rows, result.power_targets,
            result.predicted_energy / seconds, result.energy_lower / seconds,
            result.energy_upper / seconds, result.energy_targets,
            result.predicted_energy, result.energy_lower, result.energy_upper,
            result.train_count, result.session_energy_rows,
            result.config.platform_label,
        )
