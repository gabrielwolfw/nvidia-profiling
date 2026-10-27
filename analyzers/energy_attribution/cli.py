"""Build command-line parsers from the typed configuration dataclasses.

Every field of a config dataclass becomes one ``--kebab-case`` option with
the dataclass default, so the CLI and the library API cannot drift apart.
"""

import argparse
import logging
import sys
from dataclasses import MISSING, fields
from typing import TypeVar

from energy_attribution.config import ATTRIBUTION_MODES

ConfigT = TypeVar("ConfigT")

HELP = {
    "telemetry": "NVML telemetry CSV (gpu_telemetry.csv).",
    "kernels": "CUPTI kernel activity CSV (kernel_activity.csv).",
    "sessions_root": "Directory whose subdirectories are profiling sessions.",
    "train_sessions_root": "Directory with the training-GPU sessions.",
    "external_test_session": (
        "Directory containing gpu_telemetry.csv and kernel_activity.csv."
    ),
    "output_dir": "Directory for CSV, JSON and plot outputs.",
    "max_families": (
        "Maximum family columns, including __other__; rarer kernels are "
        "pooled into __other__."
    ),
    "attribution": (
        "'model' (default) keeps the intercept's share of each window as "
        "baseline energy; 'active-fraction' reproduces the legacy rule."
    ),
    "min_windows_per_parameter": "Minimum training windows per regression "
    "parameter.",
    "allow_underdetermined": (
        "Fit even when training windows per parameter is below the minimum."
    ),
    "min_window_fraction": (
        "Windows shorter than this fraction of --window-ms (the truncated "
        "last window of a session) are excluded from fitting, metrics and "
        "plots; their energy is still attributed. 0 keeps every window."
    ),
    "calibration_fraction": (
        "Initial fraction of the external session used only to estimate a "
        "non-negative offset and scale."
    ),
    "source_label": "Name of the training GPU in console output and plots.",
    "target_label": "Name of the external GPU in console output and plots.",
    "plot": "Generate diagnostic PNG plots (requires matplotlib).",
    "min_telemetry_samples": (
        "Windows with fewer telemetry samples are dropped."
    ),
    "prior_scale": "Prior standard deviation of standardized coefficients.",
    "platform_label": "Name of the profiled GPU in plot titles.",
    "skip_event_output": (
        "Skip the potentially large per-event CSV while retaining summaries."
    ),
}


def add_config_arguments(parser, config_type):
    """Add one option per dataclass field, using the field's default."""
    for field in fields(config_type):
        option = "--" + field.name.replace("_", "-")
        options = {"help": HELP.get(field.name)}
        if field.type is bool:
            parser.add_argument(option, action="store_true", **options)
            continue
        options["type"] = field.type
        if field.default is MISSING:
            options["required"] = True
        else:
            options["default"] = field.default
        if field.name == "attribution":
            options["choices"] = ATTRIBUTION_MODES
        parser.add_argument(option, **options)


def config_from_namespace(config_type: type[ConfigT], namespace) -> ConfigT:
    """Construct a config from an argparse namespace, field by field.

    Raises ``ValueError`` when the namespace lacks a field or carries an
    option the config does not define, instead of silently using a default.
    """
    names = {field.name for field in fields(config_type)}
    provided = set(vars(namespace))
    if names != provided:
        raise ValueError(
            f"{config_type.__name__} fields and CLI options differ: "
            f"missing {sorted(names - provided)}, unexpected "
            f"{sorted(provided - names)}"
        )
    return config_type(**{name: getattr(namespace, name) for name in names})


class _ConsoleFormatter(logging.Formatter):
    """Plain progress messages; warnings keep their historical prefix."""

    def format(self, record):
        message = record.getMessage()
        if record.levelno >= logging.WARNING:
            return f"{record.levelname}: {message}"
        return message


def configure_logging(level=logging.INFO):
    """Send package log records to stdout as the commands always printed."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_ConsoleFormatter())
    logger = logging.getLogger("energy_attribution")
    logger.handlers[:] = [handler]
    logger.setLevel(level)
    logger.propagate = False


def run_command(config_type, description, analysis, report, argv=None):
    """Parse ``argv``, run ``analysis`` and print ``report``.

    Configuration and input errors exit with status 1 and a one-line message.
    """
    parser = argparse.ArgumentParser(description=description)
    add_config_arguments(parser, config_type)
    namespace = parser.parse_args(argv)
    configure_logging()
    try:
        config = config_from_namespace(config_type, namespace)
        result = analysis(config)
    except ValueError as error:
        parser.exit(1, f"error: {error}\n")
    report(result)
    return result


__all__ = [
    "add_config_arguments",
    "config_from_namespace",
    "configure_logging",
    "run_command",
]
