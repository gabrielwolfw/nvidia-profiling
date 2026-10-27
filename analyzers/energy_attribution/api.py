"""Public library API for Bayesian GPU energy attribution.

``analyze_*`` functions only read their inputs and return a typed result;
``run_*`` functions additionally write the CSV/JSON/PNG outputs to
``config.output_dir`` exactly as the command-line tools do.
"""

from energy_attribution.config import (
    CrossGpuConfig,
    MultiSessionConfig,
    SingleSessionConfig,
    TelemetryMultiSessionConfig,
    TelemetrySessionConfig,
)
from energy_attribution.reporting.outputs import (
    write_cross_gpu,
    write_multisession,
    write_single_session,
    write_telemetry_multisession,
    write_telemetry_session,
)
from energy_attribution.workflows import (
    cross_gpu,
    multisession,
    single_session,
    telemetry_multisession,
    telemetry_session,
)


def analyze_single_session(config: SingleSessionConfig) -> single_session.SingleSessionResult:
    """Fit one session and attribute its energy; writes nothing."""
    config.validate()
    return single_session.analyze(config)


def analyze_multisession(config: MultiSessionConfig) -> multisession.MultiSessionResult:
    """Fit one pooled model across sessions; writes nothing."""
    config.validate()
    return multisession.analyze(config)


def analyze_cross_gpu(config: CrossGpuConfig) -> cross_gpu.CrossGpuResult:
    """Train, calibrate and evaluate a cross-GPU transfer; writes nothing."""
    config.validate()
    return cross_gpu.analyze(config)


def run_single_session(config: SingleSessionConfig) -> single_session.SingleSessionResult:
    """Analyze one session and write its outputs."""
    result = analyze_single_session(config)
    write_single_session(result)
    return result


def run_multisession(config: MultiSessionConfig) -> multisession.MultiSessionResult:
    """Analyze several sessions with one pooled model and write its outputs."""
    result = analyze_multisession(config)
    write_multisession(result)
    return result


def run_cross_gpu(config: CrossGpuConfig) -> cross_gpu.CrossGpuResult:
    """Analyze a cross-GPU transfer and write its outputs."""
    result = analyze_cross_gpu(config)
    write_cross_gpu(result)
    return result


def analyze_telemetry_session(
    config: TelemetrySessionConfig,
) -> telemetry_session.TelemetrySessionResult:
    """Fit the telemetry-feature power model to one session; writes nothing."""
    config.validate()
    return telemetry_session.analyze(config)


def analyze_telemetry_multisession(
    config: TelemetryMultiSessionConfig,
) -> telemetry_multisession.TelemetryMultiSessionResult:
    """Fit one telemetry-feature energy model across sessions; writes nothing."""
    config.validate()
    return telemetry_multisession.analyze(config)


def run_telemetry_session(
    config: TelemetrySessionConfig,
) -> telemetry_session.TelemetrySessionResult:
    """Analyze one session with the telemetry model and write its outputs."""
    result = analyze_telemetry_session(config)
    write_telemetry_session(result)
    return result


def run_telemetry_multisession(
    config: TelemetryMultiSessionConfig,
) -> telemetry_multisession.TelemetryMultiSessionResult:
    """Analyze several sessions with the telemetry model and write outputs."""
    result = analyze_telemetry_multisession(config)
    write_telemetry_multisession(result)
    return result
