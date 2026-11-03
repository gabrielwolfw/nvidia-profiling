#!/usr/bin/env python3
"""Vision Transformer (ViT) inference benchmark for GPU energy profiling.

One process runs one precision, so each launcher capture maps to a single
configuration. Like ``bench_power_cycle``, the run is split into warm-up,
measurement and cooldown phases delimited by empty marker kernels, so the
measured phase can be located in ``kernel_activity.csv`` without relying on
host clocks. Model construction and ``torch.compile`` happen before the first
marker and are excluded from every phase.

Weights and inputs are random: inference cost depends on tensor shapes, not
values, and no download is needed during a capture.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

DTYPES = ("fp32", "fp16", "bf16", "fp8")
# Same kernel names as bench_power_cycle, so analyze_power_sweep.py helpers
# can find the phases of this benchmark too.
PHASE_MARKERS = (
    "benchmark_warmup_start_marker",
    "benchmark_measurement_start_marker",
    "benchmark_cooldown_start_marker",
    "benchmark_complete_marker",
)
PROFILER_LIBRARY = "libnvidia-process-metrics"
# torch.compile modes without CUDA graphs. CUDA graph replay crashed inside
# CUPTI on Thor (see README, llama.cpp section).
COMPILE_MODES = ("default", "max-autotune-no-cudagraphs")
# FP8 tensor-core GEMMs need compute capability 8.9 (Ada) or newer.
FP8_MIN_CAPABILITY = (8, 9)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="vit_base_patch16_224")
    parser.add_argument("--dtype", choices=DTYPES, default="fp16")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup-seconds", type=float, default=10.0)
    parser.add_argument(
        "--duration",
        type=float,
        default=40.0,
        help="Measurement phase length in seconds (default: 40).",
    )
    parser.add_argument("--cooldown-seconds", type=float, default=10.0)
    parser.add_argument(
        "--sync-every",
        type=int,
        default=5,
        help="Batches queued between synchronizations (default: 5).",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile; recommended for FP8 throughput.",
    )
    parser.add_argument(
        "--compile-mode", choices=COMPILE_MODES, default="default"
    )
    parser.add_argument(
        "--tf32",
        action="store_true",
        help="Allow TF32 tensor cores for fp32 (off by default).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help=(
            "Check the PyTorch/CUDA setup and every precision with a short "
            "run, without phases. Run it outside the launcher."
        ),
    )
    parser.add_argument(
        "--device-type",
        choices=("cuda", "cpu"),
        default="cuda",
        help="cpu only smoke-tests the script logic; it measures nothing.",
    )
    args = parser.parse_args(argv)
    validate_args(args)
    return args


def validate_args(args):
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    if args.sync_every < 1:
        raise SystemExit("--sync-every must be at least 1")
    if args.duration <= 0:
        raise SystemExit("--duration must be greater than zero")
    if args.warmup_seconds < 0 or args.cooldown_seconds < 0:
        raise SystemExit("phase lengths cannot be negative")


def strip_profiler_preload(environment):
    """Removes the profiler library from LD_PRELOAD for child processes.

    The launcher preloads the profiler into this process. torch.compile starts
    compiler subprocesses (gcc, ptxas, Inductor workers); if they inherited
    the preload, each would start its own profiler and overwrite this
    session's CSV files on exit. The library already loaded here is unaffected.

    Returns:
        True when an entry was removed.
    """
    preload = environment.get("LD_PRELOAD", "")
    entries = preload.replace(":", " ").split()
    kept = [entry for entry in entries if PROFILER_LIBRARY not in entry]
    if len(kept) == len(entries):
        return False
    if kept:
        environment["LD_PRELOAD"] = " ".join(kept)
    else:
        del environment["LD_PRELOAD"]
    return True


def emit(key, value):
    print(f"{key}={value}", flush=True)


class Benchmark:
    """Builds the model for one precision and runs timed inference phases."""

    def __init__(self, args, torch, device):
        self.args = args
        self.torch = torch
        self.device = device
        self.fp8_layers = 0
        self.markers = None
        self._marker_input = None

    def synchronize(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def build(self, dtype):
        """Returns (model, inputs, rel_error) for ``dtype``.

        ``rel_error`` is the relative L2 error of the logits against the same
        model and inputs in fp32.
        """
        import timm  # pylint: disable=import-outside-toplevel

        torch = self.torch
        torch.manual_seed(self.args.seed)
        model = timm.create_model(self.args.model, pretrained=False)
        model = model.eval().to(self.device)
        config = model.pretrained_cfg
        size = config.get("input_size", (3, 224, 224))
        inputs = torch.randn((self.args.batch_size, *size), device=self.device)
        with torch.inference_mode():
            reference = model(inputs).float()

        if dtype in ("fp16", "bf16", "fp8"):
            target = torch.float16 if dtype == "fp16" else torch.bfloat16
            model = model.to(target)
            inputs = inputs.to(target)
        if dtype == "fp8":
            self.fp8_layers = self.quantize_fp8(model)
        if self.args.compile:
            # pylint: disable-next=import-outside-toplevel
            import torch._inductor.config as inductor_config

            inductor_config.triton.cudagraphs = False
            model = torch.compile(model, mode=self.args.compile_mode)

        with torch.inference_mode():
            output = model(inputs).float()
        self.synchronize()
        rel_error = (
            torch.linalg.vector_norm(output - reference)
            / torch.linalg.vector_norm(reference)
        ).item()
        return model, inputs, rel_error

    def quantize_fp8(self, model):
        """Converts eligible Linear layers to dynamic float8 with torchao."""
        torch = self.torch
        if self.device.type == "cuda":
            capability = torch.cuda.get_device_capability(self.device)
            if capability < FP8_MIN_CAPABILITY:
                raise SystemExit(
                    f"fp8 needs compute capability >= 8.9, found {capability}"
                )
        # pylint: disable=import-outside-toplevel
        from torchao.quantization import quantize_

        try:
            from torchao.quantization import (
                Float8DynamicActivationFloat8WeightConfig,
            )

            config = Float8DynamicActivationFloat8WeightConfig()
        except ImportError:  # torchao < 0.9 API
            from torchao.quantization import (
                float8_dynamic_activation_float8_weight,
            )

            config = float8_dynamic_activation_float8_weight()
        # pylint: enable=import-outside-toplevel

        def eligible(module, _):
            # FP8 GEMMs need both dimensions divisible by 16; this leaves the
            # 1000-class head (and the patch-embedding conv) in bf16.
            return (
                isinstance(module, torch.nn.Linear)
                and module.in_features % 16 == 0
                and module.out_features % 16 == 0
            )

        layers = sum(1 for module in model.modules() if eligible(module, ""))
        quantize_(model, config, filter_fn=eligible)
        return layers

    def prepare_markers(self):
        """Compiles one empty marker kernel per phase boundary.

        jiterator kernels are named after the function, so CUPTI records a
        kernel whose name contains ``benchmark_*_marker``. Returns False when
        markers are unavailable (CPU, or a build without jiterator).
        """
        if self.device.type != "cuda":
            return False
        torch = self.torch
        try:
            self.markers = {
                name: torch.cuda.jiterator._create_jit_fn(  # pylint: disable=protected-access
                    f"template <typename T> T {name}(T x) {{ return x; }}"
                )
                for name in PHASE_MARKERS
            }
            self._marker_input = torch.zeros(1, device=self.device)
        except (AttributeError, RuntimeError) as error:
            print(f"WARNING: phase markers unavailable: {error}", flush=True)
            self.markers = None
        return self.markers is not None

    def mark(self, name):
        """Launches marker ``name`` once, after all queued work finishes."""
        self.synchronize()
        if self.markers is not None:
            self.markers[name](self._marker_input)
            self.synchronize()

    def run_phase(self, model, inputs, seconds):
        """Runs inference for ``seconds``; returns (elapsed, batches)."""
        batches = 0
        start = time.perf_counter()
        deadline = start + seconds
        with self.torch.inference_mode():
            while time.perf_counter() < deadline:
                for _ in range(self.args.sync_every):
                    model(inputs)
                batches += self.args.sync_every
                self.synchronize()
        return time.perf_counter() - start, batches


def print_environment(torch, device):
    emit("TORCH_VERSION", torch.__version__)
    emit("CUDA_RUNTIME", torch.version.cuda)
    if device.type == "cuda":
        major, minor = torch.cuda.get_device_capability(device)
        emit("DEVICE_NAME", torch.cuda.get_device_name(device))
        emit("COMPUTE_CAPABILITY", f"{major}.{minor}")
        arch_list = torch.cuda.get_arch_list()
        emit("TORCH_ARCH_LIST", ",".join(arch_list))
        # PTX for an older arch can still JIT-compile, so warn, don't fail.
        if f"sm_{major}{minor}" not in arch_list:
            print(
                f"WARNING: this PyTorch build has no sm_{major}{minor} binary;"
                " kernels will be JIT-compiled from PTX or fail",
                flush=True,
            )


def print_config(args):
    emit("MODEL", args.model)
    emit("DTYPE", args.dtype)
    emit("BATCH_SIZE", args.batch_size)
    emit("DEVICE", args.device)
    emit("COMPILE", args.compile_mode if args.compile else "off")
    emit("TF32", "on" if args.tf32 else "off")
    emit("WARMUP_SECONDS", args.warmup_seconds)
    emit("DURATION_SECONDS", args.duration)
    emit("COOLDOWN_SECONDS", args.cooldown_seconds)


def preflight(benchmark):
    """Runs every precision briefly and reports which ones work."""
    failures = 0
    try:
        # jiterator compiles on first call, so launch each marker once.
        if benchmark.prepare_markers():
            for name in PHASE_MARKERS:
                benchmark.mark(name)
            emit("PHASE_MARKERS", "ok")
        else:
            emit("PHASE_MARKERS", "missing")
    except Exception as error:  # pylint: disable=broad-exception-caught
        failures += 1
        print(f"PREFLIGHT markers: FAILED  {error!r}", flush=True)
    for dtype in DTYPES:
        try:
            model, inputs, rel_error = benchmark.build(dtype)
            elapsed, batches = benchmark.run_phase(model, inputs, 2.0)
            images_per_second = batches * benchmark.args.batch_size / elapsed
            print(
                f"PREFLIGHT {dtype}: OK  {images_per_second:.1f} img/s  "
                f"rel_error_vs_fp32={rel_error:.2e}",
                flush=True,
            )
        # build() raises SystemExit for an unsupported precision (fp8 below
        # compute capability 8.9); report it like any other failure.
        except (Exception, SystemExit) as error:  # pylint: disable=broad-exception-caught
            failures += 1
            print(f"PREFLIGHT {dtype}: FAILED  {error!r}", flush=True)
    return 1 if failures else 0


def measure(benchmark):
    """Runs the phased capture for one precision."""
    args = benchmark.args
    setup_start = time.perf_counter()
    model, inputs, rel_error = benchmark.build(args.dtype)
    markers_ready = benchmark.prepare_markers()
    emit("SETUP_SECONDS", f"{time.perf_counter() - setup_start:.3f}")
    emit("PHASE_MARKERS", "ok" if markers_ready else "missing")
    emit("FP8_LINEAR_LAYERS", benchmark.fp8_layers)
    emit("REL_L2_ERROR_VS_FP32", f"{rel_error:.6e}")

    benchmark.mark(PHASE_MARKERS[0])
    warmup_elapsed, warmup_batches = benchmark.run_phase(
        model, inputs, args.warmup_seconds
    )
    benchmark.mark(PHASE_MARKERS[1])
    elapsed, batches = benchmark.run_phase(model, inputs, args.duration)
    benchmark.mark(PHASE_MARKERS[2])
    cooldown_start = time.perf_counter()
    time.sleep(args.cooldown_seconds)
    cooldown_elapsed = time.perf_counter() - cooldown_start
    benchmark.mark(PHASE_MARKERS[3])

    images = batches * args.batch_size
    emit("WARMUP_ELAPSED_SECONDS", f"{warmup_elapsed:.6f}")
    emit("WARMUP_BATCHES", warmup_batches)
    emit("MEASUREMENT_ELAPSED_SECONDS", f"{elapsed:.6f}")
    emit("MEASUREMENT_BATCHES", batches)
    emit("MEASUREMENT_IMAGES", images)
    emit("IMAGES_PER_SECOND", f"{images / elapsed:.3f}")
    emit("MS_PER_BATCH", f"{1000.0 * elapsed / batches:.4f}")
    emit("COOLDOWN_ELAPSED_SECONDS", f"{cooldown_elapsed:.6f}")
    emit("STATUS", "OK")
    emit("EXECUTION_TIME_SECONDS", f"{elapsed:.6f}")
    emit("OPERATIONS_OR_KERNEL_LAUNCHES", batches)
    return 0


def main(argv=None):
    args = parse_args(argv)
    if strip_profiler_preload(os.environ):
        emit("PROFILER_PRELOAD_STRIPPED_FOR_CHILDREN", 1)
    import torch  # pylint: disable=import-outside-toplevel

    if args.device_type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available to this PyTorch build")
    device = torch.device(
        args.device_type, args.device if args.device_type == "cuda" else None
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32

    print_environment(torch, device)
    print_config(args)
    benchmark = Benchmark(args, torch, device)
    return preflight(benchmark) if args.preflight else measure(benchmark)


if __name__ == "__main__":
    sys.exit(main())
