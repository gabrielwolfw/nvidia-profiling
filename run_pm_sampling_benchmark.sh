#!/usr/bin/env bash

set -uo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

DEVICE="${DEVICE:-0}"
BENCHMARK_NAME="${BENCHMARK_NAME:-compute}"
DURATION_SECONDS="${DURATION_SECONDS:-10}"
PM_DURATION_SECONDS="${PM_DURATION_SECONDS:-12}"
PM_MAX_SAMPLES="${PM_MAX_SAMPLES:-100000}"
WARMUP="${WARMUP:-5}"
BLOCK_SIZE="${BLOCK_SIZE:-256}"
ITERATIONS="${ITERATIONS:-10000}"

PM_SAMPLER="${ROOT_DIR}/nvidia/pm_sampling_simple"
METRICS_LIBRARY="${ROOT_DIR}/nvidia/libnvidia-process-metrics.so"
case "${BENCHMARK_NAME}" in
  compute|memory|stride|transfer|idle)
    BENCHMARK_EXECUTABLE="${ROOT_DIR}/benchmarks/build/bench_${BENCHMARK_NAME}"
    ;;
  *)
    printf 'ERROR: BENCHMARK_NAME must be compute, memory, stride, transfer, or idle.\n' >&2
    exit 1
    ;;
esac

RUN_ID="$(date +%Y%m%d-%H%M%S)-$$-${BENCHMARK_NAME}"
RESULTS_DIR="${ROOT_DIR}/results/${RUN_ID}"
PM_LOG="${RESULTS_DIR}/pm_sampling_${BENCHMARK_NAME}.log"
RAW_PM_CSV="${RESULTS_DIR}/pm_sampling.csv"
RAW_TELEMETRY_CSV="${RESULTS_DIR}/gpu_telemetry.csv"
RAW_KERNEL_CSV="${RESULTS_DIR}/kernel_activity.csv"
PM_CSV="${RESULTS_DIR}/pm_sampling_${BENCHMARK_NAME}.csv"
TELEMETRY_CSV="${RESULTS_DIR}/gpu_telemetry_${BENCHMARK_NAME}.csv"
KERNEL_CSV="${RESULTS_DIR}/kernel_activity_${BENCHMARK_NAME}.csv"

for required_file in "${PM_SAMPLER}" "${METRICS_LIBRARY}" "${BENCHMARK_EXECUTABLE}"; do
  if [[ ! -e "${required_file}" ]]; then
    printf 'ERROR: required file not found: %s\n' "${required_file}" >&2
    exit 1
  fi
done

if [[ ! -x "${PM_SAMPLER}" || ! -x "${BENCHMARK_EXECUTABLE}" ]]; then
  printf 'ERROR: PM sampler and benchmark must be executable.\n' >&2
  exit 1
fi

mkdir -p "${RESULTS_DIR}"

# Ask for the sudo password before starting anything in the background.
sudo -v || exit 1

printf 'Starting PM Sampling for %s seconds on GPU %s...\n' \
  "${PM_DURATION_SECONDS}" "${DEVICE}"
printf 'PM counter-data capacity: %s samples\n' "${PM_MAX_SAMPLES}"
printf 'PM Sampling output: %s\n' "${PM_LOG}"

(
  cd "${RESULTS_DIR}" || exit 1
  sudo "${PM_SAMPLER}" \
    --device "${DEVICE}" \
    --duration "${PM_DURATION_SECONDS}" \
    --maxsamples "${PM_MAX_SAMPLES}"
) >"${PM_LOG}" 2>&1 &

pm_pid=$!
pm_ready=false

# Do not start the benchmark until the sampler has enabled PM collection.
for ((attempt = 0; attempt < 100; ++attempt)); do
  if grep -q "PM Sampling active" "${PM_LOG}"; then
    pm_ready=true
    break
  fi

  if ! kill -0 "${pm_pid}" 2>/dev/null; then
    break
  fi

  sleep 0.1
done

if [[ "${pm_ready}" != true ]]; then
  printf 'ERROR: PM Sampling did not become ready. Its output follows:\n' >&2
  wait "${pm_pid}" 2>/dev/null || true
  sed -n '1,160p' "${PM_LOG}" >&2
  exit 1
fi

printf '\nPM Sampling is active. Starting bench_%s in the foreground.\n\n' \
  "${BENCHMARK_NAME}"

benchmark_arguments=(
  --device "${DEVICE}"
  --duration "${DURATION_SECONDS}"
  --warmup "${WARMUP}"
  --block-size "${BLOCK_SIZE}"
)

if [[ "${BENCHMARK_NAME}" == "compute" ]]; then
  benchmark_arguments+=(--iterations "${ITERATIONS}")
fi

benchmark_status=0
NVIDIA_METRICS_DEVICE="${DEVICE}" \
NVIDIA_METRICS_OUTPUT_DIR="${RESULTS_DIR}" \
LD_PRELOAD="${METRICS_LIBRARY}" \
  "${BENCHMARK_EXECUTABLE}" "${benchmark_arguments[@]}" \
  || benchmark_status=$?

printf '\nBenchmark finished. Waiting for PM Sampling...\n'

pm_status=0
wait "${pm_pid}" || pm_status=$?

if [[ -f "${RAW_PM_CSV}" ]]; then
  sudo chown "$(id -u):$(id -g)" "${RAW_PM_CSV}" 2>/dev/null || true
  mv -- "${RAW_PM_CSV}" "${PM_CSV}"
fi

if [[ -f "${RAW_TELEMETRY_CSV}" ]]; then
  mv -- "${RAW_TELEMETRY_CSV}" "${TELEMETRY_CSV}"
fi

if [[ -f "${RAW_KERNEL_CSV}" ]]; then
  mv -- "${RAW_KERNEL_CSV}" "${KERNEL_CSV}"
fi

printf '\nRun complete.\n'
printf 'Results directory: %s\n' "${RESULTS_DIR}"
printf 'PM Sampling log: %s\n' "${PM_LOG}"
if [[ -f "${TELEMETRY_CSV}" ]]; then
  printf 'NVML telemetry CSV: %s\n' "${TELEMETRY_CSV}"
fi
if [[ -f "${KERNEL_CSV}" ]]; then
  printf 'CUPTI kernel CSV: %s\n' "${KERNEL_CSV}"
fi
if [[ -f "${PM_CSV}" ]]; then
  printf 'PM Sampling CSV: %s\n' "${PM_CSV}"
fi

if ((benchmark_status != 0)); then
  printf 'ERROR: bench_%s exited with status %d.\n' \
    "${BENCHMARK_NAME}" "${benchmark_status}" >&2
  exit "${benchmark_status}"
fi

if ((pm_status != 0)); then
  printf 'ERROR: PM Sampling exited with status %d.\n' "${pm_status}" >&2
  exit "${pm_status}"
fi
