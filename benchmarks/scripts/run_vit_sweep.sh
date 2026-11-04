#!/usr/bin/env bash
# Capture the ViT inference benchmark once per precision and repetition.
#
# Each run is one launcher session. Precision order alternates between
# repetitions (forward, then reversed) so thermal drift does not always favor
# the same precision. Results land in
#   <output-dir>/sessions/<NN>-<dtype>-rep-<R>/
# with the launcher log (benchmark.log) and the daemon CSV files.
#
# Usage:
#   benchmarks/scripts/run_vit_sweep.sh --python ~/klobo/.venv-vit/bin/python \
#     --output-dir results/vit-sweep/thor [--dtypes "fp32 fp16 bf16 fp8"] \
#     [--repetitions 3] [--batch-size 64] [--duration 40] [--compile]
#     [-- extra vit_inference.py arguments]
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
benchmark="${repo_root}/benchmarks/pytorch/vit_inference.py"
launcher="nvidia-process-metrics-launcher"
python_bin="python3"
output_dir=""
dtypes="fp32 fp16 bf16 fp8"
repetitions=3
batch_size=64
device=0
window_ms=200
warmup=10
duration=40
cooldown=10
# Compile can take minutes; the session ends when the benchmark exits.
setup_margin=600
extra=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --launcher) launcher="$2"; shift 2 ;;
    --python) python_bin="$2"; shift 2 ;;
    --output-dir) output_dir="$2"; shift 2 ;;
    --dtypes) dtypes="$2"; shift 2 ;;
    --repetitions) repetitions="$2"; shift 2 ;;
    --batch-size) batch_size="$2"; shift 2 ;;
    --device) device="$2"; shift 2 ;;
    --window-ms) window_ms="$2"; shift 2 ;;
    --warmup-seconds) warmup="$2"; shift 2 ;;
    --duration) duration="$2"; shift 2 ;;
    --cooldown-seconds) cooldown="$2"; shift 2 ;;
    --compile) extra+=(--compile); shift ;;
    --) shift; extra+=("$@"); break ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${output_dir}" ]]; then
  echo "--output-dir is required" >&2
  exit 2
fi
command -v "${launcher}" >/dev/null || {
  echo "launcher not found: ${launcher}" >&2
  exit 2
}
mkdir -p "${output_dir}/sessions"

session_seconds=$(python3 -c "import math; print(math.ceil(
  ${warmup} + ${duration} + ${cooldown}) + ${setup_margin})")
read -r -a forward <<<"${dtypes}"
sequence=0
failures=0

for ((rep = 1; rep <= repetitions; ++rep)); do
  order=("${forward[@]}")
  if ((rep % 2 == 0)); then
    order=()
    for ((i = ${#forward[@]} - 1; i >= 0; --i)); do
      order+=("${forward[i]}")
    done
  fi
  for dtype in "${order[@]}"; do
    sequence=$((sequence + 1))
    name=$(printf '%02d-%s-rep-%02d' "${sequence}" "${dtype}" "${rep}")
    session_dir="${output_dir}/sessions/${name}"
    mkdir -p "${session_dir}"
    echo "=== ${name} ==="
    status=0
    "${launcher}" --device "${device}" --duration "${session_seconds}" \
      --window-ms "${window_ms}" --command -- \
      "${python_bin}" "${benchmark}" --device "${device}" --dtype "${dtype}" \
      --batch-size "${batch_size}" --warmup-seconds "${warmup}" \
      --duration "${duration}" --cooldown-seconds "${cooldown}" \
      "${extra[@]}" 2>&1 | tee "${session_dir}/benchmark.log" ||
      status=$?
    results=$(sed -n 's/^Results:[[:space:]]*"\{0,1\}\([^"]*\)"\{0,1\}[[:space:]]*$/\1/p' \
      "${session_dir}/benchmark.log" | tail -n 1)
    if [[ -n "${results}" && -d "${results}" ]]; then
      cp -r "${results}/." "${session_dir}/"
    fi
    if [[ ${status} -ne 0 ]] || ! grep -q '^STATUS=OK' "${session_dir}/benchmark.log"; then
      echo "WARNING: ${name} failed (exit ${status}); kept its log" >&2
      failures=$((failures + 1))
    fi
  done
done

echo "Sessions: ${sequence}, failed: ${failures}, study: ${output_dir}"
[[ ${failures} -eq 0 ]]
