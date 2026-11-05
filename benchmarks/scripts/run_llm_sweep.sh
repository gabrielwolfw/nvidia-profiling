#!/usr/bin/env bash
# Capture llama.cpp text generation once per GGUF model and repetition.
#
# Each run is one launcher session with the same prompt and token budget, so
# sessions of different models are comparable. Model order alternates between
# repetitions (forward, then reversed) so thermal drift does not always favor
# the same model. Results land in
#   <output-dir>/sessions/<NN>-<name>-rep-<R>/
# with the launcher log (benchmark.log) and the daemon CSV files;
# <output-dir>/models.txt records the llama.cpp commit and model checksums.
#
# Usage:
#   benchmarks/scripts/run_llm_sweep.sh \
#     --llama-bin ~/llama.cpp/build/bin/llama-completion \
#     --output-dir results/llm-sweep/v100 \
#     --model llama8b=/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
#     --model qwen7b=/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
#     [--repetitions 3] [--tokens 6000] [--ctx-size 8192] [--duration 300]
#     [--window-ms 200] [--device 0] [--prompt TEXT]
#     [--launcher-args "--socket /tmp/daemon.sock --library /path/lib.so"]
#     [-- extra llama.cpp arguments]
set -euo pipefail

launcher="nvidia-process-metrics-launcher"
launcher_args=""
llama_bin=""
output_dir=""
repetitions=3
tokens=6000
ctx_size=8192
duration=300
window_ms=200
device=0
seed=2026
prompt="Explain GPU energy profiling, including power, energy, utilization, \
hardware metrics, memory traffic, and methodological limitations."
names=()
declare -A model_path
extra=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --launcher) launcher="$2"; shift 2 ;;
    --launcher-args) launcher_args="$2"; shift 2 ;;
    --llama-bin) llama_bin="$2"; shift 2 ;;
    --output-dir) output_dir="$2"; shift 2 ;;
    --model)
      [[ "$2" == *=* ]] || { echo "--model expects NAME=PATH" >&2; exit 2; }
      names+=("${2%%=*}"); model_path["${2%%=*}"]="${2#*=}"; shift 2 ;;
    --repetitions) repetitions="$2"; shift 2 ;;
    --tokens) tokens="$2"; shift 2 ;;
    --ctx-size) ctx_size="$2"; shift 2 ;;
    --duration) duration="$2"; shift 2 ;;
    --window-ms) window_ms="$2"; shift 2 ;;
    --device) device="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --prompt) prompt="$2"; shift 2 ;;
    --) shift; extra+=("$@"); break ;;
    -h|--help) sed -n '2,21p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${output_dir}" || -z "${llama_bin}" || ${#names[@]} -eq 0 ]]; then
  echo "--output-dir, --llama-bin and at least one --model are required" >&2
  exit 2
fi
[[ -x "${llama_bin}" ]] || { echo "not executable: ${llama_bin}" >&2; exit 2; }
command -v "${launcher}" >/dev/null || {
  echo "launcher not found: ${launcher}" >&2
  exit 2
}
for name in "${names[@]}"; do
  [[ -f "${model_path[${name}]}" ]] || {
    echo "model not found: ${model_path[${name}]}" >&2
    exit 2
  }
done
read -r -a launcher_extra <<<"${launcher_args}"
mkdir -p "${output_dir}/sessions"

{
  llama_dir="$(cd "$(dirname "${llama_bin}")" && pwd)"
  echo "llama.cpp $(git -C "${llama_dir}" log -1 --format='%h %cd' 2>/dev/null ||
    echo unknown)"
  for name in "${names[@]}"; do
    echo "${name} $(sha256sum "${model_path[${name}]}")"
  done
} | tee "${output_dir}/models.txt"

sequence=0
failures=0
for ((rep = 1; rep <= repetitions; ++rep)); do
  order=("${names[@]}")
  if ((rep % 2 == 0)); then
    order=()
    for ((i = ${#names[@]} - 1; i >= 0; --i)); do
      order+=("${names[i]}")
    done
  fi
  for name in "${order[@]}"; do
    sequence=$((sequence + 1))
    session=$(printf '%02d-%s-rep-%02d' "${sequence}" "${name}" "${rep}")
    session_dir="${output_dir}/sessions/${session}"
    mkdir -p "${session_dir}"
    echo "=== ${session} ==="
    status=0
    "${launcher}" "${launcher_extra[@]}" --device "${device}" \
      --duration "${duration}" --window-ms "${window_ms}" --command -- \
      "${llama_bin}" -m "${model_path[${name}]}" --gpu-layers 99 \
      --ctx-size "${ctx_size}" -p "${prompt}" -n "${tokens}" --ignore-eos \
      --temp 0 --seed "${seed}" -no-cnv "${extra[@]}" \
      >"${session_dir}/benchmark.log" 2>&1 || status=$?
    results=$(sed -n 's/^Results:[[:space:]]*"\{0,1\}\([^"]*\)"\{0,1\}[[:space:]]*$/\1/p' \
      "${session_dir}/benchmark.log" | tail -n 1)
    if [[ -n "${results}" && -d "${results}" ]]; then
      cp -r "${results}/." "${session_dir}/"
    fi
    grep -E ' eval time' "${session_dir}/benchmark.log" | grep -v prompt || true
    if [[ ${status} -ne 0 ]] ||
      ! grep -q '^capture_status=complete' "${session_dir}/capture_summary.txt" \
        2>/dev/null; then
      echo "WARNING: ${session} failed (exit ${status}); kept its log" >&2
      failures=$((failures + 1))
    fi
  done
done

echo "Sessions: ${sequence}, failed: ${failures}, study: ${output_dir}"
[[ ${failures} -eq 0 ]]
