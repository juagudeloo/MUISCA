#!/usr/bin/env bash
#SBATCH --job-name=nicole_syn
#SBATCH --cluster=fisica           #nombre de los cluster a donde envia a procesar
#SBATCH -wmaxwell               #Nombre del nodo a usar (configurable via CLUSTER_NODE variable)
#SBATCH --partition=gpu.cecc            #Particion a usar(puede ser: cpu.cecc o gpu.cecc)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --mail-type=begin             #Send email when job begins
#SBATCH --mail-type=end               #Send email when job ends
#SBATCH --mail-user=juagudeloo@unal.edu.co
#SBATCH --output=/scratchsan/observatorio/juagudeloo/MUISCA/output/synthesis/nicole_syn_%j.out
#SBATCH --error=/scratchsan/observatorio/juagudeloo/MUISCA/output/synthesis/nicole_syn_%j.err

set -euo pipefail

# Absolute project root. Everything below addresses the repo through it, so this script
# can be submitted or invoked from any working directory -- including from tools/ itself,
# which keeps SLURM from staging the whole repo just to run one job.
MUISCA_ROOT="/scratchsan/observatorio/juagudeloo/MUISCA"
cd "${MUISCA_ROOT}" || exit 1

# ==============================================================================
# SYNTHESIS CONFIGURATION
# ==============================================================================
EXPERIMENT_ROOT="experiment_110_to_130-step_size_10-normal"
MODEL_TYPES=("no_physics" "wfa_only" "doppler_only" "black_body_only")   # one or more model
                                          # variations to run. Step 0's sampling is
                                          # model-independent (SPINOR- or MURaM-ground-truth-
                                          # sourced), so every variant here gets the SAME pixel
                                          # selection automatically. With 2+ entries, steps 4-5
                                          # (cross-model comparison) also run at the end.

# Runtime control -- mirrors tools/generate_analysis.sh's --run selector. "both" runs the
# synthesis pipeline once against MODEST (real Hinode/SOT-SP observations) and once against
# MURaM (a simulation step, e.g. one outside the model's training window, for an
# out-of-distribution generalization check) in the same submission.
RUN_TARGET="modest"                     # both | muram | modest

# MURaM args (used when RUN_TARGET is both|muram)
MURAM_STEP="198"
ADD_GT_PRESSURE="0"             # 1 => --add-gt-pressure: feed NICOLE the true MURaM gas
                                 # pressure instead of a hydrostatic-equilibrium seed. Runs
                                 # land in a sibling step-N-gt-pressure/ tree, so a plain run
                                 # and this one can be diffed. muram only.

# MODEST args (used when RUN_TARGET is both|modest)
#
# List every region to process in one run. label:Y0,Y1,X0,X1 -- same order as
# ModestData.extract_region (mirrors tools/generate_analysis.sh's REGIONS). Each region gets
# its own predictions.h5/syntheses.h5 tree under
# output/synthesis/<experiment>/modest/<model>/<label>/, and (with 2+ MODEL_TYPES) its own
# steps 4-5 cross-model comparison. Must have at least one entry, or INCLUDE_WHOLE=1 -- see
# the validation below.
REGIONS=(
  "sunspot:100,300,250,450"
  "plage:0,100,400,600"
  "negative_region:0,80,0,200"
  "quiet_sun:0,100,600,700"
)
INCLUDE_WHOLE="0"   # 1 => also process the whole (uncropped) scene alongside REGIONS
MODEST_CACHE_DIR="/scratchsan/observatorio/juagudeloo/MUISCA/.modest_cache"

# Pixel selection (shared by muram and modest). Either let step 0 auto-select a
# stratified-by-|B_LOS| sample spanning weak/mid/strong field regimes, or pin an explicit
# manual list.
USE_STRATIFIED_SAMPLING=true
N_BINS=10                    # number of log-spaced |B_LOS| bins (see utils/pixel_sampling.py)
N_PER_BIN=15                # pixels sampled per bin (violin/aggregate tier -- step 5)
N_OVERLAY_PER_BIN=5          # subset of N_PER_BIN flagged for individual overlay PNGs (step 4 only)
SAMPLING_SEED=0
PIXELS=("40,100")           # used only when USE_STRATIFIED_SAMPLING=false

NICOLE_ROOT="/scratchsan/observatorio/juagudeloo/NICOLE_v16.06"
NICOLE_ASSETS="/scratchsan/observatorio/juagudeloo/MUISCA/data/nicole_assets"
OUTPUT_ROOT="/scratchsan/observatorio/juagudeloo/MUISCA/output/synthesis"

# ==============================================================================
# CLI (overrides the CONFIGURATION defaults above)
# ==============================================================================
usage() {
  cat <<'EOF'
Usage: tools/run_nicole_synthesis.sh [--run both|muram|modest] [--step N] [--add-gt-pressure 0|1] [--include-whole 0|1]

Options:
  --run both|muram|modest   Which source(s) to synthesize against (default: modest)
  --step N                  MURaM simulation step number (used when --run is both|muram; default: 198)
  --add-gt-pressure 0|1     Feed NICOLE the true MURaM gas pressure instead of a
                             hydrostatic seed (muram only). Output lands in a sibling
                             step-N-gt-pressure/ tree.
  --include-whole 0|1       MODEST: also process the whole (uncropped) scene alongside
                             REGIONS (default: 0)
  -h, --help                 Show this help

Everything else (EXPERIMENT_ROOT, MODEL_TYPES, REGIONS, N_BINS, ...) is configured by
editing the CONFIGURATION block at the top of this script.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run)
      RUN_TARGET="${2:-}"
      shift 2
      ;;
    --step)
      MURAM_STEP="${2:-}"
      shift 2
      ;;
    --add-gt-pressure)
      ADD_GT_PRESSURE="${2:-}"
      shift 2
      ;;
    --include-whole)
      INCLUDE_WHOLE="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

case "${RUN_TARGET}" in
  both|muram|modest) ;;
  *)
    echo "Invalid value for --run: ${RUN_TARGET} (use: both|muram|modest)" >&2
    exit 1
    ;;
esac

if [[ "${ADD_GT_PRESSURE}" != "0" && "${ADD_GT_PRESSURE}" != "1" ]]; then
  echo "Invalid value for --add-gt-pressure: ${ADD_GT_PRESSURE} (use: 0|1)" >&2
  exit 1
fi

if [[ "${INCLUDE_WHOLE}" != "0" && "${INCLUDE_WHOLE}" != "1" ]]; then
  echo "Invalid value for --include-whole: ${INCLUDE_WHOLE} (use: 0|1)" >&2
  exit 1
fi

ACTIVE_SOURCES=()
[[ "${RUN_TARGET}" == "both" || "${RUN_TARGET}" == "muram" ]] && ACTIVE_SOURCES+=("muram")
[[ "${RUN_TARGET}" == "both" || "${RUN_TARGET}" == "modest" ]] && ACTIVE_SOURCES+=("modest")

if [[ " ${ACTIVE_SOURCES[*]} " == *" muram "* ]]; then
  if [[ -z "${MURAM_STEP}" || ! "${MURAM_STEP}" =~ ^[0-9]+$ ]]; then
    echo "--step N (integer) is required when --run includes muram" >&2
    exit 1
  fi
fi

if [[ " ${ACTIVE_SOURCES[*]} " == *" modest "* && ${#REGIONS[@]} -eq 0 && "${INCLUDE_WHOLE}" != "1" ]]; then
  echo "Set at least one entry in REGIONS, or INCLUDE_WHOLE=1, to run MODEST synthesis." >&2
  exit 1
fi

# Regions to process for the modest source: REGIONS entries, plus "whole" (bounds-less) when
# INCLUDE_WHOLE=1 -- mirrors modest_analysis.py's _build_regions_to_run.
MODEST_REGIONS_TO_RUN=("${REGIONS[@]}")
if [[ "${INCLUDE_WHOLE}" == "1" ]]; then
  MODEST_REGIONS_TO_RUN=("whole:" "${MODEST_REGIONS_TO_RUN[@]}")
fi

# ==============================================================================
# RUN SYNTHESIS
# ==============================================================================

for SOURCE in "${ACTIVE_SOURCES[@]}"; do
  echo "################################################################"
  echo "# Source: ${SOURCE}"
  echo "################################################################"

  SOURCE_ARGS=(--source "${SOURCE}")
  GT_PRESSURE_ARGS=()
  if [[ "${SOURCE}" == "muram" ]]; then
    SOURCE_ARGS+=(--muram-step "${MURAM_STEP}")
    [[ "${ADD_GT_PRESSURE}" == "1" ]] && GT_PRESSURE_ARGS+=(--add-gt-pressure)
  fi

  # muram has no region concept -- a single-entry loop keeps the model loop below shared
  # between both sources instead of duplicating its body.
  if [[ "${SOURCE}" == "modest" ]]; then
    REGION_ENTRIES=("${MODEST_REGIONS_TO_RUN[@]}")
  else
    REGION_ENTRIES=("__no_region__:")
  fi

  for REGION_ENTRY in "${REGION_ENTRIES[@]}"; do
    if [[ "${SOURCE}" == "modest" ]]; then
      MODEST_REGION_LABEL="${REGION_ENTRY%%:*}"
      REGION_BOUNDS_STR="${REGION_ENTRY#*:}"
      if [[ "${MODEST_REGION_LABEL}" == "whole" && -z "${REGION_BOUNDS_STR}" ]]; then
        MODEST_REGION_ARGS=(--region-label "whole" --modest-cache-dir "${MODEST_CACHE_DIR}")
      else
        IFS=',' read -r Y0 Y1 X0 X1 <<< "${REGION_BOUNDS_STR}"
        MODEST_REGION_ARGS=(--region-label "${MODEST_REGION_LABEL}" --crop-bounds "${Y0}" "${Y1}" "${X0}" "${X1}" --modest-cache-dir "${MODEST_CACHE_DIR}")
      fi
      echo "================================================================"
      echo "# Region: ${MODEST_REGION_LABEL}"
      echo "================================================================"
    fi

  for MODEL_TYPE in "${MODEL_TYPES[@]}"; do
    echo "----------------------------------------------------------------"
    echo "# Model variant: ${MODEL_TYPE}  (source=${SOURCE}$( [[ "${SOURCE}" == "muram" ]] && echo ", step=${MURAM_STEP}, gt_pressure=${ADD_GT_PRESSURE}" ))"
    echo "----------------------------------------------------------------"

    if [[ "${SOURCE}" == "muram" ]]; then
      STEP_LABEL="step-${MURAM_STEP}"
      [[ "${ADD_GT_PRESSURE}" == "1" ]] && STEP_LABEL="${STEP_LABEL}-gt-pressure"
      REGION_OUT_DIR="${OUTPUT_ROOT}/${EXPERIMENT_ROOT}/muram/${STEP_LABEL}/${MODEL_TYPE}"
      # Step 0 (sample_pixels.py) never takes --add-gt-pressure -- pixel stratification is
      # pressure-independent by design, so a plain run and its -gt-pressure sibling must
      # sample the SAME pixels for a fair diff. Its output therefore always lives under the
      # plain step-N/ dir, even when this run's REGION_OUT_DIR (predictions/syntheses) has
      # the suffix.
      SAMPLE_OUT_DIR="${OUTPUT_ROOT}/${EXPERIMENT_ROOT}/muram/step-${MURAM_STEP}/${MODEL_TYPE}"
    else
      REGION_OUT_DIR="${OUTPUT_ROOT}/${EXPERIMENT_ROOT}/modest/${MODEL_TYPE}/${MODEST_REGION_LABEL}"
      SAMPLE_OUT_DIR="${REGION_OUT_DIR}"
    fi
    RUN_PIXELS=("${PIXELS[@]}")

    if [ "${USE_STRATIFIED_SAMPLING}" = true ]; then
      echo "=== Step 0: stratified pixel sampling by |B_LOS| (model-independent) ==="
      SAMPLE_ARGS=(
        "${SOURCE_ARGS[@]}"
        --experiment-root "${EXPERIMENT_ROOT}"
        --model-type "${MODEL_TYPE}"
        --n-bins "${N_BINS}"
        --n-per-bin "${N_PER_BIN}"
        --n-overlay-per-bin "${N_OVERLAY_PER_BIN}"
        --seed "${SAMPLING_SEED}"
        --output-root "${OUTPUT_ROOT}"
      )
      [[ "${SOURCE}" == "modest" ]] && SAMPLE_ARGS+=("${MODEST_REGION_ARGS[@]}")
      python "${MUISCA_ROOT}/scripts/synthesis/sample_pixels.py" "${SAMPLE_ARGS[@]}"

      SELECTED_JSON="${SAMPLE_OUT_DIR}/pixel_selection/selected_pixels.json"
      mapfile -t RUN_PIXELS < <(python -c "
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
for p in data['pixels']:
    print(f\"{p['ix']},{p['iy']}\")
" "${SELECTED_JSON}")
      echo "Selected ${#RUN_PIXELS[@]} pixels: ${RUN_PIXELS[*]}"
      echo
    fi

    PIXEL_ARGS=()
    for px in "${RUN_PIXELS[@]}"; do
      PIXEL_ARGS+=(--pixel "${px}")
    done

    echo "=== Step 1: export predictions ==="
    EXPORT_ARGS=(
      "${SOURCE_ARGS[@]}"
      "${GT_PRESSURE_ARGS[@]}"
      --experiment-root "${EXPERIMENT_ROOT}"
      --model-type "${MODEL_TYPE}"
      "${PIXEL_ARGS[@]}"
      --output-root "${OUTPUT_ROOT}"
    )
    [[ "${SOURCE}" == "modest" ]] && EXPORT_ARGS+=("${MODEST_REGION_ARGS[@]}")
    python "${MUISCA_ROOT}/scripts/synthesis/export_predictions.py" "${EXPORT_ARGS[@]}"

    PRED_H5="${REGION_OUT_DIR}/predictions.h5"

    echo
    echo "=== Step 2: run NICOLE synthesis ==="
    python "${MUISCA_ROOT}/scripts/synthesis/run_nicole_synthesis.py" \
      --predictions-h5 "${PRED_H5}" \
      --nicole-root "${NICOLE_ROOT}" \
      --nicole-assets "${NICOLE_ASSETS}"

    SYNTH_H5="${REGION_OUT_DIR}/syntheses.h5"

    echo
    echo "=== Step 3: compare (single model) ==="
    python "${MUISCA_ROOT}/scripts/synthesis/compare_synthesis.py" \
      --predictions-h5 "${PRED_H5}" \
      --syntheses-h5 "${SYNTH_H5}"

    echo
  done

  if [ "${#MODEL_TYPES[@]}" -ge 2 ]; then
    echo "################################################################"
    echo "# Step 4: cross-model comparison (pixel_comparison/, overlay tier) -- source=${SOURCE}"
    echo "################################################################"
    MODEL_TYPE_ARGS=()
    for mt in "${MODEL_TYPES[@]}"; do
      MODEL_TYPE_ARGS+=(--model-type "${mt}")
    done
    COMPARE_ARGS=(
      "${SOURCE_ARGS[@]}"
      "${GT_PRESSURE_ARGS[@]}"
      --experiment-root "${EXPERIMENT_ROOT}"
      --output-root "${OUTPUT_ROOT}"
      "${MODEL_TYPE_ARGS[@]}"
    )
    [[ "${SOURCE}" == "modest" ]] && COMPARE_ARGS+=(--region-label "${MODEST_REGION_LABEL}")
    python "${MUISCA_ROOT}/scripts/synthesis/compare_models.py" "${COMPARE_ARGS[@]}"

    echo
    echo "################################################################"
    echo "# Step 5: aggregate distribution comparison (aggregate_plots/, violin tier) -- source=${SOURCE}"
    echo "################################################################"
    python "${MUISCA_ROOT}/scripts/synthesis/aggregate_comparison.py" "${COMPARE_ARGS[@]}"
  fi

  done

  echo
done
