#!/bin/bash
#
# Multi-GPU Parallel Evaluation Script
#
# Usage: ./run_parallel.sh "<model1> <model2> ..." "<gpu1> <gpu2> ..." [dataset_subset] [num_samples]
#
# Examples:
#   ./run_parallel.sh "Qwen/Qwen3-VL-4B-Instruct Qwen/Qwen3-VL-8B-Instruct" "0 1"
#   ./run_parallel.sh "Qwen/Qwen3-VL-4B-Instruct lmms-lab/LLaVA-OneVision-1.5-8B-Instruct Qwen/Qwen3-VL-8B-Thinking" "0 1 2" 4-room 50
#

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
if [ "$#" -lt 2 ]; then
    echo -e "${RED}Error: Missing required arguments${NC}"
    echo ""
    echo "Usage: $0 \"<model1> <model2> ...\" \"<gpu1> <gpu2> ...\" [dataset_subset] [num_samples]"
    echo ""
    echo "Arguments:"
    echo "  models           - Space-separated list of model paths (in quotes)"
    echo "  gpus             - Space-separated list of GPU IDs (in quotes)"
    echo "  dataset_subset   - Dataset subset (default: 3-room)"
    echo "  num_samples      - Number of samples to evaluate (default: 25)"
    echo ""
    echo "Example:"
    echo "  $0 \"Qwen/Qwen3-VL-4B-Instruct Qwen/Qwen3-VL-8B-Instruct\" \"0 1\""
    echo ""
    echo "Supported Models:"
    echo "  - Qwen/Qwen3-VL-4B-Instruct"
    echo "  - Qwen/Qwen3-VL-4B-Thinking"
    echo "  - Qwen/Qwen3-VL-8B-Thinking"
    echo "  - Qwen/Qwen3-VL-8B-Instruct"
    echo "  - lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"
    echo "  - lmms-lab/LLaVA-OneVision-1.5-4B-Instruct"
    echo "  - ByteDance-Seed/BAGEL-7B-MoT (requires special setup)"
    echo ""
    exit 1
fi

MODELS_STR="$1"
GPUS_STR="$2"
DATASET_SUBSET="${3:-3-room}"
NUM_SAMPLES="${4:-25}"

# Convert space-separated strings to arrays
IFS=' ' read -r -a MODELS <<< "$MODELS_STR"
IFS=' ' read -r -a GPUS <<< "$GPUS_STR"

# Validate array lengths
if [ ${#MODELS[@]} -ne ${#GPUS[@]} ]; then
    echo -e "${RED}Error: Number of models (${#MODELS[@]}) must match number of GPUs (${#GPUS[@]})${NC}"
    exit 1
fi

# Configuration
OUTPUT_DIR="results"
RENDER_MODE="vision,text"
EXP_TYPE="active,passive"
MAX_EXP_STEPS=20
INFERENCE_MODE="direct"

# Print configuration
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Theory-of-Space Parallel Evaluation${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Configuration:"
echo "  Number of models: ${#MODELS[@]}"
echo "  Dataset:          $DATASET_SUBSET"
echo "  Samples:          $NUM_SAMPLES"
echo "  Output:           $OUTPUT_DIR"
echo "  Render modes:     $RENDER_MODE"
echo "  Experiment type:  $EXP_TYPE"
echo ""
echo "Model-GPU Mapping:"
for i in "${!MODELS[@]}"; do
    echo "  GPU ${GPUS[$i]}: ${MODELS[$i]}"
done
echo ""
echo -e "${YELLOW}Starting parallel evaluation...${NC}"
echo ""

# Array to store background process PIDs
declare -a PIDS
declare -a LOG_FILES

# Launch evaluation for each model in parallel
for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    GPU="${GPUS[$i]}"
    VLLM_PORT=$((9999 + GPU))

    # Create log file name
    MODEL_SAFE=$(echo "$MODEL" | tr '/' '_' | tr '-' '_')
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="logs/${MODEL_SAFE}_gpu${GPU}_${TIMESTAMP}.log"
    mkdir -p logs

    echo -e "${BLUE}[GPU $GPU]${NC} Starting evaluation for $MODEL"
    echo "         Log file: $LOG_FILE"

    # Run evaluation in background
    (
        export CUDA_VISIBLE_DEVICES=$GPU
        python evaluate_vlm.py \
            --model_path "$MODEL" \
            --gpu_id "$GPU" \
            --dataset_subset "$DATASET_SUBSET" \
            --num_samples "$NUM_SAMPLES" \
            --output_dir "$OUTPUT_DIR" \
            --render_mode "$RENDER_MODE" \
            --exp_type "$EXP_TYPE" \
            --max_exp_steps "$MAX_EXP_STEPS" \
            --inference_mode "$INFERENCE_MODE" \
            --vllm_port "$VLLM_PORT" \
            > "$LOG_FILE" 2>&1

        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 0 ]; then
            echo -e "${GREEN}[GPU $GPU] ✓ $MODEL completed successfully${NC}"
        else
            echo -e "${RED}[GPU $GPU] ✗ $MODEL failed (exit code: $EXIT_CODE)${NC}"
        fi
        exit $EXIT_CODE
    ) &

    PID=$!
    PIDS+=($PID)
    LOG_FILES+=($LOG_FILE)

    # Small delay to avoid startup conflicts
    sleep 2
done

echo ""
echo -e "${YELLOW}All evaluations launched. Waiting for completion...${NC}"
echo ""
echo "You can monitor progress by tailing the log files:"
for LOG in "${LOG_FILES[@]}"; do
    echo "  tail -f $LOG"
done
echo ""

# Wait for all background processes
FAILED=0
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    MODEL=${MODELS[$i]}
    GPU=${GPUS[$i]}

    echo -e "${YELLOW}Waiting for GPU $GPU: $MODEL (PID: $PID)...${NC}"

    if wait $PID; then
        echo -e "${GREEN}[GPU $GPU] ✓ $MODEL completed successfully${NC}"
    else
        EXIT_CODE=$?
        echo -e "${RED}[GPU $GPU] ✗ $MODEL failed (exit code: $EXIT_CODE)${NC}"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo -e "${GREEN}========================================${NC}"
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All Evaluations Completed!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Summary:"
    echo "  Total models:    ${#MODELS[@]}"
    echo "  Successful:      $((${#MODELS[@]} - FAILED))"
    echo "  Failed:          $FAILED"
    echo ""
    echo "To view results, you can:"
    echo "  1. Start HTTP server: python -m http.server 8000"
    echo "  2. Open browser: http://localhost:8000/$OUTPUT_DIR/[model_name]/env_data.html"
else
    echo -e "${RED}✗ Some Evaluations Failed!${NC}"
    echo -e "${RED}========================================${NC}"
    echo ""
    echo "Summary:"
    echo "  Total models:    ${#MODELS[@]}"
    echo "  Successful:      $((${#MODELS[@]} - FAILED))"
    echo "  Failed:          $FAILED"
    echo ""
    echo "Check log files for details:"
    for LOG in "${LOG_FILES[@]}"; do
        echo "  $LOG"
    done
    exit 1
fi
