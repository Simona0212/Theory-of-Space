#!/bin/bash
#
# Single-GPU Single-Model Evaluation Script
#
# Usage: ./run_single.sh <model_name> <gpu_id> [dataset_subset] [num_samples]
#
# Examples:
#   ./run_single.sh Qwen/Qwen3-VL-4B-Instruct 0
#   ./run_single.sh lmms-lab/LLaVA-OneVision-1.5-8B-Instruct 1 4-room 25
#   ./run_single.sh Qwen/Qwen3-VL-8B-Thinking 0 3-room 50
#

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
if [ "$#" -lt 2 ]; then
    echo -e "${RED}Error: Missing required arguments${NC}"
    echo ""
    echo "Usage: $0 <model_name> <gpu_id> [dataset_subset] [num_samples]"
    echo ""
    echo "Arguments:"
    echo "  model_name       - Hugging Face model path (e.g., Qwen/Qwen3-VL-4B-Instruct)"
    echo "  gpu_id           - GPU device ID to use (e.g., 0, 1, 2, ...)"
    echo "  dataset_subset   - Dataset subset (default: 3-room)"
    echo "  num_samples      - Number of samples to evaluate (default: 25)"
    echo ""
    echo "Supported Models:"
    echo "  Qwen Models:"
    echo "    - Qwen/Qwen3-VL-4B-Instruct"
    echo "    - Qwen/Qwen3-VL-4B-Thinking"
    echo "    - Qwen/Qwen3-VL-8B-Thinking"
    echo "    - Qwen/Qwen3-VL-8B-Instruct"
    echo ""
    echo "  LLaVA Models:"
    echo "    - lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"
    echo "    - lmms-lab/LLaVA-OneVision-1.5-4B-Instruct"
    echo ""
    echo "  BAGEL Models:"
    echo "    - ByteDance-Seed/BAGEL-7B-MoT (requires special setup)"
    echo ""
    exit 1
fi

MODEL_NAME="$1"
GPU_ID="$2"
DATASET_SUBSET="${3:-3-room}"
NUM_SAMPLES="${4:-25}"

# Configuration
OUTPUT_DIR="results"
RENDER_MODE="vision,text"
EXP_TYPE="active,passive"
MAX_EXP_STEPS=20
INFERENCE_MODE="direct"
VLLM_PORT=$((9999 + GPU_ID))  # Offset port by GPU ID to avoid conflicts

# Print configuration
echo -e "${GREEN}=================================${NC}"
echo -e "${GREEN}Theory-of-Space Evaluation${NC}"
echo -e "${GREEN}=================================${NC}"
echo ""
echo "Configuration:"
echo "  Model:           $MODEL_NAME"
echo "  GPU ID:          $GPU_ID"
echo "  Dataset:         $DATASET_SUBSET"
echo "  Samples:         $NUM_SAMPLES"
echo "  Output:          $OUTPUT_DIR"
echo "  Render modes:    $RENDER_MODE"
echo "  Experiment type: $EXP_TYPE"
echo "  vLLM Port:       $VLLM_PORT"
echo ""
echo -e "${YELLOW}Starting evaluation...${NC}"
echo ""

# Set CUDA device
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Run evaluation
python evaluate_vlm.py \
    --model_path "$MODEL_NAME" \
    --gpu_id "$GPU_ID" \
    --dataset_subset "$DATASET_SUBSET" \
    --num_samples "$NUM_SAMPLES" \
    --output_dir "$OUTPUT_DIR" \
    --render_mode "$RENDER_MODE" \
    --exp_type "$EXP_TYPE" \
    --max_exp_steps "$MAX_EXP_STEPS" \
    --inference_mode "$INFERENCE_MODE" \
    --vllm_port "$VLLM_PORT"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=================================${NC}"
    echo -e "${GREEN}✓ Evaluation Completed!${NC}"
    echo -e "${GREEN}=================================${NC}"
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "To view results, you can:"
    echo "  1. Start HTTP server: python -m http.server 8000"
    echo "  2. Open browser: http://localhost:8000/$OUTPUT_DIR/[model_name]/env_data.html"
else
    echo ""
    echo -e "${RED}=================================${NC}"
    echo -e "${RED}✗ Evaluation Failed!${NC}"
    echo -e "${RED}=================================${NC}"
    echo ""
    exit 1
fi
