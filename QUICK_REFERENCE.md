# Theory-of-Space VLM Evaluation - Quick Reference

## Supported Models

| Family | Model | Notes |
|--------|-------|-------|
| Qwen3-VL | `Qwen/Qwen3-VL-4B-Instruct` | ✅ Ready |
| Qwen3-VL | `Qwen/Qwen3-VL-4B-Thinking` | ✅ Ready |
| Qwen3-VL | `Qwen/Qwen3-VL-8B-Thinking` | ✅ Ready |
| Qwen3-VL | `Qwen/Qwen3-VL-8B-Instruct` | ✅ Ready |
| LLaVA | `lmms-lab/LLaVA-OneVision-1.5-8B-Instruct` | ✅ Ready |
| LLaVA | `lmms-lab/LLaVA-OneVision-1.5-4B-Instruct` | ✅ Ready |
| BAGEL | `ByteDance-Seed/BAGEL-7B-MoT` | ⚠️ Manual setup required |

---

## Quick Start Commands

### Single GPU Evaluation

```bash
# Evaluate one model on GPU 0
./run_single.sh Qwen/Qwen3-VL-4B-Instruct 0

# With custom dataset and samples
./run_single.sh Qwen/Qwen3-VL-8B-Instruct 1 4-room 50
```

### Multi-GPU Parallel Evaluation

```bash
# 2 models on 2 GPUs
./run_parallel.sh "Qwen/Qwen3-VL-4B-Instruct Qwen/Qwen3-VL-8B-Instruct" "0 1"

# 4 Qwen models on 4 GPUs
./run_parallel.sh \
  "Qwen/Qwen3-VL-4B-Instruct Qwen/Qwen3-VL-4B-Thinking Qwen/Qwen3-VL-8B-Thinking Qwen/Qwen3-VL-8B-Instruct" \
  "0 1 2 3"

# All 6 models (except BAGEL) on 6 GPUs
./run_parallel.sh \
  "Qwen/Qwen3-VL-4B-Instruct Qwen/Qwen3-VL-4B-Thinking Qwen/Qwen3-VL-8B-Thinking Qwen/Qwen3-VL-8B-Instruct lmms-lab/LLaVA-OneVision-1.5-8B-Instruct lmms-lab/LLaVA-OneVision-1.5-4B-Instruct" \
  "0 1 2 3 4 5"
```

---

## Files Overview

| File | Purpose |
|------|---------|
| `evaluate_vlm.py` | Main Python evaluation script |
| `run_single.sh` | Single-GPU evaluation wrapper |
| `run_parallel.sh` | Multi-GPU parallel evaluation wrapper |
| `EVALUATION_GUIDE.md` | Complete documentation (Chinese) |
| `QUICK_REFERENCE.md` | This file |

---

## Output Structure

```
results/
└── Qwen_Qwen3-VL-4B-Instruct/
    ├── env_data.html                    # Visualization homepage
    └── <room_hash>/
        ├── vision/
        │   ├── active/think/
        │   │   ├── config.json          # Configuration
        │   │   ├── exploration.json     # Exploration history
        │   │   ├── evaluation.json      # Evaluation results
        │   │   └── images/              # Visualizations
        │   └── passive/think/scout/
        │       └── ...
        └── text/
            ├── active/think/
            └── passive/think/strategist/
```

---

## View Results

```bash
# Start HTTP server
python -m http.server 8000

# Open in browser:
# http://localhost:8000/results/<model_name>/env_data.html
```

---

## Common Parameters

### run_single.sh / run_parallel.sh

```bash
./run_single.sh <model> <gpu> [dataset] [samples]
./run_parallel.sh "<models>" "<gpus>" [dataset] [samples]
```

- **dataset**: `3-room` (default) or `4-room`
- **samples**: Number of samples (default: 25)

### evaluate_vlm.py

```bash
python evaluate_vlm.py \
    --model_path <model>          # Required
    --gpu_id <id>                 # GPU ID (default: 0)
    --dataset_subset <subset>     # 3-room (default) or 4-room
    --num_samples <n>             # Number of samples (default: 25)
    --output_dir <dir>            # Output directory (default: results)
    --render_mode <mode>          # vision, text, or vision,text (default)
    --exp_type <type>             # active, passive, or active,passive (default)
    --max_exp_steps <steps>       # Max exploration steps (default: 20)
    --inference_mode <mode>       # direct (default) or batch
    --vllm_port <port>            # vLLM server port (default: 9999)
```

---

## Troubleshooting

### Port Conflict
```bash
# Kill process using port 9999
lsof -ti:9999 | xargs kill -9

# Or use different port
python evaluate_vlm.py --vllm_port 10000 ...
```

### Out of Memory
- Use smaller models (4B instead of 8B)
- Reduce `--num_samples`
- Use model parallelism
- Lower `--max_model_len` in vLLM config

### Dataset Not Found
```bash
# Check dataset exists
ls room_data/3-room/
ls room_data/4-room/
```

### Permission Denied
```bash
chmod +x run_single.sh run_parallel.sh
```

### vLLM Server Failed to Start
- Check GPU memory: `nvidia-smi`
- Check CUDA version compatibility
- Review vLLM logs for specific errors
- Try lowering `--gpu-memory-utilization`

---

## Time Estimates

| Configuration | Per Sample | 25 Samples |
|---------------|------------|------------|
| 3-room, active, vision | ~5-10 min | ~2-4 hours |
| 3-room, passive, vision | ~2-3 min | ~1-1.5 hours |
| 4-room, active, vision | ~10-20 min | ~4-8 hours |
| Text mode | ~50% faster | - |

---

## Advanced Usage

### Evaluate Specific Mode Only

```bash
# Vision only
python evaluate_vlm.py --render_mode vision ...

# Text only
python evaluate_vlm.py --render_mode text ...

# Active exploration only
python evaluate_vlm.py --exp_type active ...

# Passive understanding only
python evaluate_vlm.py --exp_type passive ...
```

### Background Execution with tmux

```bash
# Create session
tmux new -s tos_eval

# Run evaluation
./run_parallel.sh "model1 model2" "0 1"

# Detach: Ctrl+B, D
# Reattach: tmux attach -t tos_eval
```

### Monitor Progress

```bash
# Single model
tail -f logs/<model_name>_gpu<id>_<timestamp>.log

# Multiple models (separate terminals)
tail -f logs/Qwen_Qwen3_VL_4B_Instruct_gpu0_*.log
tail -f logs/Qwen_Qwen3_VL_8B_Instruct_gpu1_*.log
```

---

## Example Workflows

### Workflow 1: Quick Test (1 GPU)
```bash
# Test with small sample size first
./run_single.sh Qwen/Qwen3-VL-4B-Instruct 0 3-room 5

# If successful, run full evaluation
./run_single.sh Qwen/Qwen3-VL-4B-Instruct 0 3-room 25
```

### Workflow 2: Full Qwen Family (4 GPUs)
```bash
./run_parallel.sh \
  "Qwen/Qwen3-VL-4B-Instruct Qwen/Qwen3-VL-4B-Thinking Qwen/Qwen3-VL-8B-Thinking Qwen/Qwen3-VL-8B-Instruct" \
  "0 1 2 3"
```

### Workflow 3: Compare Qwen vs LLaVA (2 GPUs)
```bash
# Run both families in parallel
./run_parallel.sh \
  "Qwen/Qwen3-VL-4B-Instruct lmms-lab/LLaVA-OneVision-1.5-4B-Instruct" \
  "0 1"
```

### Workflow 4: Multi-Dataset Evaluation
```bash
# Evaluate on both 3-room and 4-room
./run_single.sh Qwen/Qwen3-VL-4B-Instruct 0 3-room 25 &
./run_single.sh Qwen/Qwen3-VL-4B-Instruct 1 4-room 25 &
wait
```

---

## Interpreting Results

### Key Metrics in evaluation.json

```json
{
  "results": [
    {
      "task": "direction",           // Task type
      "accuracy": 0.65,               // Success rate
      "avg_steps": 15.2,              // Average exploration steps
      "completion_rate": 0.92         // Completion percentage
    }
  ]
}
```

### Task Types

| Task | Category | Description |
|------|----------|-------------|
| `dir` | Route | Direction prediction |
| `pov` | Route | Perspective taking |
| `fwd_fov` | Route | Forward field of view |
| `e2a` | Route | View to action mapping |
| `rot` | Survey | Mental rotation |
| `fwd_loc` | Survey | Allocentric mapping |
| `bwd_loc_text` | Survey | Coordinate inference |

---

## Performance Optimization

1. **Use SSD for results**: `--output_dir /path/to/ssd/results`
2. **Parallel evaluation**: Use `run_parallel.sh` with multiple GPUs
3. **Background execution**: Use tmux/screen for long-running jobs
4. **Batch mode**: Use `--inference_mode batch` for API-based models
5. **Reduce samples**: Start with fewer samples for testing

---

## Hardware Requirements

### Minimum (per model)
- **GPU**: 1x 24GB (e.g., RTX 3090, A5000)
- **RAM**: 32GB
- **Storage**: 100GB free space
- **Network**: Fast internet for model downloads

### Recommended (for parallel evaluation)
- **GPU**: 4-8x 40GB+ (e.g., A100, H100)
- **RAM**: 128GB+
- **Storage**: 500GB+ SSD
- **Network**: High-speed connection

---

## Support

- **Full Documentation**: See `EVALUATION_GUIDE.md`
- **Theory-of-Space**: [GitHub](https://github.com/mll-lab-nu/Theory-of-Space)
- **Issues**: Check [GitHub Issues](https://github.com/mll-lab-nu/Theory-of-Space/issues)

---

**Last Updated**: 2026-03-17
