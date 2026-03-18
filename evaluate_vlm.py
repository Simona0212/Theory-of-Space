#!/usr/bin/env python3
"""
Unified VLM Evaluation Script for Theory-of-Space Benchmark

Supports 7 Vision-Language Models:
- Qwen/Qwen3-VL-4B-Instruct
- Qwen/Qwen3-VL-4B-Thinking
- Qwen/Qwen3-VL-8B-Thinking
- Qwen/Qwen3-VL-8B-Instruct
- lmms-lab/LLaVA-OneVision-1.5-8B-Instruct
- lmms-lab/LLaVA-OneVision-1.5-4B-Instruct
- ByteDance-Seed/BAGEL-7B-MoT

Usage:
    python evaluate_vlm.py --model_path Qwen/Qwen3-VL-4B-Instruct --output_dir results --gpu_id 0
"""

import argparse
import os
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime

# Model families for dynamic loading
QWEN_MODELS = [
    "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen/Qwen3-VL-4B-Thinking",
    "Qwen/Qwen3-VL-8B-Thinking",
    "Qwen/Qwen3-VL-8B-Instruct",
]

LLAVA_MODELS = [
    "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct",
    "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct",
]

BAGEL_MODELS = [
    "ByteDance-Seed/BAGEL-7B-MoT",
]

ALL_SUPPORTED_MODELS = QWEN_MODELS + LLAVA_MODELS + BAGEL_MODELS


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VLMs on Theory-of-Space Benchmark")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Hugging Face model path (e.g., Qwen/Qwen3-VL-4B-Instruct)")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Base output directory for results")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU device ID to use")
    parser.add_argument("--dataset_subset", type=str, default="3-room",
                        help="Dataset subset to evaluate (e.g., 3-room, 4-room)")
    parser.add_argument("--num_samples", type=int, default=25,
                        help="Number of samples to evaluate (default: 25)")
    parser.add_argument("--render_mode", type=str, default="vision,text",
                        help="Render modes to use: vision, text, or both (comma-separated)")
    parser.add_argument("--exp_type", type=str, default="active,passive",
                        help="Experiment types: active, passive, or both (comma-separated)")
    parser.add_argument("--vllm_port", type=int, default=9999,
                        help="Port for vLLM server (if using local serving)")
    parser.add_argument("--max_exp_steps", type=int, default=20,
                        help="Maximum exploration steps (default: 20)")
    parser.add_argument("--inference_mode", type=str, default="direct", choices=["direct", "batch"],
                        help="Inference mode: direct or batch (default: direct)")

    return parser.parse_args()


def sanitize_model_name(model_path: str) -> str:
    """Convert model path to filesystem-safe directory name."""
    return model_path.replace("/", "_")


def get_model_family(model_path: str) -> str:
    """Determine which model family the model belongs to."""
    if model_path in QWEN_MODELS:
        return "qwen"
    elif model_path in LLAVA_MODELS:
        return "llava"
    elif model_path in BAGEL_MODELS:
        return "bagel"
    else:
        raise ValueError(f"Unsupported model: {model_path}. Supported models: {ALL_SUPPORTED_MODELS}")


def setup_vllm_server(model_path: str, port: int, gpu_id: int):
    """Start vLLM server for the model."""
    model_name = sanitize_model_name(model_path)

    # Qwen and LLaVA models can use vLLM
    if get_model_family(model_path) in ["qwen", "llava"]:
        cmd = [
            "vllm", "serve", model_path,
            "--host", "0.0.0.0",
            "--port", str(port),
            "--dtype", "bfloat16",
            "--served-model-name", model_name,
            "--max_model_len", "128000",
            "--tensor-parallel-size", "1",
            "--gpu-memory-utilization", "0.9",
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        print(f"Starting vLLM server for {model_path} on GPU {gpu_id}, port {port}...")
        print(f"Command: {' '.join(cmd)}")

        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # Wait for server to be ready
        print("Waiting for vLLM server to start...")
        max_wait = 300  # 5 minutes
        start_time = time.time()

        while time.time() - start_time < max_wait:
            try:
                import requests
                response = requests.get(f"http://localhost:{port}/health", timeout=2)
                if response.status_code == 200:
                    print(f"✓ vLLM server ready on port {port}")
                    return proc, model_name
            except:
                pass
            time.sleep(5)

        raise RuntimeError(f"vLLM server failed to start within {max_wait}s")

    return None, None


def setup_bagel_server(model_path: str, port: int, gpu_id: int):
    """Start BAGEL server with OpenAI-compatible API."""
    model_name = sanitize_model_name(model_path)

    # Check for BAGEL-specific conda environment
    bagel_env_python = os.path.expanduser("~/miniconda3/envs/bagel/bin/python")
    if os.path.exists(bagel_env_python):
        print("Using BAGEL-specific conda environment")
        python_exec = bagel_env_python
    else:
        print("Using current Python environment")
        python_exec = sys.executable

    cmd = [
        python_exec,
        "bagel_server.py",
        "--model-path", model_path,
        "--port", str(port),
        "--host", "0.0.0.0",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"Starting BAGEL server for {model_path} on GPU {gpu_id}, port {port}...")
    print(f"Command: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # Wait for server to be ready
    print("Waiting for BAGEL server to start...")
    max_wait = 600  # 10 minutes (BAGEL takes longer to load)
    start_time = time.time()

    while time.time() - start_time < max_wait:
        try:
            import requests
            response = requests.get(f"http://localhost:{port}/health", timeout=2)
            if response.status_code == 200:
                print(f"✓ BAGEL server ready on port {port}")
                return proc, model_name
        except:
            pass
        time.sleep(5)

    raise RuntimeError(f"BAGEL server failed to start within {max_wait}s")


def update_model_config(model_path: str, served_model_name: str, port: int):
    """Update base_model_config.yaml with the new model configuration."""
    config_path = Path("scripts/SpatialGym/base_model_config.yaml")

    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Add model configuration
    model_key = served_model_name.replace("/", "_").replace("-", "_").lower()

    config['models'][model_key] = {
        'provider': 'openai',
        'organization': 'self-hosted',
        'model_name': served_model_name,
        'base_url': f'http://localhost:{port}/v1',
        'max_completion_tokens': 8192,
        'temperature': 0.0,
        'max_workers': 16,
        'max_retries': 3,
        'timeout': 600,
    }

    # Save updated config
    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f, sort_keys=False)

    print(f"✓ Updated model config: {model_key}")
    return model_key


def run_spatial_gym_evaluation(
    model_key: str,
    output_dir: str,
    dataset_subset: str,
    num_samples: int,
    render_mode: str,
    exp_type: str,
    max_exp_steps: int,
    inference_mode: str,
):
    """Run the Theory-of-Space evaluation pipeline."""
    data_dir = f"room_data/{dataset_subset}/"

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    cmd = [
        sys.executable, "-u",
        "scripts/SpatialGym/spatial_run.py",
        "--phase", "all",
        "--model-name", model_key,
        "--num", str(num_samples),
        "--data-dir", data_dir,
        "--output-root", output_dir,
        "--render-mode", render_mode,
        "--exp-type", exp_type,
        "--max-exp-steps", str(max_exp_steps),
        "--inference-mode", inference_mode,
    ]

    print("\n" + "="*80)
    print("Running Theory-of-Space Evaluation")
    print("="*80)
    print(f"Model: {model_key}")
    print(f"Dataset: {data_dir}")
    print(f"Samples: {num_samples}")
    print(f"Render modes: {render_mode}")
    print(f"Experiment types: {exp_type}")
    print(f"Command: {' '.join(cmd)}")
    print("="*80 + "\n")

    result = subprocess.run(
        cmd,
        check=True,
        cwd=os.getcwd(),
        env=os.environ.copy()
    )

    if result.returncode == 0:
        print(f"\n✓ Evaluation completed successfully!")
        print(f"Results saved to: {output_dir}/{model_key}/")
    else:
        print(f"\n✗ Evaluation failed with return code {result.returncode}")
        sys.exit(result.returncode)


def main():
    args = parse_args()

    # Validate model
    if args.model_path not in ALL_SUPPORTED_MODELS:
        print(f"Error: Unsupported model '{args.model_path}'")
        print(f"\nSupported models:")
        print("\nQwen Models:")
        for m in QWEN_MODELS:
            print(f"  - {m}")
        print("\nLLaVA Models:")
        for m in LLAVA_MODELS:
            print(f"  - {m}")
        print("\nBAGEL Models:")
        for m in BAGEL_MODELS:
            print(f"  - {m}")
        sys.exit(1)

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    print(f"Using GPU: {args.gpu_id}")

    server_proc = None
    try:
        # Determine model family and start appropriate server
        model_family = get_model_family(args.model_path)

        if model_family == "bagel":
            # Start BAGEL server
            server_proc, served_model_name = setup_bagel_server(
                args.model_path,
                args.vllm_port,
                args.gpu_id
            )
        else:
            # Start vLLM server for Qwen/LLaVA models
            server_proc, served_model_name = setup_vllm_server(
                args.model_path,
                args.vllm_port,
                args.gpu_id
            )

        # Update model configuration
        model_key = update_model_config(
            args.model_path,
            served_model_name,
            args.vllm_port
        )

        # Run evaluation
        run_spatial_gym_evaluation(
            model_key=model_key,
            output_dir=args.output_dir,
            dataset_subset=args.dataset_subset,
            num_samples=args.num_samples,
            render_mode=args.render_mode,
            exp_type=args.exp_type,
            max_exp_steps=args.max_exp_steps,
            inference_mode=args.inference_mode,
        )

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)

    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        # Cleanup server (vLLM or BAGEL)
        if server_proc:
            print("\nStopping server...")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_proc.kill()
            print("✓ Server stopped")


if __name__ == "__main__":
    main()
