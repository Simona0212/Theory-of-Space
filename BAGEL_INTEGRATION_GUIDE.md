# BAGEL Integration Guide

## 概述

本指南说明如何在 Theory-of-Space 基准测试中评估 BAGEL-7B-MoT 模型。BAGEL 模型通过 `bagel_server.py` 包装器提供 OpenAI 兼容的 API，与其他 VLM 模型（Qwen、LLaVA）使用相同的评估流程。

## 架构

```
Theory-of-Space 评估流程
    ↓
evaluate_vlm.py
    ├─ Qwen/LLaVA → 启动 vLLM 服务器 → OpenAI API
    └─ BAGEL → 启动 bagel_server.py → OpenAI 兼容 API
           ↓
    spatial_run.py 调用 llm_inference.py
           ↓
    ModelFactory.create(config) → OpenAIModelInterface
           ↓
    发送 HTTP 请求到 localhost:9999/v1/chat/completions
           ↓
    bagel_server.py 接收请求
           ├─ 解析 OpenAI 格式
           ├─ 转换为 BAGEL 格式
           ├─ 调用 model.generate()
           ├─ 处理 <think> 标签
           └─ 返回 OpenAI 格式响应
```

## 前置条件

### 1. 安装 Theory-of-Space

```bash
cd /path/to/Theory-of-Space
pip install -e .
```

### 2. 准备 BAGEL 依赖

**选项 A：统一环境（推荐先尝试）**

```bash
cd Bagel
pip install -r requirements.txt
pip install flask
```

**选项 B：独立环境（如果有依赖冲突）**

```bash
# 创建 BAGEL 专用环境
conda create -n bagel python=3.10 -y
conda activate bagel

cd Bagel
pip install -r requirements.txt
pip install flask

# evaluate_vlm.py 会自动检测并使用 ~/miniconda3/envs/bagel/bin/python
```

### 3. 下载 BAGEL 模型

```bash
# 方法 1：使用 Hugging Face CLI
huggingface-cli download ByteDance-Seed/BAGEL-7B-MoT --local-dir /path/to/BAGEL-7B-MoT

# 方法 2：使用 Python
python -c "
from huggingface_hub import snapshot_download
snapshot_download('ByteDance-Seed/BAGEL-7B-MoT', local_dir='/path/to/BAGEL-7B-MoT')
"
```

### 4. 准备数据集

```bash
cd /path/to/Theory-of-Space
source setup.sh
```

验证数据集：
```bash
ls room_data/3-room/run00/
# 应该看到 scene.json, trajectory.json 等文件
```

## 使用方法

### 单模型评估

```bash
# 评估 BAGEL 模型（1 个样本测试）
python evaluate_vlm.py \
    --model_path ByteDance-Seed/BAGEL-7B-MoT \
    --gpu_id 0 \
    --dataset_subset 3-room \
    --num_samples 1 \
    --output_dir test_results

# 完整评估（25 个样本）
python evaluate_vlm.py \
    --model_path ByteDance-Seed/BAGEL-7B-MoT \
    --gpu_id 0 \
    --dataset_subset 3-room \
    --num_samples 25 \
    --output_dir results
```

### 使用 Shell 脚本

**单卡单模型运行：**

```bash
./run_single.sh ByteDance-Seed/BAGEL-7B-MoT 0 3-room
```

**多卡并行运行所有 7 个模型：**

```bash
./run_parallel.sh \
  "Qwen/Qwen3-VL-4B-Instruct Qwen/Qwen3-VL-4B-Thinking Qwen/Qwen3-VL-8B-Thinking Qwen/Qwen3-VL-8B-Instruct lmms-lab/LLaVA-OneVision-1.5-8B-Instruct lmms-lab/LLaVA-OneVision-1.5-4B-Instruct ByteDance-Seed/BAGEL-7B-MoT" \
  "0 1 2 3 4 5 6"
```

## 输出结构

评估结果保存在以下目录结构中：

```
results/
└── ByteDance-Seed_BAGEL-7B-MoT/
    └── 3-room/
        ├── evaluation_2026-03-18_12-00-00.json
        ├── metrics_2026-03-18_12-00-00.json
        └── env_data.html
```

## 故障排除

### 问题 1：BAGEL 服务器启动失败

**症状：** `BAGEL server failed to start within 600s`

**解决方案：**
1. 检查模型路径是否正确
2. 确认 GPU 内存足够（至少 16GB）
3. 查看服务器日志：
   ```bash
   python bagel_server.py --model-path /path/to/BAGEL-7B-MoT --port 9999
   ```

### 问题 2：依赖冲突

**症状：** `ImportError: cannot import name 'xxx'`

**解决方案：** 使用独立 conda 环境（见前置条件选项 B）

### 问题 3：<think> 标签出现在答案中

**症状：** `evaluation.json` 中的 `user_answer` 包含 `<think>...</think>`

**解决方案：** 这已经在 `bagel_server.py` 的 `extract_answer()` 函数中处理，如果仍然出现，请检查：
```python
def extract_answer(response: str) -> str:
    if '<think>' in response and '</think>' in response:
        return response.split('</think>')[-1].strip()
    return response.strip()
```

### 问题 4：图像加载失败

**症状：** `Error loading image: xxx`

**解决方案：**
1. 确认图像路径正确
2. 检查图像格式（支持 PNG、JPG）
3. 验证 PIL 可以打开图像：
   ```python
   from PIL import Image
   img = Image.open('/path/to/image.png')
   ```

## 性能说明

- **BAGEL 启动时间：** 约 5-10 分钟（比 vLLM 慢）
- **推理速度：** 比 vLLM 慢约 10-15%（Flask 开销）
- **内存占用：** 约 16-20GB GPU 内存

## 与其他模型的对比

| 特性 | Qwen/LLaVA | BAGEL |
|------|------------|-------|
| 服务器 | vLLM | bagel_server.py |
| 启动时间 | 1-2 分钟 | 5-10 分钟 |
| API 格式 | OpenAI 原生 | OpenAI 兼容 |
| 推理速度 | 快 | 中等 |
| 特殊处理 | 无 | 移除 <think> 标签 |

## 验证清单

在运行完整评估前，请确认：

- [ ] Theory-of-Space 已安装（`pip install -e .`）
- [ ] BAGEL 依赖已安装（`pip install -r Bagel/requirements.txt`）
- [ ] BAGEL 模型已下载
- [ ] 数据集已准备（`source setup.sh`）
- [ ] GPU 内存足够（至少 16GB）
- [ ] 单样本测试通过（`--num_samples 1`）

## 高级配置

### 自定义端口

```bash
python evaluate_vlm.py \
    --model_path ByteDance-Seed/BAGEL-7B-MoT \
    --vllm_port 8888 \
    --gpu_id 0
```

### 调整生成参数

修改 `bagel_server.py` 中的 `chat_completions()` 函数：

```python
output_ids = model.generate(
    input_ids=input_ids,
    pixel_values=pixel_values,
    new_token_ids=new_token_ids,
    max_length=max_tokens,
    do_sample=(temperature > 0),
    temperature=max(temperature, 0.01) if temperature > 0 else 1.0,
    top_p=0.9,  # 添加 top_p
    top_k=50,   # 添加 top_k
)
```

## 参考资料

- [BAGEL GitHub](https://github.com/ByteDance-Seed/Bagel)
- [BAGEL 评估文档](Bagel/EVAL.md)
- [Theory-of-Space 论文](paper.txt)
- [vLLM 文档](https://docs.vllm.ai/)

## 联系支持

如遇到问题，请：
1. 检查本指南的故障排除部分
2. 查看 `Bagel/EVAL.md`
3. 提交 Issue 到 Theory-of-Space 仓库
