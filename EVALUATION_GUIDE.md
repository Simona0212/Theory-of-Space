# Theory-of-Space VLM Evaluation Pipeline

完整的自动化评估流程，用于在 Theory-of-Space 基准上评估 7 个视觉语言模型（VLMs）。

## 📋 目录

- [支持的模型](#支持的模型)
- [环境配置](#环境配置)
- [快速开始](#快速开始)
- [使用方法](#使用方法)
  - [单卡单模型评估](#单卡单模型评估)
  - [多卡并行评估](#多卡并行评估)
  - [Python 脚本直接调用](#python-脚本直接调用)
- [输出结构](#输出结构)
- [结果查看](#结果查看)
- [BAGEL 特殊说明](#bagel-特殊说明)
- [故障排除](#故障排除)

---

## 支持的模型

本评估流程支持以下 7 个模型：

### Qwen3-VL 系列 (4 个模型)
- `Qwen/Qwen3-VL-4B-Instruct`
- `Qwen/Qwen3-VL-4B-Thinking`
- `Qwen/Qwen3-VL-8B-Thinking`
- `Qwen/Qwen3-VL-8B-Instruct`

### LLaVA-OneVision 系列 (2 个模型)
- `lmms-lab/LLaVA-OneVision-1.5-8B-Instruct`
- `lmms-lab/LLaVA-OneVision-1.5-4B-Instruct`

### BAGEL 系列 (1 个模型)
- `ByteDance-Seed/BAGEL-7B-MoT` ⚠️ 需要特殊配置

---

## 环境配置

### 1. 基础环境

```bash
# 克隆仓库（如果还没有）
git clone --single-branch --branch release https://github.com/mll-lab-nu/Theory-of-Space.git
cd Theory-of-Space

# 安装依赖
pip install -r requirements.txt

# 安装 vLLM（用于模型推理加速）
pip install vllm

# 安装额外依赖
pip install pyyaml requests
```

### 2. 数据集准备

确保数据集目录存在：
```bash
ls room_data/3-room/  # 3-room 数据集
ls room_data/4-room/  # 4-room 数据集（如果需要）
```

### 3. GPU 检查

```bash
# 检查可用 GPU
nvidia-smi

# 确保有足够的 GPU 显存（建议每个模型至少 24GB）
```

---

## 快速开始

### 最简单的使用方式

评估单个模型（在 GPU 0 上）：
```bash
./run_single.sh Qwen/Qwen3-VL-4B-Instruct 0
```

并行评估多个模型（使用 GPU 0 和 1）：
```bash
./run_parallel.sh "Qwen/Qwen3-VL-4B-Instruct Qwen/Qwen3-VL-8B-Instruct" "0 1"
```

---

## 使用方法

### 单卡单模型评估

**脚本：** `run_single.sh`

**语法：**
```bash
./run_single.sh <model_name> <gpu_id> [dataset_subset] [num_samples]
```

**参数说明：**
- `model_name`: 模型名称（Hugging Face 路径）
- `gpu_id`: 使用的 GPU ID（0, 1, 2, ...）
- `dataset_subset`: 数据集子集，可选，默认 `3-room`
- `num_samples`: 评估样本数，可选，默认 `25`

**示例：**

```bash
# 示例 1：评估 Qwen3-VL-4B-Instruct（默认配置）
./run_single.sh Qwen/Qwen3-VL-4B-Instruct 0

# 示例 2：评估 LLaVA-OneVision-8B 在 4-room 数据集上
./run_single.sh lmms-lab/LLaVA-OneVision-1.5-8B-Instruct 1 4-room

# 示例 3：评估 Qwen3-VL-8B-Thinking，使用 50 个样本
./run_single.sh Qwen/Qwen3-VL-8B-Thinking 0 3-room 50

# 示例 4：在不同 GPU 上评估不同模型
./run_single.sh Qwen/Qwen3-VL-4B-Instruct 0  # GPU 0
./run_single.sh Qwen/Qwen3-VL-8B-Instruct 1  # GPU 1
```

**执行流程：**
1. 设置 CUDA 设备为指定 GPU
2. 启动 vLLM 服务器加载模型
3. 更新模型配置文件
4. 运行完整评估流程（exploration + evaluation + cogmap）
5. 生成可视化结果
6. 自动清理 vLLM 服务器

---

### 多卡并行评估

**脚本：** `run_parallel.sh`

**语法：**
```bash
./run_parallel.sh "<model1> <model2> ..." "<gpu1> <gpu2> ..." [dataset_subset] [num_samples]
```

**参数说明：**
- `models`: 空格分隔的模型列表（需要用引号包裹）
- `gpus`: 空格分隔的 GPU ID 列表（需要用引号包裹）
- 模型和 GPU 数量必须一致（一一对应）
- `dataset_subset`: 数据集子集，可选，默认 `3-room`
- `num_samples`: 评估样本数，可选，默认 `25`

**示例：**

```bash
# 示例 1：2 个模型并行评估（GPU 0 和 1）
./run_parallel.sh "Qwen/Qwen3-VL-4B-Instruct Qwen/Qwen3-VL-8B-Instruct" "0 1"

# 示例 2：3 个模型并行评估
./run_parallel.sh \
  "Qwen/Qwen3-VL-4B-Instruct lmms-lab/LLaVA-OneVision-1.5-8B-Instruct Qwen/Qwen3-VL-8B-Thinking" \
  "0 1 2"

# 示例 3：评估 4 个 Qwen 模型（使用 4 张 GPU）
./run_parallel.sh \
  "Qwen/Qwen3-VL-4B-Instruct Qwen/Qwen3-VL-4B-Thinking Qwen/Qwen3-VL-8B-Thinking Qwen/Qwen3-VL-8B-Instruct" \
  "0 1 2 3"

# 示例 4：在 4-room 数据集上评估，使用 50 个样本
./run_parallel.sh \
  "Qwen/Qwen3-VL-4B-Instruct Qwen/Qwen3-VL-8B-Instruct" \
  "0 1" \
  4-room \
  50

# 示例 5：评估所有 6 个模型（除 BAGEL）
./run_parallel.sh \
  "Qwen/Qwen3-VL-4B-Instruct Qwen/Qwen3-VL-4B-Thinking Qwen/Qwen3-VL-8B-Thinking Qwen/Qwen3-VL-8B-Instruct lmms-lab/LLaVA-OneVision-1.5-8B-Instruct lmms-lab/LLaVA-OneVision-1.5-4B-Instruct" \
  "0 1 2 3 4 5"
```

**执行流程：**
1. 验证模型和 GPU 数量是否匹配
2. 为每个模型在对应 GPU 上启动独立进程
3. 所有进程并行运行，互不干扰
4. 日志分别保存到 `logs/` 目录
5. 等待所有进程完成
6. 显示汇总结果

**日志监控：**

脚本会为每个模型生成独立的日志文件：
```bash
logs/Qwen_Qwen3_VL_4B_Instruct_gpu0_20260317_143022.log
logs/Qwen_Qwen3_VL_8B_Instruct_gpu1_20260317_143022.log
```

实时监控某个模型的评估进度：
```bash
tail -f logs/Qwen_Qwen3_VL_4B_Instruct_gpu0_20260317_143022.log
```

---

### Python 脚本直接调用

如果需要更细粒度的控制，可以直接调用 Python 脚本：

```bash
python evaluate_vlm.py \
    --model_path Qwen/Qwen3-VL-4B-Instruct \
    --gpu_id 0 \
    --dataset_subset 3-room \
    --num_samples 25 \
    --output_dir results \
    --render_mode vision,text \
    --exp_type active,passive \
    --max_exp_steps 20 \
    --inference_mode direct \
    --vllm_port 9999
```

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model_path` | str | **必需** | 模型路径（Hugging Face） |
| `--gpu_id` | int | 0 | GPU 设备 ID |
| `--dataset_subset` | str | 3-room | 数据集子集 |
| `--num_samples` | int | 25 | 评估样本数 |
| `--output_dir` | str | results | 输出目录 |
| `--render_mode` | str | vision,text | 渲染模式（vision/text/both） |
| `--exp_type` | str | active,passive | 实验类型（active/passive/both） |
| `--max_exp_steps` | int | 20 | 最大探索步数 |
| `--inference_mode` | str | direct | 推理模式（direct/batch） |
| `--vllm_port` | int | 9999 | vLLM 服务器端口 |

---

## 输出结构

评估结果按照以下严格的目录结构保存：

```
results/
└── Qwen_Qwen3-VL-4B-Instruct/           # 模型名称（/替换为_）
    ├── env_data.html                     # 可视化主页面
    ├── <room_hash_1>/                    # 房间环境哈希
    │   ├── vision/                       # 视觉模式
    │   │   ├── active/                   # 主动探索
    │   │   │   └── think/
    │   │   │       ├── config.json       # 配置文件
    │   │   │       ├── exploration.json  # 探索历史
    │   │   │       ├── evaluation.json   # 评估结果
    │   │   │       └── images/           # 可视化图片
    │   │   └── passive/                  # 被动推理
    │   │       └── think/
    │   │           └── scout/            # 代理类型
    │   │               ├── config.json
    │   │               ├── exploration.json
    │   │               ├── evaluation.json
    │   │               └── images/
    │   └── text/                         # 文本模式
    │       ├── active/
    │       │   └── think/
    │       │       └── ...
    │       └── passive/
    │           └── think/
    │               └── strategist/
    │                   └── ...
    └── <room_hash_2>/
        └── ...
```

**关键文件说明：**

- **`config.json`**: 环境配置和评估元数据
- **`exploration.json`**: 探索轨迹和动作历史
- **`evaluation.json`**: 评估指标和任务结果
- **`env_data.html`**: 交互式可视化界面
- **`images/`**: 环境渲染图片和认知地图

---

## 结果查看

### 方法 1：本地 HTTP 服务器（推荐）

```bash
# 在仓库根目录启动 HTTP 服务器
python -m http.server 8000

# 在浏览器中打开
# http://localhost:8000/results/Qwen_Qwen3-VL-4B-Instruct/env_data.html
```

**可视化界面功能：**
- 📊 每个样本的详细指标
- 🗺️ 认知地图可视化
- 📈 性能图表和趋势
- 🔍 探索轨迹回放
- 📝 任务级别分析

### 方法 2：直接读取 JSON 文件

```bash
# 查看某个样本的评估结果
cat results/Qwen_Qwen3-VL-4B-Instruct/<room_hash>/vision/active/think/evaluation.json | jq

# 提取特定指标
cat results/Qwen_Qwen3-VL-4B-Instruct/<room_hash>/vision/active/think/evaluation.json | \
    jq '.results[] | select(.task=="direction") | .accuracy'
```

### 方法 3：聚合分析

```python
import json
from pathlib import Path

def aggregate_results(model_dir):
    """聚合某个模型的所有评估结果"""
    results = []
    for eval_file in Path(model_dir).rglob("evaluation.json"):
        with open(eval_file) as f:
            data = json.load(f)
            results.append(data)
    return results

# 使用示例
model_results = aggregate_results("results/Qwen_Qwen3-VL-4B-Instruct")
print(f"Total samples: {len(model_results)}")
```

---

## BAGEL 特殊说明

⚠️ **BAGEL 模型需要特殊配置，当前脚本暂不直接支持。**

### 为什么 BAGEL 不同？

1. **不支持 vLLM**: BAGEL 使用自定义推理代码
2. **特殊依赖**: 需要安装 Bagel 仓库的特定依赖
3. **不同 API**: 推理接口与 Hugging Face 标准不同

### 手动评估 BAGEL

如需评估 BAGEL，请按照以下步骤：

1. **安装 BAGEL 依赖**
```bash
cd Bagel
pip install -r requirements.txt
pip install flash_attn==2.5.8 --no-build-isolation
```

2. **下载模型**
```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ByteDance-Seed/BAGEL-7B-MoT",
    local_dir="models/BAGEL-7B-MoT",
    cache_dir="models/BAGEL-7B-MoT/cache",
    local_dir_use_symlinks=False,
)
```

3. **参考 BAGEL 评估文档**
```bash
cd Bagel
cat EVAL.md  # 查看官方评估指南
```

4. **自定义集成**
 - 需要修改 `vagen/env/spatial/llm_inference.py` 来支持 BAGEL 的推理 API
 - 或者使用 BAGEL 的原生评估脚本，然后转换输出格式

---

## 故障排除

### 问题 1: vLLM 启动失败

**症状:**
```
RuntimeError: vLLM server failed to start within 300s
```

**解决方案:**
- 检查 GPU 显存是否足够（`nvidia-smi`）
- 尝试降低 `--gpu-memory-utilization` 参数
- 检查端口是否被占用（`lsof -i :9999`）
- 查看 vLLM 日志确认具体错误

### 问题 2: 端口冲突

**症状:**
```
Address already in use: 0.0.0.0:9999
```

**解决方案:**
```bash
# 方法 1: 杀死占用端口的进程
lsof -ti:9999 | xargs kill -9

# 方法 2: 使用不同端口
python evaluate_vlm.py --model_path ... --vllm_port 10000
```

### 问题 3: CUDA Out of Memory

**症状:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**解决方案:**
- 使用更小的模型（4B 而不是 8B）
- 减少 `--num_samples`
- 增加 GPU 或使用模型并行
- 降低 `--max_model_len` 参数

### 问题 4: 数据集未找到

**症状:**
```
FileNotFoundError: Dataset directory not found: room_data/3-room/
```

**解决方案:**
```bash
# 检查数据集是否存在
ls -la room_data/

# 如果没有，请按照 README.md 的说明准备数据集
```

### 问题 5: 依赖缺失

**症状:**
```
ModuleNotFoundError: No module named 'vllm'
```

**解决方案:**
```bash
# 重新安装依赖
pip install -r requirements.txt
pip install vllm pyyaml requests

# 如果使用 conda
conda install -c conda-forge pyyaml
pip install vllm requests
```

### 问题 6: 权限错误

**症状:**
```
Permission denied: './run_single.sh'
```

**解决方案:**
```bash
# 添加执行权限
chmod +x run_single.sh run_parallel.sh
```

---

## 高级用法

### 仅评估特定模式

```bash
# 仅评估 vision 模式
python evaluate_vlm.py \
    --model_path Qwen/Qwen3-VL-4B-Instruct \
    --render_mode vision \
    --gpu_id 0

# 仅评估 active 探索
python evaluate_vlm.py \
    --model_path Qwen/Qwen3-VL-4B-Instruct \
    --exp_type active \
    --gpu_id 0
```

### 批量推理模式

对于支持批量 API 的模型（如 OpenAI、Gemini）：

```bash
python evaluate_vlm.py \
    --model_path Qwen/Qwen3-VL-4B-Instruct \
    --inference_mode batch \
    --gpu_id 0
```

### 调整探索步数

```bash
# 更长的探索（更准确但更慢）
./run_single.sh Qwen/Qwen3-VL-4B-Instruct 0 3-room 25 30

# 在 Python 脚本中
python evaluate_vlm.py \
    --model_path Qwen/Qwen3-VL-4B-Instruct \
    --max_exp_steps 30 \
    --gpu_id 0
```

---

## 性能优化建议

### 1. 使用 SSD 存储结果
```bash
# 将输出目录设置到 SSD
python evaluate_vlm.py --output_dir /path/to/ssd/results ...
```

### 2. 并行评估多个数据集
```bash
# 在不同 GPU 上评估不同数据集
./run_single.sh Qwen/Qwen3-VL-4B-Instruct 0 3-room &
./run_single.sh Qwen/Qwen3-VL-8B-Instruct 1 4-room &
wait
```

### 3. 使用 tmux/screen 后台运行
```bash
# 创建 tmux 会话
tmux new -s tos_eval

# 运行评估
./run_parallel.sh "model1 model2" "0 1"

# 分离会话（Ctrl+B, D）
# 重新连接：tmux attach -t tos_eval
```

---

## 时间估算

评估时间取决于多个因素：

| 配置 | 每个样本 | 25 个样本 |
|------|----------|-----------|
| 3-room, active, vision | ~5-10 分钟 | ~2-4 小时 |
| 3-room, passive, vision | ~2-3 分钟 | ~1-1.5 小时 |
| 4-room, active, vision | ~10-20 分钟 | ~4-8 小时 |
| Text 模式 | 更快（~50%） | - |

**加速建议：**
- 使用多 GPU 并行
- 优先评估 passive 模式（更快）
- 使用 batch 推理模式（如果支持）

---

## 许可证

本评估流程基于 Theory-of-Space 基准，遵循其原始许可证。

## 致谢

- **Theory-of-Space 团队**: 提供了出色的空间推理基准
- **Qwen 团队**: Qwen3-VL 系列模型
- **LLaVA 团队**: LLaVA-OneVision 模型
- **ByteDance-Seed**: BAGEL 模型

---

## 联系方式

如有问题或建议，请：
1. 查看 [Theory-of-Space GitHub Issues](https://github.com/mll-lab-nu/Theory-of-Space/issues)
2. 参考各模型的官方文档
3. 检查本文档的故障排除部分

---

**最后更新**: 2026-03-17
