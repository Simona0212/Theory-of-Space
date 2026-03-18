# 快速开始指南

## 一键运行（在 Linux 服务器上）

### 1. 安装和准备（首次运行）

```bash
# 进入项目目录
cd /path/to/Theory-of-Space

# 安装 Theory-of-Space
pip install -e .

# 安装 BAGEL 依赖
cd Bagel && pip install -r requirements.txt && pip install flask && cd ..

# 准备数据集
source setup.sh
```

### 2. 快速测试（1 个样本）

```bash
# 测试 BAGEL 模型
python evaluate_vlm.py \
    --model_path ByteDance-Seed/BAGEL-7B-MoT \
    --gpu_id 0 \
    --num_samples 1 \
    --output_dir test_results

# 检查结果
ls test_results/ByteDance-Seed_BAGEL-7B-MoT/3-room/
```

### 3. 完整评估（25 个样本）

```bash
# 单个模型
./run_single.sh ByteDance-Seed/BAGEL-7B-MoT 0 3-room

# 所有 7 个模型并行
./run_parallel.sh \
  "Qwen/Qwen3-VL-4B-Instruct Qwen/Qwen3-VL-4B-Thinking Qwen/Qwen3-VL-8B-Thinking Qwen/Qwen3-VL-8B-Instruct lmms-lab/LLaVA-OneVision-1.5-8B-Instruct lmms-lab/LLaVA-OneVision-1.5-4B-Instruct ByteDance-Seed/BAGEL-7B-MoT" \
  "0 1 2 3 4 5 6"
```

## 支持的模型

| 模型 | 服务器 | 启动时间 |
|------|--------|----------|
| Qwen/Qwen3-VL-4B-Instruct | vLLM | 1-2 分钟 |
| Qwen/Qwen3-VL-4B-Thinking | vLLM | 1-2 分钟 |
| Qwen/Qwen3-VL-8B-Thinking | vLLM | 1-2 分钟 |
| Qwen/Qwen3-VL-8B-Instruct | vLLM | 1-2 分钟 |
| lmms-lab/LLaVA-OneVision-1.5-8B-Instruct | vLLM | 1-2 分钟 |
| lmms-lab/LLaVA-OneVision-1.5-4B-Instruct | vLLM | 1-2 分钟 |
| ByteDance-Seed/BAGEL-7B-MoT | bagel_server.py | 5-10 分钟 |

## 常见问题

### Q: BAGEL 服务器启动失败？
```bash
# 手动启动查看错误
python bagel_server.py --model-path /path/to/BAGEL-7B-MoT --port 9999
```

### Q: 依赖冲突？
```bash
# 使用独立环境
conda create -n bagel python=3.10 -y
conda activate bagel
cd Bagel && pip install -r requirements.txt && pip install flask
```

### Q: 如何测试服务器？
```bash
# 终端 1：启动服务器
python bagel_server.py --model-path /path/to/BAGEL-7B-MoT --port 9999

# 终端 2：运行测试
python test_bagel_server.py --port 9999
```

## 输出位置

```
results/
└── ByteDance-Seed_BAGEL-7B-MoT/
    └── 3-room/
        ├── evaluation_*.json  # 详细评估结果
        ├── metrics_*.json     # 性能指标
        └── env_data.html      # 可视化报告
```

## 详细文档

- **完整指南：** `BAGEL_INTEGRATION_GUIDE.md`
- **实施总结：** `IMPLEMENTATION_SUMMARY.md`
- **测试脚本：** `test_bagel_server.py`

## 需要帮助？

1. 查看 `BAGEL_INTEGRATION_GUIDE.md` 的故障排除部分
2. 运行 `python test_bagel_server.py` 诊断问题
3. 检查服务器日志输出
