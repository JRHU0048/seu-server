# Multi-Agent FGVC — 细粒度图像视角生成

基于 Qwen-Image-Edit 系列模型 + Qwen VLM，从细粒度分类数据集（CUB / Cars / Dogs / NABirds）中生成新视角图像，通过 **多 Agent 协作** 实现质量可控的迭代式生成。

---

## 项目结构

```
├── core/                          # Multi-Agent 框架核心
│   ├── agent.py                   #   BaseAgent 基类（load/run/unload 生命周期）
│   ├── generator.py               #   GeneratorAgent — Qwen-Image-Edit-2511
│   ├── critic.py                  #   CriticAgent — Qwen VLM 结构化评估
│   ├── refiner.py                 #   RefinerAgent — 评估反馈 → prompt 改进
│   ├── memory.py                  #   MemoryModule — 经验记忆 / 断点续跑
│   └── orchestrator.py            #   OrchestratorAgent — Pipeline 编排器
├── tasks/                         # 任务配置（可插拔）
│   ├── base.py                    #   TaskConfig 基类
│   ├── cub_bird.py                #   CUB-200-2011 (200 classes)
│   ├── stanford_car.py            #   Stanford Cars (196 cls, TODO)
│   ├── stanford_dog.py            #   Stanford Dogs (120 cls, TODO)
│   └── nabird.py                  #   NABirds (555 cls, TODO)
├── experiments/                   # 实验入口
│   ├── run_pipeline.py            #   CLI 入口（参数可配）
│   └── run_baselines.py           #   基线对比（Phase 2）
├── output/                        # 输出
│   ├── logs/                      #   运行日志
│   └── memory/                    #   经验记忆（JSON）
├── edit_single.py                 # [保留] 旧版单图脚本
├── edit_multi.py                  # [保留] 旧版三图脚本
├── pipeline.py                    # [保留] 旧版 pipeline
├── tools.py                       # 共享工具函数
├── prompts.py                     # 共享 prompt 库
├── archive/                       # 废弃实验版本
├── requirements.txt
└── README.md
```

## Multi-Agent 架构

```
                         Orchestrator
                    ┌──────────────────────┐
                    │  流程控制 / 重试决策   │
                    └──────────┬───────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         ▼                     ▼                     ▼
  ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐
  │  Generator   │   │   Critic     │   │    Memory        │
  │ Agent        │──▶│   Agent      │──▶│    Module        │
  │ Qwen-Image-  │   │  Qwen VLM    │   │  经验积累 /      │
  │ Edit-2511    │   │  结构化打分   │   │  断点续跑        │
  └──────────────┘   └──────────────┘   └──────────────────┘
         │                   │
         └───────────────────┘
             Refiner Agent
        (反馈 → prompt 改进)
```

### Agent 职责

| Agent | 模型 | 职责 |
|-------|------|------|
| **Generator** | Qwen-Image-Edit-2511 | 根据参考图 + prompt 生成新视角 |
| **Critic** | Qwen2.5-VL 系列 | 多维度评分 (0-10)，输出结构化反馈 |
| **Refiner** | 规则 + LLM (可选) | 将反馈翻译为 prompt 改进指令 |
| **Orchestrator** | 无（控制逻辑） | 协调调度、重试决策、统计记录 |
| **Memory** | 无（JSON 持久化） | 记录历史、断点续跑、同类经验迁移 |

### 生成循环

```
对每张图片:
  ┌─ Generation ──→ Evaluation ──→ Score ≥ THRESHOLD? ──Yes──→ 保存
  │    ↑                             │No
  │    │   retries < MAX_RETRIES?    │
  │    └──── Refiner 注入反馈 ────────┘
  │          No → 跳过
  └── Memory 记录经验 ──→ 下一张
```

## 快速开始

### 环境安装

```bash
conda create -n qwenedit python=3.10 -y
conda activate qwenedit

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate safetensors sentencepiece pillow opencv-python
pip install git+https://github.com/huggingface/diffusers
pip install -U cache-dit
```

### 运行

```bash
# CUB 数据集（默认，无 Critic，所有生成直接接受）
CUDA_VISIBLE_DEVICES=2,3 python experiments/run_pipeline.py

# 指定参数
CUDA_VISIBLE_DEVICES=2,3 python experiments/run_pipeline.py \
    --task cub_bird \
    --threshold 7.0 \
    --max-retries 3 \
    --use-critic \
    --skip-upto 6

# 查看所有可用任务
python experiments/run_pipeline.py --list-tasks
```

### CLI 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--task` | `cub_bird` | 任务配置 |
| `--input` | (任务默认) | 覆盖输入路径 |
| `--output` | (任务默认) | 覆盖输出路径 |
| `--threshold` | 7.0 | 最低接受分数 |
| `--max-retries` | 3 | 最大重试次数 |
| `--use-critic` | False | 启用 Critic 评估 |
| `--gen-model` | (任务默认) | 覆盖生成模型路径 |
| `--eval-model` | (任务默认) | 覆盖评估模型路径 |
| `--skip-upto` | 0 | 跳过前 N 个类别 |

## 添加新数据集

继承 `TaskConfig` 并注册即可：

```python
from tasks import register
from tasks.base import TaskConfig

@register("my_dataset")
@dataclass
class MyDatasetConfig(TaskConfig):
    name: str = "my_dataset"
    category_name: str = "car"
    input_root: str = "/path/to/data"
    output_root: str = "/path/to/output"
    num_classes: int = 100

    # 定义输入迭代策略
    def iter_inputs(self, image_paths):
        for i in range(len(image_paths)):
            yield {"name": ..., "paths": [...]}

    # 定义生成 prompt
    def build_prompt(self, feedback_history=None) -> str:
        return self.gen_prompt_template.format(...)
```

实现后放入 `tasks/` 目录，在 `tasks/__init__.py` 中导入即可自动注册。

## 模型对比

| 脚本 | 模型 | 说明 |
|------|------|------|
| `edit_single.py` | Qwen-Image-Edit | 单图输入，独立脚本 |
| `edit_multi.py` | Qwen-Image-Edit-2511 | 三图输入，独立脚本 |
| `core/orchestrator.py` | 多 Agent | 生成 + 评估 + 反馈循环 + 记忆 |
| `experiments/run_pipeline.py` | CLI | 上述框架的命令行入口 |

## 推理参数

| 参数 | 作用 | 推荐值 |
|------|------|--------|
| `num_inference_steps` | 推理步数 | 28～50 |
| `true_cfg_scale` | 提示词强度 | 3.5～7 |
| `SEED` | 随机种子 | 42 |
| `SIZE` | 输入尺寸 | 384px |
| `THRESHOLD` | 最低接受分 | 7.0 |
| `MAX_RETRIES` | 最大重试 | 3 |

---

## 三端同步

```
服务器 ↔ GitHub ↔ 本地
```

```bash
git pull && ...修改... && git add . && git commit -m "..." && git push
```
