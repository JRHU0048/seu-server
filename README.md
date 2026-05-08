# Multi-Agent FGVC — 细粒度图像视角生成

基于 Qwen-Image-Edit-2511 生成模型 + Qwen VLM 评估模型，从细粒度分类数据集（CUB / Cars / Dogs / NABirds）中生成新视角图像，通过 **多 Agent 协作** 实现质量可控的迭代式生成。

---

## 项目结构

```
├── core/                          # Multi-Agent 框架核心
│   ├── agent.py                   #   BaseAgent 基类
│   ├── generator.py               #   GeneratorAgent（Qwen-Image-Edit-2511）
│   ├── critic.py                  #   CriticAgent — 结构化评估（Qwen VLM）
│   ├── refiner.py                 #   RefinerAgent — prompt 改进
│   ├── memory.py                  #   MemoryModule — 经验记忆
│   └── orchestrator.py            #   OrchestratorAgent + SerialPipelineOrchestrator
├── tasks/                         # 任务配置（可插拔）
│   ├── base.py                    #   TaskConfig 基类
│   ├── cub_bird.py                #   CUB-200-2011 (200 cls) ✅
│   ├── stanford_car.py            #   Stanford Cars (196 cls) 🔜
│   ├── stanford_dog.py            #   Stanford Dogs (120 cls) 🔜
│   └── nabird.py                  #   NABirds (555 cls) 🔜
├── experiments/                   # 实验系统
│   ├── run_pipeline.py            #   CLI 入口（支持两种模式）
│   ├── run_baselines.py           #   基线对比实验
│   ├── metrics.py                 #   指标计算（FID / CLIP / Identity）
│   ├── analysis.py                #   错误分类 / 报告生成
│   ├── visualize.py               #   论文级图表（5 类）
│   ├── paper_output.py            #   LaTeX / 统计检验 / 补充材料
│   └── configs/                   #   Baseline 配置
├── output/                        # 输出
│   ├── logs/                      #   运行日志
│   ├── memory/                    #   经验记忆
│   ├── metrics/                   #   指标缓存
│   ├── baselines/                 #   基线实验结果
│   ├── figures/                   #   图表
│   └── paper/                     #   论文材料
├── edit_single.py                 # 单图输入脚本（Qwen-Image-Edit-2511）
├── edit_multi.py                  # 三图输入脚本（Qwen-Image-Edit-2511）
├── pipeline.py                    # [保留] 旧版 pipeline
├── tools.py                       # 共享工具函数
├── prompts.py                     # 共享 prompt 库
├── archive/                       # 废弃实验版本
├── requirements.txt
└── README.md
```

## Multi-Agent 架构

### 并行模式（OrchestratorAgent）— 图片级循环

Generator 和 Critic 同时驻留显存，每张图独立完成生成→评估→反馈→重试的闭环：

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
  │  Agent       │──▶│   Agent      │──▶│    Module        │
  │ Qwen-Image-  │   │  Qwen VLM    │   │  经验积累 /      │
  │ Edit-2511    │   │  结构化打分   │   │  断点续跑        │
  └──────────────┘   └──────────────┘   └──────────────────┘
         │                   │
         └───────────────────┘
             Refiner Agent
        (反馈 → prompt 改进)
```

### 串行模式（SerialPipelineOrchestrator）— Agent 级多轮迭代

Generator 和 Critic **分时占用显存**，先统一生成所有图片，再统一评估，聚合跨图反馈后进入下一轮：

```
Round 1..N:
  ┌──────────────────────────────────────────────────────┐
  │  Phase 1: GENERATE 全部图片（仅 Generator 在显存）    │
  │  Phase 2: EVALUATE 全部生成图（仅 Critic 在显存）     │
  │  Phase 3: 聚合跨图反馈 → 更新 prompt                  │
  └──────────────────────────────────────────────────────┘
  │
  ▼ (下一轮，携带聚合反馈继续改进)
```

解决双模型同时推理的显存瓶颈，适用于 GPU 显存有限的场景。

### Agent 职责

| Agent | 模型 | 职责 |
|-------|------|------|
| **Generator** | Qwen-Image-Edit-2511 | 根据参考图 + prompt 生成新视角 |
| **Critic** | Qwen2.5-VL / Qwen3.5 系列 | 多维度评分 (0-10)，输出结构化反馈 |
| **Refiner** | 规则 + LLM (可选) | 将反馈翻译为 prompt 改进指令 |
| **Orchestrator** | 无（控制逻辑） | 协调调度、重试决策、统计记录 |
| **Memory** | 无（JSON 持久化） | 记录历史、断点续跑、同类经验迁移 |

### 并行模式 — 图片级生成循环

```
对每张图片:
  ┌─ Generation ──→ Evaluation ──→ Score ≥ THRESHOLD? ──Yes──→ 保存
  │    ↑                             │No
  │    │   retries < MAX_RETRIES?    │
  │    └──── Refiner 注入反馈 ────────┘
  │          No → 跳过
  └── Memory 记录经验 ──→ 下一张
```

### 串行模式 — Agent 级生成循环

```
Round 1:
  ┌─────────────────────────────────────────┐
  │  Generate ALL images (同一 prompt)       │  ← Generator 独占显存
  │  Evaluate ALL → collect issues           │  ← Critic 独占显存
  │  Aggregate top failure patterns          │
  └────────────┬────────────────────────────┘
               │ global_feedback
               ▼
Round 2:
  ┌─────────────────────────────────────────┐
  │  Regenerate ALL (带聚合反馈改进 prompt)   │
  │  Re-evaluate ALL → collect issues       │
  │  Aggregate → update global_feedback     │
  └────────────┬────────────────────────────┘
               ▼
         ...（直到达到指定轮数）
               
保存每张图片跨轮次的最佳结果 → *_best.png
```

### 并行模式重试策略

OrchestratorAgent 支持 3 种重试策略（通过 `retry_strategy` 配置）：

| 策略 | 行为 | 适用场景 |
|------|------|----------|
| `fixed`（默认） | 固定重试 N 次 | 标准评估 |
| `adaptive` | 连续 2 次分数下降则提前停止 | 节约资源 |
| `aggressive` | 每轮阈值降低 0.5 | 提高接受率 |

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
# 并行模式（默认，Generator + Critic 同时驻留显存）
CUDA_VISIBLE_DEVICES=2,3 python experiments/run_pipeline.py \
    --task cub_bird \
    --use-critic

# 串行模式（Generator 和 Critic 分时占用显存，推荐显存有限时使用）
CUDA_VISIBLE_DEVICES=2,3 python experiments/run_pipeline.py \
    --task cub_bird \
    --use-critic \
    --serial \
    --num-rounds 3

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

代码完善至 Phase 3，项目已达到完整状态：

- **Agent 框架**（`core/`）：BaseAgent → Generator → Critic → Refiner → Memory → Orchestrator
- **任务系统**（`tasks/`）：基类 + CUB ✅ + 3 个预留
- **实验系统**（`experiments/`）：Pipeline 运行 → 基线对比 → 指标计算 → 错误分析 → 可视化 → 论文输出
- **所有生成脚本统一使用 Qwen-Image-Edit-2511**

在服务器上 `git pull` 后按以下流程操作即可开展实验：

```
1. 单次运行:   python experiments/run_pipeline.py --task cub_bird --use-critic
2. 串行迭代:   python experiments/run_pipeline.py --task cub_bird --use-critic --serial --num-rounds 3
3. 基线对比:   python experiments/run_baselines.py --all
4. 指标计算:   python experiments/run_baselines.py --all --metrics-only
5. 可视化:     python experiments/visualize.py --memory <path> --baselines-dir <path>
6. 论文输出:   python experiments/paper_output.py --all --baselines-dir <path> --memory <path>
```

## 基线对比实验

运行 5 组对比实验，系统比较反馈循环的效果：

| Baseline | Critic | Retries | 目标 |
|----------|--------|---------|------|
| `no_feedback` | ✗ | 0 | 单次生成的基线质量 |
| `multi_attempt` | ✗ | 3 | 纯随机性（同 prompt 重复） |
| `critic_threshold_6` | ✓ | 3 | 低阈值效果 |
| `critic_threshold_7` | ✓ | 3 | **默认方法** |
| `critic_threshold_8` | ✓ | 3 | 高阈值效果 |

### 运行

```bash
# 查看实验计划（不实际运行）
python experiments/run_baselines.py --task cub_bird --dry-run

# 运行单个 baseline
CUDA_VISIBLE_DEVICES=2,3 python experiments/run_baselines.py \
    --baseline no_feedback

# 运行全部
CUDA_VISIBLE_DEVICES=2,3 python experiments/run_baselines.py --all

# 只计算指标（跳过生成，用于已有输出的重算）
CUDA_VISIBLE_DEVICES=2,3 python experiments/run_baselines.py \
    --all --metrics-only
```

### 指标说明

| 指标 | 来源 | 含义 | 方向 |
|------|------|------|------|
| **FID** | torchmetrics + InceptionV3 | 生成分布与真实分布的距离 | ↓ 低更好 |
| **CLIP Score** | openai/clip-vit-base-patch32 | 图片与任务 prompt 的对齐度 | ↑ 高更好 |
| **Identity Score** | CriticAgent 采样评估 | 主体身份保持程度 | ↑ 高更好 |
| **修复率** | Memory 错误统计 | 各类型错误被修复的比例 | ↑ 高更好 |

### 分析报告

实验完成后生成分析的 Markdown 报告：

```bash
# 错误分类统计
python experiments/analysis.py --memory output/baselines/cub_bird/critic_threshold_7/memory.json

# 完整报告
python experiments/analysis.py \
    --memory <path> \
    --baselines-dir output/baselines/cub_bird \
    --output output/report
```

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
| `edit_single.py` | Qwen-Image-Edit-2511 | 单图输入（与多图同模型，输入包装为 list） |
| `edit_multi.py` | Qwen-Image-Edit-2511 | 三图输入（滑动窗口） |
| `core/orchestrator.py` | 多 Agent | 生成 → 评估 → 反馈循环 + 记忆 |
| `experiments/run_pipeline.py` | CLI | 上述框架的命令行入口（支持并行/串行模式） |

所有生成脚本已统一为 **Qwen-Image-Edit-2511**，不再依赖旧版 `Qwen-Image-Edit`。

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
