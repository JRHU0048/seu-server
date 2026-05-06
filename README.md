# Qwen-Image-Edit — 鸟类图像视角生成

基于 Qwen-Image-Edit 系列模型，从 [CUB-200-2011](http://www.vision.caltech.edu/datasets/cub_200_2011/) 数据集中生成鸟类新视角图像（旋转、背景替换等）。

## 项目结构

```
├── pipeline.py                 # [新增] Multi-Agent Pipeline（生成 + 评估 + 反馈循环）
├── edit_single.py              # 单图输入 + Qwen-Image-Edit + cache-dit
├── edit_multi.py               # 三图输入 + Qwen-Image-Edit-2511 + cache-dit
├── prompts.py                  # 所有 prompt 统一管理
├── tools.py                    # 共享工具函数
├── archive/                    # 废弃的实验版本
│   ├── inferv1.py              # 单图 + 3 视角（单类别）
│   ├── inferv2.py              # 单图/三图双模式（单类别）
│   ├── inferv3.py              # 三图 2×2 网格（单类别）
│   └── inferv4.py              # 四图参考 + 仅 view_60（全类别）
├── requirements.txt
└── README.md
```

## 快速开始

### 1. 环境安装

```bash
conda create -n qwenedit python=3.10 -y
conda activate qwenedit

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate safetensors sentencepiece pillow opencv-python
pip install git+https://github.com/huggingface/diffusers   # 最新版 diffusers
pip install -U cache-dit                                    # cache-dit 加速
```

### 2. 下载模型

**方案 A：Hugging Face 直连**

```bash
pip install -U huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen-Image-Edit', local_dir='./Qwen-Image-Edit')"
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen-Image-Edit-2511', local_dir='./Qwen-Image-Edit-2511')"
```

**方案 B：国内镜像**

```bash
export HF_ENDPOINT=https://hf-mirror.com
# 永久生效
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc && source ~/.bashrc
```

### 3. 运行

```bash
# 单图输入（Qwen-Image-Edit）
CUDA_VISIBLE_DEVICES=2,3 python edit_single.py

# 三图输入（Qwen-Image-Edit-2511）
CUDA_VISIBLE_DEVICES=2,3 python edit_multi.py

# Multi-Agent Pipeline（生成 + 评估 + 反馈重试）
CUDA_VISIBLE_DEVICES=2,3 python pipeline.py
```

## 模型对比

| 脚本 | 模型 | 输入 | 输出 | 加速 |
|------|------|------|------|------|
| `edit_single.py` | Qwen-Image-Edit | 单张图片 | 60° 视角变换 | cache-dit |
| `edit_multi.py` | Qwen-Image-Edit-2511 | 三张图片（滑动窗口） | 60° 视角变换 + 背景替换 | cache-dit |
| `pipeline.py` | 双 Agent | 三张图片（滑动窗口） | 自适应：评估不通过则重试 | cache-dit |

## Multi-Agent Pipeline

`pipeline.py` 搭建了一套双 Agent 协作流程：

```
Agent 1 (Generator):  Qwen-Image-Edit-2511 → 生成新视角
       ↓
Agent 2 (Evaluator):  Qwen VLM → 量化评分 (0~10) + 指出问题
       ↓
Score ≥ 7.0?  ──Yes──→ 保存结果，继续下一张
       │No
  重试 < 3次? ──Yes──→ 将评估意见注入 prompt → 重新生成
       │No
       └──→ 跳过当前图片
```

**关键参数**（脚本顶部修改）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `THRESHOLD` | 7.0 | 最低接受分数 |
| `MAX_RETRIES` | 3 | 最大重试次数 |
| `EVAL_MODEL_ID` | `../Qwen2.5-VL-7B-Instruct` | 评估模型路径 |

> **注意**：评估模型需要额外 ~16GB 显存。如果显存不足，可在 `run_pipeline()` 中将 `evaluator_model, evaluator_proc = load_evaluator()` 这行注释掉，pipeline 会自动跳过评估（所有生成直接接受）。

## 推理参数说明

| 参数 | 作用 | 推荐值 |
|------|------|--------|
| `num_inference_steps` | 推理步数 | 28～50 |
| `true_cfg_scale` | 提示词强度 | 3.5～7 |
| `generator` (SEED) | 随机种子 | 42（固定可复现） |
| `negative_prompt` | 负面词 | 见 `prompts.py` |
| `torch_dtype` | 精度 | bfloat16 |
| `SIZE` | 输入图片尺寸 | 384px |

## 数据路径

默认配置在服务器 `/home/huyanhan/data/CUB_200_2011/`：

```
images/train/              ← 输入（按类别文件夹组织）
qwen_output/train/         ← 输出（自动镜像输入结构）
```

如需更换路径，直接修改脚本开头的 `INPUT_ROOT` / `OUTPUT_ROOT`。

## 备选 Prompt

`prompts.py` 中额外提供了以下实验性 prompt，可自行替换：

- **`PROMPT_BG_REPLACE`** — 仅替换背景（森林/草地/湿地等）
- **`PROMPT_VIEW_BG`** — 视角变换 + 背景替换联合任务

---

## 三端同步（服务器 ↔ GitHub ↔ 本地）

本项目通过 GitHub 在三端之间同步代码：

```
服务器 /home/huyanhan/FGVC/Qwenedit/src  ↔  GitHub  ↔  本地 D:\seu-server
```

### 日常同步流程

```bash
git pull          # 1. 拉取最新代码（修改前先执行）
# ... 修改文件 ...
git add .         # 2. 暂存所有修改
git commit -m "改动说明"  # 3. 提交
git push          # 4. 推送到 GitHub
```

### 注意事项

- **修改前先 `git pull`**，避免冲突
- **改完及时 `git push`**，否则另一端拉不到更新
- **两端同时改了同一文件** → 后 push 的一端先 `git pull` → 解决冲突 → 重新 `git add/commit/push`
