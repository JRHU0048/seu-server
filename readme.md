# 说明

## 三端同步（服务器 ↔ GitHub ↔ 本地）

此项目通过 GitHub 在服务器和本地之间同步：

```
服务器 /home/huyanhan/FGVC/Qwenedit/src  ↔  GitHub  ↔  本地 D:\seu-server
```

### 日常同步流程

无论在哪端修改代码，按此流程操作：

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

---

CUDA_VISIBLE_DEVICES=2,3 python inferv5.py
v1-v4 属于测试版本
v5 使用qwen-image-edit 实现了单图输入，视角切换，环境变换的功能
v6 使用qwen-image-edit-2511 实现了多图输入的功能


# Qwen-Image-Edit 简介

Alibaba Group 发布的 Qwen-Image-Edit 是基于 20B 参数的图像编辑模型，支持：

### 核心能力

### ① 精准图像编辑

* 改颜色
* 换背景
* 删除物体
* 增加物体
* 局部重绘

### ② 文字编辑

* 修改海报文字
* 改中文招牌
* 修复错字

### ③ 高级语义编辑

* 视角旋转
* 风格迁移
* IP角色一致性生成


# 环境安装

## 1. 创建环境

```bash
conda create -n qwenedit python=3.10 -y
conda activate qwenedit
```

## 2. 安装 PyTorch（CUDA 12.1）

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

验证：
```bash
python -c "import torch;print(torch.cuda.is_available())"
```


## 3. 安装依赖

```bash
python -m pip install diffusers transformers accelerate safetensors sentencepiece pillow opencv-python
```

再安装最新版：

```bash
conda install git -y
python -m pip install git+https://github.com/huggingface/diffusers
```

安装加速包：
pip install -U cache-dit

# 下载模型

官方模型：
Hugging Face 上：
```bash
Qwen/Qwen-Image-Edit
Qwen/Qwen-Image-Edit-2509
Qwen/Qwen-Image-Edit-2511
```

## 多模态图片编辑模型 Qwen-Image 系列

Qwen-Image-Edit (2025.08)
- Single-Image Editing
- https://huggingface.co/Qwen/Qwen-Image-Edit

Qwen-Image-Edit-2509 (2025.09)
- Multi-image Editing Support
- https://huggingface.co/Qwen/Qwen-Image-Edit-2509

Qwen-Image-Edit-2511 (2025.11)
- Multi-image Editing Support
- https://huggingface.co/Qwen/Qwen-Image-Edit-2511


```bash
# windows 本地下载再传到服务器
pip install -U huggingface_hub hf_transfer  
$env:HF_ENDPOINT="https://hf-mirror.com"   
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen-Image-Edit', local_dir='E:/Qwen-Image-Edit', resume_download=True)"
```


## 国内用户镜像

设置：

```bash
export HF_ENDPOINT=https://hf-mirror.com

# 永久生效
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc

# 添加在代码中
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```

然后自动下载。

<!-- 本地下载压缩包并安装 -->
pip download --platform manylinux2014_x86_64 --python-version 310 --only-binary=:all: torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 transformers==4.57.1 huggingface_hub==0.34.4 accelerate safetensors tokenizers -f https://download.pytorch.org/whl/cu121/torch_stable.html


conda activate qwenedit
python -m pip uninstall torch torchvision torchaudio -y
python -m pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -U transformers accelerate sentencepiece
python -m pip install -U diffusers


```bash
CUDA_VISIBLE_DEVICES=2,3 python inferv_.py
```


# 推理参数说明

| 参数                 |  作用     | 推荐    |
| -------------------  | -----     | ----- |
| num_inference_steps  | 步数      | 30~50 |
| true_cfg_scale       | 提示词强度 | 4~7   |
| generator            | 随机种子   | 固定复现  |
| negative_prompt      | 负面词    | 模糊/低质 |
| torch_dtype          | 精度      | bf16  |