# -*- coding: utf-8 -*-
"""
Qwen-Image-Edit 本地部署统一版（单图 / 三图融合）

功能：
1. 使用 Qwen/Qwen-Image-Edit
2. 批量读取鸟类图片
3. 支持两种输入模式：
   A. 单图生成
   B. 三图融合生成（滑动窗口）
4. 每张图生成3种新视角
5. 自动保存输出
6. 两种模式共用同一套推理参数

安装：
pip install -U diffusers transformers accelerate sentencepiece safetensors pillow
#至少需要两张卡才能跑
CUDA_VISIBLE_DEVICES=2,3 python inferv2.py
"""

import os
from pathlib import Path
from PIL import Image
import torch
from diffusers import QwenImageEditPipeline

# =====================================================
# 环境设置
# =====================================================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# =====================================================
# 路径配置
# =====================================================
INPUT_FOLDER = "/home/huyanhan/data/CUB_200_2011/images/train/003.Sooty_Albatross"
OUTPUT_FOLDER = "/home/huyanhan/data/CUB_200_2011/qwen_output/train/003.Sooty_Albatross"
MODEL_ID = "./Qwen-Image-Edit"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# =====================================================
# 模式选择
# =====================================================
# "single" = 单图模式
# "triple" = 三图融合模式
RUN_MODE = "triple"

# =====================================================
# 推理参数
# =====================================================
DEVICE = "cuda"
DTYPE = torch.bfloat16

STEPS = 28 # 40
CFG = 3.5 # 4.0
SEED = 42
MAX_IMAGE_SIZE = 896 # 1024
MERGE_IMAGE_SIZE = 448 # 512

NEG_PROMPT = """
blurry, low quality, cartoon, deformed, ugly,
duplicate wings, extra legs, bad anatomy,
artifact, oversaturated
"""

# =====================================================
# 三种视角提示词
# =====================================================
VIEW_PROMPTS = [
    {
        "name": "水平60度侧视",
        "prompt": """
Keep the same bird species, feather colors, beak shape,
body structure and identity.

Rotate the bird horizontally by 60 degrees.
Generate a realistic natural wildlife side-view photo.
"""
    },

    {
        "name": "水平30度侧视",
        "prompt": """
Keep the same bird identity, texture and appearance.

Rotate the bird horizontally by 30 degrees.
Generate a realistic side-view bird photograph.
"""
    },

    {
        "name": "斜45度俯视",
        "prompt": """
Keep the same species and appearance.

Generate a realistic 45-degree top-down view of the bird,
showing back feathers and body structure.
Wildlife photography style.
"""
    }
]

# =====================================================
# 加载模型
# =====================================================
print("Loading Qwen-Image-Edit...")

# pipe = QwenImageEditPipeline.from_pretrained(
#     MODEL_ID,
#     torch_dtype=DTYPE,
#     device_map="cuda" 
# )

pipe = QwenImageEditPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="balanced",
    max_memory={
        0: "46GiB",
        1: "46GiB"
    }
)

# pipe.to(DEVICE)

# 显存优化
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
pipe.set_progress_bar_config(disable=None)

print("Model loaded successfully.")

# =====================================================
# 工具函数
# =====================================================
def get_all_images(folder):
    files = []
    for f in os.listdir(folder):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            files.append(os.path.join(folder, f))
    files.sort()
    return files


def preprocess_image(img):
    """统一输入图尺寸"""
    img = img.convert("RGB")
    img.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))
    return img


def merge_3_images_into_one(img1_path, img2_path, img3_path, size=512):
    """3张图横向拼接"""
    im1 = Image.open(img1_path).convert("RGB").resize((size, size))
    im2 = Image.open(img2_path).convert("RGB").resize((size, size))
    im3 = Image.open(img3_path).convert("RGB").resize((size, size))

    merged = Image.new("RGB", (size * 3, size))
    merged.paste(im1, (0, 0))
    merged.paste(im2, (size, 0))
    merged.paste(im3, (size * 2, 0))

    return preprocess_image(merged)


def run_generation(input_image, save_prefix):
    """
    统一生成函数
    单图模式 / 三图模式 都调用这里
    """
    for vp in VIEW_PROMPTS:
        print("生成:", vp["name"])

        generator = torch.manual_seed(SEED)

        with torch.inference_mode():
            result = pipe(
                image=input_image,
                prompt=vp["prompt"],
                negative_prompt=NEG_PROMPT,
                true_cfg_scale=CFG,
                num_inference_steps=STEPS,
                generator=generator
            )

        out = result.images[0]

        save_path = os.path.join(
            OUTPUT_FOLDER,
            f"{save_prefix}_{vp['name']}.png"
        )

        out.save(save_path)
        print("Saved:", save_path)

        torch.cuda.empty_cache()


# =====================================================
# 单图模式
# =====================================================
def batch_process_single():
    images = get_all_images(INPUT_FOLDER)
    print("发现图片数量:", len(images))

    for img_path in images:
        main_name = Path(img_path).stem
        print("\n=================================")
        print("主图:", main_name)

        image = Image.open(img_path)
        image = preprocess_image(image)

        run_generation(image, main_name)


# =====================================================
# 三图融合模式（滑动窗口）
# =====================================================
def batch_process_triple():
    images = get_all_images(INPUT_FOLDER)
    total = len(images)

    print("发现图片数量:", total)

    for i in range(total):
        main_path = images[i]
        main_name = Path(main_path).stem

        i1 = i % total
        i2 = (i + 1) % total
        i3 = (i + 2) % total

        ref1 = images[i1]
        ref2 = images[i2]
        ref3 = images[i3]

        print("\n=================================")
        print(f"主图: {main_name}")
        print(f"参考图: {Path(ref1).stem}, {Path(ref2).stem}, {Path(ref3).stem}")

        merged_img = merge_3_images_into_one(
            ref1, ref2, ref3,
            size=MERGE_IMAGE_SIZE
        )

        run_generation(merged_img, main_name)


# =====================================================
# 主程序入口
# =====================================================
if __name__ == "__main__":

    if RUN_MODE == "single":
        batch_process_single()

    elif RUN_MODE == "triple":
        batch_process_triple()

    else:
        print("RUN_MODE 只能是 single 或 triple")