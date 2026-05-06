# -*- coding: utf-8 -*-
"""
Qwen-Image-Edit 本地部署版

功能：
1. 使用 Qwen/Qwen-Image-Edit
2. 批量读取鸟类图片
3. 每张图生成3种新视角
4. 自动保存输出

要求：
pip install -U diffusers transformers accelerate sentencepiece safetensors pillow
CUDA_VISIBLE_DEVICES=3 python inferv1.py
"""

import os
from pathlib import Path
from PIL import Image
import torch
from diffusers import QwenImageEditPipeline

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3" 

# 路径配置
INPUT_FOLDER = "/home/huyanhan/data/CUB_200_2011/images/train/003.Sooty_Albatross"
OUTPUT_FOLDER = "/home/huyanhan/data/CUB_200_2011/qwen_output/train/003.Sooty_Albatross"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 参数配置
MODEL_ID = "./Qwen-Image-Edit"
DEVICE = "cuda"

# 支持bf16就用bf16，否则改float16
DTYPE = torch.bfloat16

STEPS = 40
CFG = 4.0
SEED = 42

NEG_PROMPT = "blurry, low quality, cartoon, deformed, ugly, duplicate wings"

# 三种视角提示词
VIEW_PROMPTS = [
    {
        "name": "水平60度侧视",
        "prompt": """
Based on the input bird image, keep the same species,
same feather colors, same beak shape, same body structure.

Rotate the bird horizontally by 60 degrees.
Generate a realistic natural side-view wildlife photo.
"""
    },

    {
        "name": "水平30度侧视",
        "prompt": """
Based on the input bird image, keep the same bird identity,
same appearance and texture.

Rotate the bird horizontally by 30 degrees.
Generate a realistic side-view bird photograph.
"""
    },

    {
        "name": "斜45度俯视",
        "prompt": """
Based on the input bird image, keep the same species and appearance.

Generate a realistic 45 degree top-down view of the bird,
showing back feathers and body structure.
Wildlife photography style.
"""
    }
]

# ==========================================================================
# 加载模型
print("Loading Qwen-Image-Edit...")

# pipe = QwenImageEditPipeline.from_pretrained(
#     MODEL_ID,
#     torch_dtype=DTYPE
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
# pipe.enable_model_cpu_offload()

pipe.set_progress_bar_config(disable=None)

print("Model loaded.")


# 工具函数
def get_all_images(folder):
    files = []
    for f in os.listdir(folder):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            files.append(os.path.join(folder, f))
    files.sort()
    return files

def merge_3_images_into_one(img1_path, img2_path, img3_path, size=512):
    im1 = Image.open(img1_path).convert("RGB").resize((size, size))
    im2 = Image.open(img2_path).convert("RGB").resize((size, size))
    im3 = Image.open(img3_path).convert("RGB").resize((size, size))

    merged = Image.new("RGB", (size * 3, size))
    merged.paste(im1, (0, 0))
    merged.paste(im2, (size, 0))
    merged.paste(im3, (size*2, 0))
    return merged


# 主生成逻辑
def batch_process_single_img():
    images = get_all_images(INPUT_FOLDER)
    print("发现图片数量:", len(images))

    for img_path in images:
        image = Image.open(img_path).convert("RGB")

        # 最长边控制到1024以内
        image.thumbnail((1024, 1024))

        main_name = Path(img_path).stem

        print("主图:", main_name)

        for vp in VIEW_PROMPTS:
            print("生成:", vp["name"])
            generator = torch.manual_seed(SEED)

            with torch.inference_mode():
                result = pipe(
                    image=image,
                    prompt=vp["prompt"],
                    generator=generator,
                    true_cfg_scale=CFG,
                    negative_prompt=NEG_PROMPT,
                    num_inference_steps=STEPS
                )

            out = result.images[0]
            save_path = os.path.join(
                OUTPUT_FOLDER,
                f"{main_name}_{vp['name']}.png"
            )

            out.save(save_path)
            print("Saved:", save_path)


def batch_process_3_imgs():
    images = get_all_images(INPUT_FOLDER)
    print("发现图片数量:", len(images))
    total = len(images)

    for i in range(total):
        # 当前要生成的目标图
        main_path = images[i]
        main_name = Path(main_path).stem

        # 滑动窗口取 3 张参考图
        i1 = i % total
        i2 = (i+1) % total
        i3 = (i+2) % total

        ref1 = images[i1]
        ref2 = images[i2]
        ref3 = images[i3]

        print(f"\n生成: {main_name} | 参考: {i1}-{i2}-{i3}")

        # 融合 3 张 → 1 张
        merged_img = merge_3_images_into_one(ref1, ref2, ref3)

        # 3个视角生成
        for vp in VIEW_PROMPTS:
            out = pipe(
                image=merged_img,    # 传入3张融合后的图
                prompt=vp["prompt"],
                true_cfg_scale=3.5,
                num_inference_steps=25
            ).images[0]

            save_path = os.path.join(OUTPUT_FOLDER, f"{main_name}_{vp['name']}.png")
            out.save(save_path)
            print("Saved:", save_path)

# 启动
if __name__ == "__main__":
    batch_process_3_imgs()