# 相对于v3修改为4张图作为参考，4图拼成一张大图输入
# 先只测试一个视角！！！ view_60
# -*- coding: utf-8 -*-
import os
from pathlib import Path
from PIL import Image
import torch
from diffusers import QwenImageEditPipeline

# =====================================================
# 路径配置
# =====================================================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

INPUT_ROOT = "/home/huyanhan/data/CUB_200_2011/images/train"
OUTPUT_ROOT = "/home/huyanhan/data/CUB_200_2011/qwen_output/train"
MODEL_ID = "./Qwen-Image-Edit"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# =====================================================
# 参数（核心控制性能）
# =====================================================
DTYPE = torch.bfloat16
STEPS = 28
CFG = 3.5
SEED = 42
SIZE = 384

NEG_PROMPT = "collage, multiple birds, split image, duplicated bird, blurry, low quality, deformed"

# =====================================================
# Prompt
# =====================================================
VIEW_PROMPTS = [
{
"name":"view_60",
"prompt":"""
You are given 4 reference images of the SAME bird.

The FIRST image is the primary identity reference.

The other images are auxiliary multi-view references.

Generate ONLY ONE bird.

Keep identity, feather, beak, texture consistent.

Rotate horizontally by 60 degrees.

Do NOT generate multiple birds.
Do NOT generate collage.
"""
},
# {
# "name":"view_30",
# "prompt":"""
# You are given 4 reference images of the SAME bird.

# FIRST image defines identity and background style.

# Generate ONLY ONE bird.

# Rotate horizontally by 30 degrees.

# Realistic wildlife photo.

# No collage, no duplication.
# """
# },
# {
# "name":"view_top",
# "prompt":"""
# You are given 4 reference images of the SAME bird.

# FIRST image is primary reference.

# Generate ONLY ONE bird.

# Top-down 45 degree view.

# Show back feathers.

# No multiple subjects.
# """
# }
]

# =====================================================
# 模型加载
# =====================================================
print("Loading model...")

pipe = QwenImageEditPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    device_map="balanced",
    max_memory={0:"46GiB",1:"46GiB"}
)

# 降显存
# pipe.enable_attention_slicing()
# pipe.enable_vae_slicing()

print("Loaded.")

# =====================================================
# 工具函数
# =====================================================
def get_all_images(folder):
    files = []
    for f in os.listdir(folder):
        if f.lower().endswith((".jpg",".jpeg",".png")):
            files.append(os.path.join(folder,f))
    files.sort()
    return files


def build_4ref_input(imgs, size=384):
    """
    2x2 layout:
    [img1 img2]
    [img3 img4]
    """
    imgs = [Image.open(p).convert("RGB").resize((size,size)) for p in imgs]

    canvas = Image.new("RGB",(size*2,size*2),(255,255,255))

    canvas.paste(imgs[0], (0,0))      # 主背景（关键）
    canvas.paste(imgs[1], (size,0))
    canvas.paste(imgs[2], (0,size))
    canvas.paste(imgs[3], (size,size))

    return canvas


# =====================================================
# 主逻辑
# =====================================================
def run():

    class_folders = sorted(os.listdir(INPUT_ROOT))

    for cls in class_folders:

        input_dir = os.path.join(INPUT_ROOT, cls)
        output_dir = os.path.join(OUTPUT_ROOT, cls)
        os.makedirs(output_dir, exist_ok=True)

        images = get_all_images(input_dir)

        if len(images) < 4:
            print("skip:", cls)
            continue

        print("\n==============================")
        print("Class:", cls, "Images:", len(images))

        for i in range(len(images)-3):

            refs = images[i:i+4]

            main_name = Path(refs[0]).stem

            input_img = build_4ref_input(refs, SIZE)

            print("Generate:", cls, main_name)

            for vp in VIEW_PROMPTS:

                with torch.inference_mode():

                    out = pipe(
                        image=input_img,
                        prompt=vp["prompt"],
                        negative_prompt=NEG_PROMPT,
                        true_cfg_scale=CFG,
                        num_inference_steps=STEPS,
                        generator=torch.manual_seed(SEED)
                    ).images[0]

                save_path = os.path.join(
                    output_dir,
                    f"{main_name}_{vp['name']}.png"
                )

                out.save(save_path)
                print("Saved:", save_path)

                torch.cuda.empty_cache()


if __name__ == "__main__":
    run()