# -*- coding: utf-8 -*-
# 先测试一个视角的编辑，同时改回单图输入试试效果
# model:qwen-image-edit 
import os
from pathlib import Path
from PIL import Image
import torch
from diffusers import QwenImageEditPipeline

import cache_dit # 导入 cache-dit 加速包 

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

# v1
# NEG_PROMPT = "collage, multiple birds, split image, duplicated bird, blurry, low quality, deformed, extra wings"

# v2
NEG_PROMPT = (
    "collage, multiple birds, split image, duplicated bird, "
    "blurry, low quality, deformed, malformed, bad anatomy, "
    "extra wings, extra legs, extra body parts, fused limbs"
)

# =====================================================
# Prompt（角度变换任务）
# =====================================================
VIEW_PROMPTS = [
{
"name":"view_60",
"prompt":"""
You are given a single reference image of a bird.

Generate ONLY ONE bird.

Keep the bird's identity, feather details, beak shape, and texture completely consistent with the reference.

Change the camera viewing angle to rotate horizontally 60 degrees to the left around the bird, rather than flat image rotation.

Keep the bird's natural standing posture, realistic body proportions and anatomical structure unchanged.

Do NOT generate multiple birds.
Do NOT generate collage.
Do NOT deform or distort the bird's body.
"""
}
]

# 测试过的提示词：
# Rotate the bird horizontally to the left by 60 degrees.

# =====================================================
# Prompt（环境替换任务）
# =====================================================
# VIEW_PROMPTS = [
# {
# "name":"bg_replace",
# "prompt":"""
# You are given a single reference image of a bird.

# Keep ONLY the bird identity unchanged:
# same species, feather colors, beak shape, body structure.

# Replace the original background with a natural forest environment.

# Generate ONLY ONE bird.

# The bird should remain clear, realistic, centered, and complete.

# Do NOT change the bird appearance.

# Do NOT generate multiple birds.
# Do NOT generate collage.
# Do NOT crop the bird.
# """
# }
# ]


# =====================================================
# Prompt（环境替换 + 主体自然融合）更自然的版本，选择此
# =====================================================
# VIEW_PROMPTS = [
# {
# "name":"bg_replace_natural",
# "prompt":"""
# You are given a single reference image of a bird.

# Keep the bird identity consistent:
# same species, feather pattern, beak shape, and overall appearance.

# Replace the original background with a realistic natural outdoor environment
# such as forest, grassland, lake shore, wetland, or tree branches.

# Adjust the bird pose, lighting, shadow, color tone, feather direction,
# body contact, and perspective when necessary so the bird naturally fits
# the new environment.

# The bird should interact realistically with the scene,
# as if photographed there originally.

# Generate ONLY ONE bird.

# Do NOT make the bird look pasted, floating, cut out, or artificially inserted.

# Do NOT generate multiple birds.
# Do NOT generate collage.
# Do NOT distort the bird excessively.
# """
# }
# ]


# =====================================================
# Prompt（环境替换 + 视角变换 联合任务版）
# =====================================================
# VIEW_PROMPTS = [
# {
# "name":"view_bg_edit",
# "prompt":"""
# You are given a single reference image of a bird.

# Keep the same bird identity:
# same species, feather colors, beak shape, texture, and body characteristics.

# Rotate the bird horizontally by about 60 degrees to create a new side view.

# At the same time, replace the original background with a realistic natural outdoor environment
# such as forest, grassland, wetland, lakeside, rocks, or tree branches.

# Adjust the bird pose, body balance, lighting, shadow, feather direction,
# perspective, and color tone so the bird naturally matches the new environment.

# The final image should look like a real wildlife photograph taken from the new angle.

# Generate ONLY ONE bird.

# Do NOT make the bird look pasted, floating, duplicated, cropped, or artificial.

# Do NOT generate collage.
# Do NOT generate multiple birds.
# """
# }
# ]



# =====================================================
# 模型加载
# =====================================================
print("Loading Qwen-Image-Edit...")

pipe = QwenImageEditPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    device_map="balanced",
    max_memory={0:"46GiB",1:"46GiB"}
)

# ===================== 开启 cache-dit 加速 =====================
cache_options = {
    "cache_type": cache_dit.DBCache,
    "warmup_steps": 8,            # 推荐 8，精度最高
    "max_cached_steps": -1,
    "Fn_compute_blocks": 8,      # F8 最优配置
    "Bn_compute_blocks": 0,
    "residual_diff_threshold": 0.12,
    "do_separate_classifier_free_guidance": True,  # Cache CFG
    "cfg_compute_first": False,
    "enable_taylorseer": True,   # 开启泰勒加速
    "enable_encoder_taylorseer": True,
    "taylorseer_cache_type": "residual",
    "taylorseer_kwargs": {"n_derivatives": 4},
}
cache_dit.enable_cache(pipe, **cache_options)
# ====================================================================

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


def load_single_image(img_path, size=384):
    """
    单图输入
    """
    img = Image.open(img_path).convert("RGB")
    img = img.resize((size, size))
    return img


def is_skipped_class(cls_name):
    # 提取文件夹名开头的数字，例如 "005." → 5
    if "." in cls_name:
        num_part = cls_name.split(".")[0]
        try:
            class_num = int(num_part)
            return 1 <= class_num <= 27  # 跳过
        except:
            pass
    return False

# =====================================================
# 主逻辑
# =====================================================
def run():

    class_folders = sorted(os.listdir(INPUT_ROOT))

    for cls in class_folders:

        # =========================
        # ✔ 跳过指定类别
        # =========================
        if is_skipped_class(cls):
            print("Skip class:", cls)
            continue

        input_dir = os.path.join(INPUT_ROOT, cls)
        output_dir = os.path.join(OUTPUT_ROOT, cls)
        os.makedirs(output_dir, exist_ok=True)

        images = get_all_images(input_dir)

        if len(images) == 0:
            print("skip:", cls)
            continue

        print("\n==============================")
        print("Class:", cls, "Images:", len(images))

        # 改为单图遍历
        for i in range(len(images)):

            img_path = images[i]
            main_name = Path(img_path).stem

            # =========================
            # ✔ 单图输入（关键修改点）
            # =========================
            input_img = load_single_image(img_path, SIZE)

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

    # 打印加速统计信息 
    cache_dit.summary(pipe)

if __name__ == "__main__":
    run()