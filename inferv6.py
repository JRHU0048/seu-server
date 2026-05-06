# -*- coding: utf-8 -*-
# Qwen-Image-Edit-2511（三图输入版：第一张为主图）

import os
from pathlib import Path
from PIL import Image
import torch
from diffusers import QwenImageEditPlusPipeline

from prompts import VIEW_PROMPTS, NEG_PROMPT
from tools import get_all_images, load_image, is_skipped_class

import cache_dit # 导入 cache-dit 加速包 

# =====================================================
# 路径配置
# =====================================================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

INPUT_ROOT = "/home/huyanhan/data/CUB_200_2011/images/train"
OUTPUT_ROOT = "/home/huyanhan/data/CUB_200_2011/qwen_output/train"
MODEL_ID = "../Qwen-Image-Edit-2511"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# =====================================================
# 参数（核心控制性能）
# =====================================================
DTYPE = torch.bfloat16
STEPS = 28
CFG = 3.5
SEED = 42
SIZE = 384

# =====================================================
# 模型加载（多卡）
# =====================================================
print("Loading Qwen-Image-Edit-2511...")

pipe = QwenImageEditPlusPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    device_map="balanced",
    max_memory={0: "46GiB", 1: "46GiB"}
)

# ===================== 开启 cache-dit 加速 =====================
cache_options = {
    "cache_type": cache_dit.DBCache,
    "warmup_steps": 8,            # 8精度最高
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

pipe.set_progress_bar_config(disable=None)

print("Loaded.")


# =====================================================
# 主逻辑（三图输入）
# =====================================================
def run():

    class_folders = sorted(os.listdir(INPUT_ROOT))

    for cls in class_folders:

        # =========================
        # ✔ 跳过指定类别
        # =========================
        # if is_skipped_class(cls):
        #     print("Skip class:", cls)
        #     continue

        input_dir = os.path.join(INPUT_ROOT, cls)
        output_dir = os.path.join(OUTPUT_ROOT, cls)
        os.makedirs(output_dir, exist_ok=True)

        images = get_all_images(input_dir)

        if len(images) < 3:
            print("skip (need >=3 images):", cls)
            continue

        print("\n==============================")
        print("Class:", cls, "Images:", len(images))

        total = len(images)

        # 滑动窗口三图
        for i in range(total):

            img1 = images[i]
            img2 = images[(i+1) % total]
            img3 = images[(i+2) % total]

            # =========================
            print(f"[当前输入三张图] i={i}")
            print(f"  主图: {img1}")
            print(f"  辅助1: {img2}")
            print(f"  辅助2: {img3}\n")

            main_name = Path(img1).stem

            # =========================
            # ✔ 三图输入（关键修改点）
            # =========================
            input_imgs = [
                load_image(img1, SIZE),  # 主图（最重要）
                load_image(img2, SIZE),
                load_image(img3, SIZE)
            ]

            print("Generate:", cls, main_name)

            for vp in VIEW_PROMPTS:

                with torch.inference_mode():

                    out = pipe(
                        image=input_imgs,   # ⭐ 这里变成 list
                        prompt=vp["prompt"],
                        negative_prompt=NEG_PROMPT,
                        true_cfg_scale=CFG,
                        num_inference_steps=STEPS,
                        guidance_scale=1.0,
                        generator=torch.manual_seed(SEED),
                        # num_images_per_prompt=2   # 一次生成2张
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