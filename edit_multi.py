"""
多图输入 —— 使用 Qwen-Image-Edit-2511 生成新视角 + 背景替换

功能：
  - 遍历 CUB-200-2011 训练集所有类别
  - 滑动窗口取 3 张图作为输入（第一张为主图）
  - 同时完成视角变换（60°）与背景替换
  - cache-dit 加速推理

用法：
  CUDA_VISIBLE_DEVICES=2,3 python edit_multi.py
"""

import os
from pathlib import Path
import torch
from diffusers import QwenImageEditPlusPipeline
import cache_dit

from tools import get_all_images, load_image, is_skipped_class
from prompts import NEG_PROMPT, VIEW_PROMPTS_MULTI

# =====================================================
# 路径配置
# =====================================================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

INPUT_ROOT = "/home/huyanhan/data/CUB_200_2011/images/train"
OUTPUT_ROOT = "/home/huyanhan/data/CUB_200_2011/qwen_output/train"
MODEL_ID = "../Qwen-Image-Edit-2511"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# =====================================================
# 推理参数
# =====================================================
DTYPE = torch.bfloat16
STEPS = 28
CFG = 3.5
SEED = 42
SIZE = 384

SKIP_UPTO = 6  # 设为 0 不跳过任何类别

# =====================================================
# 模型加载 + cache-dit 加速
# =====================================================
print("Loading Qwen-Image-Edit-2511 ...")

pipe = QwenImageEditPlusPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    device_map="balanced",
    max_memory={0: "46GiB", 1: "46GiB"}
)

cache_options = {
    "cache_type": cache_dit.DBCache,
    "warmup_steps": 8,
    "max_cached_steps": -1,
    "Fn_compute_blocks": 8,
    "Bn_compute_blocks": 0,
    "residual_diff_threshold": 0.12,
    "do_separate_classifier_free_guidance": True,
    "cfg_compute_first": False,
    "enable_taylorseer": True,
    "enable_encoder_taylorseer": True,
    "taylorseer_cache_type": "residual",
    "taylorseer_kwargs": {"n_derivatives": 4},
}
cache_dit.enable_cache(pipe, **cache_options)

pipe.set_progress_bar_config(disable=None)
print("Loaded.")

# =====================================================
# 主逻辑（滑动窗口三图输入）
# =====================================================
def run():
    class_folders = sorted(os.listdir(INPUT_ROOT))

    for cls in class_folders:
        if is_skipped_class(cls, max_class=SKIP_UPTO):
            print("Skip class:", cls)
            continue

        input_dir = os.path.join(INPUT_ROOT, cls)
        output_dir = os.path.join(OUTPUT_ROOT, cls)
        os.makedirs(output_dir, exist_ok=True)

        images = get_all_images(input_dir)
        if len(images) < 3:
            print("skip (need >=3 images):", cls)
            continue

        print(f"\n{'=' * 30}")
        print(f"Class: {cls}  Images: {len(images)}")

        total = len(images)
        for i in range(total):
            img1, img2, img3 = images[i], images[(i + 1) % total], images[(i + 2) % total]

            main_name = Path(img1).stem
            print(f"  Generate: {cls}/{main_name}")

            input_imgs = [load_image(img1, SIZE), load_image(img2, SIZE), load_image(img3, SIZE)]

            for vp in VIEW_PROMPTS_MULTI:
                with torch.inference_mode():
                    out = pipe(
                        image=input_imgs,
                        prompt=vp["prompt"],
                        negative_prompt=NEG_PROMPT,
                        true_cfg_scale=CFG,
                        num_inference_steps=STEPS,
                        guidance_scale=1.0,
                        generator=torch.manual_seed(SEED),
                    ).images[0]

                save_path = os.path.join(output_dir, f"{main_name}_{vp['name']}.png")
                out.save(save_path)
                print(f"    Saved: {save_path}")

                torch.cuda.empty_cache()

    cache_dit.summary(pipe)


if __name__ == "__main__":
    run()
