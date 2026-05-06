"""
单图输入 —— 使用 Qwen-Image-Edit 生成新视角

功能：
  - 遍历 CUB-200-2011 训练集所有类别
  - 每张图经过 cache-dit 加速推理，生成 60° 旋转视角
  - 自动跳过已处理的类别（可配置）

用法：
  CUDA_VISIBLE_DEVICES=2,3 python edit_single.py
"""

import os
from pathlib import Path
import torch
from diffusers import QwenImageEditPipeline
import cache_dit

from tools import get_all_images, load_image, is_skipped_class
from prompts import NEG_PROMPT, VIEW_PROMPTS_SINGLE

# =====================================================
# 路径配置
# =====================================================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

INPUT_ROOT = "/home/huyanhan/data/CUB_200_2011/images/train"
OUTPUT_ROOT = "/home/huyanhan/data/CUB_200_2011/qwen_output/train"
MODEL_ID = "./Qwen-Image-Edit"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# =====================================================
# 推理参数
# =====================================================
DTYPE = torch.bfloat16
STEPS = 28
CFG = 3.5
SEED = 42
SIZE = 384

# 跳过类别范围（1~N 的类别不处理）
SKIP_UPTO = 27

# =====================================================
# 模型加载 + cache-dit 加速
# =====================================================
print("Loading Qwen-Image-Edit ...")

pipe = QwenImageEditPipeline.from_pretrained(
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

print("Loaded.")

# =====================================================
# 主逻辑
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
        if len(images) == 0:
            print("skip (empty):", cls)
            continue

        print(f"\n{'=' * 30}")
        print(f"Class: {cls}  Images: {len(images)}")

        for img_path in images:
            main_name = Path(img_path).stem
            input_img = load_image(img_path, SIZE)

            print(f"  Generate: {cls}/{main_name}")

            for vp in VIEW_PROMPTS_SINGLE:
                with torch.inference_mode():
                    out = pipe(
                        image=input_img,
                        prompt=vp["prompt"],
                        negative_prompt=NEG_PROMPT,
                        true_cfg_scale=CFG,
                        num_inference_steps=STEPS,
                        generator=torch.manual_seed(SEED),
                    ).images[0]

                save_path = os.path.join(output_dir, f"{main_name}_{vp['name']}.png")
                out.save(save_path)
                print(f"    Saved: {save_path}")

                torch.cuda.empty_cache()

    cache_dit.summary(pipe)


if __name__ == "__main__":
    run()
