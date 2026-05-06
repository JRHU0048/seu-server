"""
Multi-Agent Pipeline：生成 → 评估 → 反馈 → 重试

Agent 1 (Generator):  Qwen-Image-Edit-2511 — 生成鸟类新视角图像
Agent 2 (Evaluator):  Qwen2.5-VL (或自定义 Qwen VLM) — 量化评分 + 问题反馈

流程:
  对每张图片:
    ┌─  Generation ──→  Evaluation ──→ Score ≥ THRESHOLD? ──→ Yes → 保存 & 下一张
    │    ↑                              │ No
    │    │   retries < MAX_RETRIES?     │
    │    └── Yes: 注入反馈 → 改进 prompt ┘
    │         No: 跳过 → 下一张
    └────────────────────────────────────

用法:
  CUDA_VISIBLE_DEVICES=2,3 python pipeline.py
"""

import os
import re
import time
from pathlib import Path
from PIL import Image
import torch
from diffusers import QwenImageEditPlusPipeline
import cache_dit

from tools import get_all_images, load_image

# ═════════════════════════════════════════════════════════
# 配置
# ═════════════════════════════════════════════════════════

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ── 路径 ──
INPUT_ROOT = "/home/huyanhan/data/CUB_200_2011/images/train"
OUTPUT_ROOT = "/home/huyanhan/data/CUB_200_2011/qwen_output/train"
GEN_MODEL_ID = "../Qwen-Image-Edit-2511"
EVAL_MODEL_ID = "../Qwen2.5-VL-7B-Instruct"   # ← 改成你的评估模型路径

# ── 生成参数 ──
DTYPE = torch.bfloat16
STEPS = 28
CFG = 3.5
SEED = 42
SIZE = 384

# ── Pipeline 参数 ──
THRESHOLD = 7.0          # 最低接受分数 (0~10)
MAX_RETRIES = 3          # 每张图最大重试次数（不含首次生成）
LOG_FILE = "pipeline_log.txt"

# ── 跳过已处理的类别 ──
SKIP_UPTO = 0            # 0 = 不跳过

# ═════════════════════════════════════════════════════════
# Prompt 定义
# ═════════════════════════════════════════════════════════

NEG_PROMPT = (
    "collage, multiple birds, split image, duplicated bird, "
    "blurry, low quality, deformed, malformed, bad anatomy, "
    "extra wings, extra legs, extra body parts, fused limbs"
)

# 基础生成 prompt（同 edit_multi.py，但留了 feedback 插槽）
BASE_GEN_PROMPT = (
    "You are given THREE reference images of a bird.\n"
    "The FIRST image is the MAIN reference and defines the bird's identity: "
    "species, exact feather colors, beak shape, texture, pattern, body proportions, "
    "and all visual characteristics. The other two images are AUXILIARY ONLY.\n"
    "STRICT RULE: The resulting bird MUST EXACTLY match the FIRST image's appearance. "
    "Do NOT use any identity attribute from the auxiliary images.\n"
    "Task:\n"
    "- Rotate the camera horizontally about 60 degrees around the bird.\n"
    "- Replace the background with a realistic natural outdoor environment.\n"
    "- Adjust pose, lighting, shadow, feather direction, and perspective "
    "so the result looks like a natural wildlife photograph.\n"
    "Important: Generate ONLY ONE bird. Do NOT deform or distort. "
    "Do NOT generate collage or multiple birds.\n"
    "{feedback_section}"
)

EVALUATION_PROMPT = """\
You are an expert evaluator of AI-generated wildlife images.

You will see TWO images:
1. REFERENCE image — the original bird photo (first of three references).
2. GENERATED image — the AI-generated new viewpoint of the same bird.

The generation task was: "Rotate the bird 60 degrees and replace the background."

Evaluate the GENERATED image on these four criteria (each score 0-10):

1. Identity consistency — Does the bird's species, feather colors, beak shape, and body structure match the reference? Are there any morphs or feature changes?
2. Viewpoint quality — Is the 60-degree horizontal rotation realistic? Does it look like a genuine 3D rotation, not a flat 2D transformation?
3. Image quality — Any artifacts, blurring, deformities, extra limbs, malformed anatomy, or pixelation?
4. Background naturalness — Is the new background a realistic natural environment (not artificial, not plain)? Does lighting match consistently?

Output your evaluation in this EXACT format (one item per line):

SCORE: <overall average score out of 10, one decimal>
ISSUES: <comma-separated specific problems, or "none">
SUGGESTIONS: <actionable improvement for regeneration, or "none">
"""

# ═════════════════════════════════════════════════════════
# Agent 1: Generator
# ═════════════════════════════════════════════════════════

def load_generator():
    """加载 Qwen-Image-Edit-2511 + cache-dit 加速"""
    print("[Agent 1] Loading Qwen-Image-Edit-2511 ...")
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        GEN_MODEL_ID,
        torch_dtype=DTYPE,
        device_map="balanced",
        max_memory={0: "46GiB", 1: "46GiB"},
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
    print("[Agent 1] Generator ready.")
    return pipe


def generate_image(generator, input_imgs, prompt, neg_prompt):
    """单次生成"""
    with torch.inference_mode():
        result = generator(
            image=input_imgs,
            prompt=prompt,
            negative_prompt=neg_prompt,
            true_cfg_scale=CFG,
            num_inference_steps=STEPS,
            guidance_scale=1.0,
            generator=torch.manual_seed(SEED + hash(prompt) % 10000),
        )
    torch.cuda.empty_cache()
    return result.images[0]


# ═════════════════════════════════════════════════════════
# Agent 2: Evaluator (Qwen VLM)
# ═════════════════════════════════════════════════════════

def load_evaluator():
    """加载 Qwen VLM 评估模型"""
    print("[Agent 2] Loading evaluator ...")
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        EVAL_MODEL_ID,
        torch_dtype=DTYPE,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(EVAL_MODEL_ID, trust_remote_code=True)
    print("[Agent 2] Evaluator ready.")
    return model, processor


def evaluate_image(model, processor, gen_image, ref_image):
    """
    用 VLM 评估生成图，返回 (score, issues, suggestions, raw_output).
    score: float (0~10)
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": ref_image},
                {"type": "image", "image": gen_image},
                {"type": "text", "text": EVALUATION_PROMPT},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[ref_image, gen_image],
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    raw = processor.decode(
        output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()

    score = _parse_score(raw)
    issues = _parse_field(raw, "ISSUES")
    suggestions = _parse_field(raw, "SUGGESTIONS")
    return score, issues, suggestions, raw


def _parse_score(text):
    """从评估输出中提取分数"""
    m = re.search(r"SCORE:\s*([\d.]+)", text, re.IGNORECASE)
    if m:
        val = float(m.group(1))
        return max(0.0, min(10.0, val))
    return 0.0


def _parse_field(text, field_name):
    """从评估输出中提取指定字段"""
    pattern = rf"{field_name}:\s*(.+?)(?:\n(?:SCORE|ISSUES|SUGGESTIONS):|$)"
    m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else ""


# ═════════════════════════════════════════════════════════
# Prompt 构建
# ═════════════════════════════════════════════════════════

def build_gen_prompt(feedback_list=None):
    """构建生成 prompt，可选的 feedback 追加到底部"""
    feedback = ""
    if feedback_list:
        items = "\n".join(f"- {fb}" for fb in feedback_list if fb)
        feedback = (
            "\n\nNote — Previous generation had the following issues. "
            "Please fix them this time:\n" + items
        )
    return BASE_GEN_PROMPT.format(feedback_section=feedback)


# ═════════════════════════════════════════════════════════
# Pipeline 主逻辑
# ═════════════════════════════════════════════════════════

def log(msg):
    msg = str(msg)
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def run_pipeline():
    log("=" * 60)
    log("Pipeline started")
    log(f"  Threshold: {THRESHOLD}, Max retries: {MAX_RETRIES}")
    log(f"  Generator: {GEN_MODEL_ID}")
    log(f"  Evaluator: {EVAL_MODEL_ID}")
    log("=" * 60)

    generator = None
    evaluator_model = None
    evaluator_proc = None

    stats = {"total": 0, "accepted": 0, "skipped": 0, "retry_attempts": 0}

    try:
        # ── 加载 Agent 1 ──
        generator = load_generator()

        # ── 加载 Agent 2 ──
        # 如果暂时没有评估模型，把下一行取消注释来启用评估
        # evaluator_model, evaluator_proc = load_evaluator()

        class_folders = sorted(os.listdir(INPUT_ROOT))

        for cls in class_folders:
            # 跳过已处理的类别
            if SKIP_UPTO > 0:
                from tools import is_skipped_class
                if is_skipped_class(cls, max_class=SKIP_UPTO):
                    log(f"Skip class: {cls}")
                    continue

            input_dir = os.path.join(INPUT_ROOT, cls)
            output_dir = os.path.join(OUTPUT_ROOT, cls)
            os.makedirs(output_dir, exist_ok=True)

            images = get_all_images(input_dir)
            if len(images) < 3:
                log(f"Skip {cls} (need >=3 images, got {len(images)})")
                continue

            log(f"\n{'─' * 50}")
            log(f"Class: {cls}  ({len(images)} images)")

            total = len(images)
            for i in range(total):
                img1, img2, img3 = images[i], images[(i + 1) % total], images[(i + 2) % total]
                main_name = Path(img1).stem
                stats["total"] += 1

                output_path = os.path.join(output_dir, f"{main_name}_view_bg_edit_v1.png")
                if os.path.exists(output_path):
                    log(f"  [{stats['total']}] {main_name} — already exists, skip")
                    stats["accepted"] += 1
                    continue

                log(f"\n  [{stats['total']}] {cls}/{main_name}")

                # 准备输入
                input_imgs = [load_image(img1, SIZE), load_image(img2, SIZE), load_image(img3, SIZE)]
                ref_pil = load_image(img1, SIZE)

                # ── 重试循环 ──
                feedback_history = []
                accepted = False

                for attempt in range(1 + MAX_RETRIES):
                    prompt = build_gen_prompt(feedback_history if attempt > 0 else None)

                    log(f"    ── Attempt {attempt + 1}/{1 + MAX_RETRIES} ──")

                    # Step 1: 生成
                    gen_img = generate_image(generator, input_imgs, prompt, NEG_PROMPT)

                    # Step 2: 评估
                    if evaluator_model is not None:
                        score, issues, suggestions, raw = evaluate_image(
                            evaluator_model, evaluator_proc, gen_img, ref_pil
                        )
                        log(f"    Score: {score:.1f}/10")
                        if issues and issues.lower() != "none":
                            log(f"    Issues: {issues}")
                    else:
                        # 无评估模型时的默认值
                        score = THRESHOLD + 1
                        issues = "none"

                    # Step 3: 判断
                    if score >= THRESHOLD:
                        gen_img.save(output_path)
                        log(f"    ✅ Accepted — saved to {output_path}")
                        stats["accepted"] += 1
                        if attempt > 0:
                            stats["retry_attempts"] += 1
                        accepted = True
                        break
                    else:
                        if attempt < MAX_RETRIES:
                            if issues and issues.lower() != "none":
                                feedback_history.append(issues)
                            if suggestions and suggestions.lower() != "none":
                                feedback_history.append(suggestions)
                            log(f"    ⬇ Below threshold, will retry with feedback")
                            # 清理显存，为重试准备
                            torch.cuda.empty_cache()
                        else:
                            log(f"    ❌ All attempts failed, skipping")
                            stats["skipped"] += 1

                if not accepted:
                    stats["skipped"] += 1

                # 每处理 20 张图输出一次中间统计
                if stats["total"] % 20 == 0:
                    log(f"\n  [Progress] {stats['total']} processed — "
                        f"{stats['accepted']} accepted, {stats['skipped']} skipped")

    finally:
        # 清理
        if generator:
            del generator
        if evaluator_model:
            del evaluator_model
        torch.cuda.empty_cache()
        cache_dit.clear_cache()

    # ── 最终统计 ──
    log("\n" + "=" * 60)
    log("Pipeline completed!")
    log(f"  Total images:   {stats['total']}")
    log(f"  Accepted:       {stats['accepted']}")
    log(f"  Skipped:        {stats['skipped']}")
    log(f"  Retry attempts: {stats['retry_attempts']}")
    log("=" * 60)


if __name__ == "__main__":
    run_pipeline()
