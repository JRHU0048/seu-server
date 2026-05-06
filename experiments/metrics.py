"""
实验指标计算模块

提供统一的接口计算 FID / CLIP Score / Identity Score 等指标。
所有指标通过 MetricsCalculator 类访问，支持缓存和采样。

用法:
  calc = MetricsCalculator(device="cuda")
  results = calc.evaluate_all(
      real_dir="/path/to/real",
      gen_dir="/path/to/generated",
      ref_dir="/path/to/references",
      task_prompt="A bird in a natural environment",
      sample_size=200,
  )
  print(results)
"""

import os
import json
import warnings
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image

from tools import get_all_images

# ═════════════════════════════════════════════════════════
# FID — 使用 torchmetrics 的 Inception 特征
# ═════════════════════════════════════════════════════════

FID_TRANSFORM = None  # lazy init


def _build_fid_transform():
    """InceptionV3 的标准预处理"""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def _compute_fid_stats(image_dir: str, cache_path: Optional[str] = None):
    """
    计算某目录下所有图片的 Inception 特征统计量 (mean, cov)。
    支持缓存到 .pt 文件。
    """
    from torchmetrics.image.fid import FrechetInceptionDistance

    # 缓存命中
    if cache_path and os.path.exists(cache_path):
        return torch.load(cache_path)

    global FID_TRANSFORM
    if FID_TRANSFORM is None:
        FID_TRANSFORM = _build_fid_transform()

    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    fid = FrechetInceptionDistance(feature=2048).to(device)

    images = get_all_images(image_dir)
    if len(images) == 0:
        raise ValueError(f"No images found in {image_dir}")

    print(f"  Computing FID stats from {len(images)} images in {image_dir}...")

    for i, img_path in enumerate(images):
        img = Image.open(img_path).convert("RGB")
        tensor = FID_TRANSFORM(img).unsqueeze(0).to(device)
        fid.update(tensor, real=True)

        if (i + 1) % 500 == 0:
            print(f"    Processed {i+1}/{len(images)}")

    # 提取统计量
    mean = fid.real_features_mean.to("cpu")
    cov = fid.real_features_cov.to("cpu")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save((mean, cov), cache_path)
        print(f"  Cached stats to {cache_path}")

    return mean, cov


def compute_fid(real_dir: str, gen_dir: str, cache_dir: str = "output/metrics") -> float:
    """
    计算 FID 分数（越小越好）
    """
    from torchmetrics.image.fid import FrechetInceptionDistance

    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

    # 预计算或加载统计量
    real_cache = os.path.join(cache_dir, "real_stats.pt") if cache_dir else None
    gen_cache = os.path.join(cache_dir, f"gen_{Path(gen_dir).name}_stats.pt") if cache_dir else None

    real_mean, real_cov = _compute_fid_stats(real_dir, real_cache)
    gen_mean, gen_cov = _compute_fid_stats(gen_dir, gen_cache)

    # 计算 FID
    fid = FrechetInceptionDistance(feature=2048).to(device)
    fid.real_features_mean = real_mean.to(device)
    fid.real_features_cov = real_cov.to(device)
    fid.gen_features_mean = gen_mean.to(device)
    fid.gen_features_cov = gen_cov.to(device)

    return fid.compute().item()


# ═════════════════════════════════════════════════════════
# CLIP Score — 图文对齐度
# ═════════════════════════════════════════════════════════

def compute_clip_score(
    gen_dir: str,
    prompt: str,
    sample_size: Optional[int] = 500,
    device: str = "cuda",
) -> float:
    """
    计算生成图片与任务 prompt 的 CLIP Score（越高越好）

    使用 transformers 中的 CLIP 模型（已安装），无需额外依赖。
    """
    from transformers import CLIPProcessor, CLIPModel

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    images = get_all_images(gen_dir)
    if sample_size and len(images) > sample_size:
        import random
        random.seed(42)
        images = random.sample(images, sample_size)

    print(f"  Computing CLIP Score on {len(images)} images...")

    scores = []
    batch_size = 32

    for i in range(0, len(images), batch_size):
        batch_paths = images[i:i + batch_size]
        batch_pils = [Image.open(p).convert("RGB") for p in batch_paths]

        inputs = processor(
            text=[prompt] * len(batch_pils),
            images=batch_pils,
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image  # image->text similarity
            scores.extend(logits_per_image.squeeze(-1).cpu().tolist())

    model.cpu()
    del model
    torch.cuda.empty_cache()

    return sum(scores) / len(scores)


# ═════════════════════════════════════════════════════════
# Identity Score — 使用 CriticAgent 采样评估
# ═════════════════════════════════════════════════════════

def compute_identity_score(
    gen_dir: str,
    ref_dir: str,
    eval_model_id: str,
    sample_size: int = 200,
    category: str = "bird",
    task_desc: str = "",
) -> dict:
    """
    使用 Qwen VLM 评估生成图的身份一致性。

    返回: {"identity_score": float, "quality_score": float, "samples": int}
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from core.critic import CriticAgent

    # 采样
    gen_images = get_all_images(gen_dir)
    ref_images = get_all_images(ref_dir)

    if len(gen_images) != len(ref_images):
        warnings.warn(f"gen ({len(gen_images)}) != ref ({len(ref_images)}), using min")

    n = min(len(gen_images), len(ref_images), sample_size)
    import random
    random.seed(42)
    indices = random.sample(range(min(len(gen_images), len(ref_images))), n)

    print(f"  Computing Identity Score on {n} samples...")

    agent = CriticAgent({"eval_model_id": eval_model_id})
    agent.load()

    scores = []
    quality_scores = []

    for idx in indices:
        from tools import load_image
        gen_img = load_image(gen_images[idx])
        ref_img = load_image(ref_images[idx])

        result = agent.run(
            gen_img, ref_img,
            category=category,
            task_description=task_desc,
        )
        scores.append(result.identity_score)
        quality_scores.append(result.quality_score)

    agent.unload()

    return {
        "identity_score": sum(scores) / len(scores),
        "quality_score": sum(quality_scores) / len(quality_scores),
        "samples": n,
    }


# ═════════════════════════════════════════════════════════
# 统一接口
# ═════════════════════════════════════════════════════════

class MetricsCalculator:
    """一站式指标计算器"""

    def __init__(self, device: str = "cuda", cache_dir: str = "output/metrics"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.cache_dir = cache_dir
        self._results = {}

    def evaluate_all(
        self,
        real_dir: str,          # 真实图片目录
        gen_dir: str,           # 生成图片目录
        ref_dir: str,           # 参考图片目录（用于 identity）
        task_prompt: str,       # 任务 prompt（用于 CLIP）
        eval_model_id: str = "",  # 评估模型路径（为空则跳过 identity）
        category: str = "bird",
        task_desc: str = "",
        sample_size: int = 200,
        fid: bool = True,
        clip: bool = True,
        identity: bool = True,
    ) -> dict:
        """计算所有指标"""
        results = {}

        if fid:
            results["fid"] = self._run_with_time(
                "FID",
                compute_fid, real_dir, gen_dir, self.cache_dir,
            )

        if clip:
            results["clip_score"] = self._run_with_time(
                "CLIP Score",
                compute_clip_score, gen_dir, task_prompt, sample_size, self.device,
            )

        if identity and eval_model_id:
            id_results = self._run_with_time(
                "Identity Score",
                compute_identity_score, gen_dir, ref_dir, eval_model_id,
                sample_size, category, task_desc,
            )
            results.update(id_results)

        self._results = results
        return results

    def _run_with_time(self, name, func, *args, **kwargs):
        print(f"\n[{name}]")
        import time
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"  -> {name}: {result:.4f}" if isinstance(result, float)
              else f"  -> {name}: {result}")
        print(f"  Time: {elapsed:.1f}s")
        return result

    def save_results(self, path: str):
        """保存指标结果到 JSON"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # 转换 tensor 为 float
        clean = {}
        for k, v in self._results.items():
            if isinstance(v, torch.Tensor):
                clean[k] = v.item()
            else:
                clean[k] = v
        with open(path, "w") as f:
            json.dump(clean, f, indent=2)
        print(f"Results saved to {path}")

    def load_results(self, path: str) -> dict:
        """加载已保存的指标结果"""
        with open(path) as f:
            return json.load(f)


# ═════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute evaluation metrics")
    parser.add_argument("--real-dir", required=True, help="Real images dir")
    parser.add_argument("--gen-dir", required=True, help="Generated images dir")
    parser.add_argument("--ref-dir", required=True, help="Reference images dir")
    parser.add_argument("--prompt", default="A bird in a natural outdoor environment",
                        help="Task prompt for CLIP score")
    parser.add_argument("--eval-model", default="", help="Evaluator model path")
    parser.add_argument("--sample", type=int, default=200, help="Sample size for identity")
    parser.add_argument("--output", default="output/metrics/results.json", help="Output path")
    parser.add_argument("--no-fid", action="store_true", help="Skip FID")
    parser.add_argument("--no-clip", action="store_true", help="Skip CLIP")
    parser.add_argument("--no-identity", action="store_true", help="Skip Identity")
    args = parser.parse_args()

    calc = MetricsCalculator()
    results = calc.evaluate_all(
        real_dir=args.real_dir,
        gen_dir=args.gen_dir,
        ref_dir=args.ref_dir,
        task_prompt=args.prompt,
        eval_model_id=args.eval_model,
        sample_size=args.sample,
        fid=not args.no_fid,
        clip=not args.no_clip,
        identity=not args.no_identity,
    )
    calc.save_results(args.output)
    print(json.dumps(results, indent=2))
