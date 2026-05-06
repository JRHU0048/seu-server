"""
可视化模块 — 论文级图表生成

生成 5 类图表:
  1. Score 迭代曲线    — 个体图片的分数随重试变化
  2. FID 对比柱状图     — 各 baseline 的 FID 对比
  3. 错误分类分布       — 错误类型频率 + 修复率
  4. Case Study 拼图   — 参考图 + 各轮生成结果
  5. Score 分布直方图   — 最终分数的分布

用法:
  python experiments/visualize.py --memory <path> --baselines-dir <path>
"""

import os
import sys
import json
import argparse
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")  # 无头模式
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from experiments.analysis import compute_error_taxonomy, load_baseline_results

# ── 全局样式 ──

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.figsize": (8, 5),
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ═════════════════════════════════════════════════════════
# 图 1: Score 迭代曲线
# ═════════════════════════════════════════════════════════

def plot_score_curves(memory_path: str, output_path: str = "output/figures/score_curves.png",
                      max_images: int = 12):
    """
    在单张图上绘制多个代表性样本的 Score 迭代曲线。
    每张子图显示一张图片的 score progression。
    """
    if not os.path.exists(memory_path):
        raise FileNotFoundError(f"Memory file not found: {memory_path}")

    with open(memory_path) as f:
        memory = json.load(f)

    # 收集所有有多次尝试的图片
    trajectories = []
    for cls_name, cls_data in memory.get("classes", {}).items():
        for img_name, img_data in cls_data.items():
            attempts = img_data.get("attempts", [])
            scores = [a["score"] for a in attempts if "score" in a]
            if len(scores) >= 2:  # 至少 2 次尝试
                trajectories.append({
                    "class": cls_name,
                    "name": img_name,
                    "scores": scores,
                    "n_attempts": len(scores),
                    "accepted": img_data.get("accepted", False),
                })

    if not trajectories:
        print("No images with multiple attempts found.")
        return

    # 按尝试次数排序，选最长的几个
    trajectories.sort(key=lambda x: -x["n_attempts"])
    selected = trajectories[:max_images]

    # 绘制
    n_cols = min(4, len(selected))
    n_rows = (len(selected) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.5 * n_rows))
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

    for ax, t in zip(axes, selected):
        ax.plot(range(1, len(t["scores"]) + 1), t["scores"],
                "o-", color="#2E86AB", linewidth=1.5, markersize=4)
        ax.axhline(y=7.0, color="#A23B72", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_xlabel("Attempt")
        ax.set_ylabel("Score")
        ax.set_title(f"{t['class'][:20]}/{t['name'][:15]}", fontsize=8)
        ax.set_ylim(0, 10.5)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(2))

    # 隐藏多余的子图
    for ax in axes[len(selected):]:
        ax.set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Score curves saved to {output_path}")


# ═════════════════════════════════════════════════════════
# 图 2: FID/CLIP 对比柱状图
# ═════════════════════════════════════════════════════════

def plot_metrics_comparison(
    baselines_dir: str,
    output_path: str = "output/figures/metrics_comparison.png",
):
    """各 baseline 的 FID 和 CLIP Score 对比柱状图"""
    results = load_baseline_results(baselines_dir)

    baselines = []
    fid_values = []
    clip_values = []

    for name, r in results.items():
        if name.startswith("_"):
            continue
        baselines.append(name)
        fid_values.append(r.get("fid", 0) or 0)
        clip_values.append(r.get("clip_score", 0) or 0)

    if not baselines:
        print("No baseline results found.")
        return

    colors_fid = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#4A7C59"]
    colors_clip = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#4A7C59"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # FID（越低越好）
    bars1 = ax1.bar(range(len(baselines)), fid_values,
                    color=colors_fid[:len(baselines)], width=0.5)
    ax1.set_xticks(range(len(baselines)))
    ax1.set_xticklabels(baselines, rotation=30, ha="right", fontsize=8)
    ax1.set_ylabel("FID \u2193")
    ax1.set_title("FID (lower is better)")
    for bar, v in zip(bars1, fid_values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    # CLIP（越高越好）
    bars2 = ax2.bar(range(len(baselines)), clip_values,
                    color=colors_clip[:len(baselines)], width=0.5)
    ax2.set_xticks(range(len(baselines)))
    ax2.set_xticklabels(baselines, rotation=30, ha="right", fontsize=8)
    ax2.set_ylabel("CLIP Score \u2191")
    ax2.set_title("CLIP Score (higher is better)")
    for bar, v in zip(bars2, clip_values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Metrics comparison saved to {output_path}")


# ═════════════════════════════════════════════════════════
# 图 3: 错误分类分布
# ═════════════════════════════════════════════════════════

def plot_error_taxonomy(
    memory_path: str,
    output_path: str = "output/figures/error_taxonomy.png",
):
    """错误类型频率（条形图）+ 修复率（叠加点）"""
    taxonomy = compute_error_taxonomy(memory_path)

    categories = list(taxonomy["category_counts"].keys())
    counts = list(taxonomy["category_counts"].values())
    repair_rates = [taxonomy["category_repair_rates"].get(c, 0) for c in categories]

    if not categories:
        print("No errors found in memory.")
        return

    fig, ax1 = plt.subplots(figsize=(8, 4))

    # 条形图：频率
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(categories)))
    bars = ax1.barh(range(len(categories)), counts, color=colors, height=0.6)
    ax1.set_yticks(range(len(categories)))
    ax1.set_yticklabels(categories, fontsize=9)
    ax1.set_xlabel("Count")
    ax1.set_title("Error Type Distribution & Repair Rate")

    # 条上标注数字
    for bar, v in zip(bars, counts):
        ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                 str(v), ha="left", va="center", fontsize=8)

    # 叠加线：修复率
    ax2 = ax1.twiny()
    ax2.plot(repair_rates, range(len(categories)),
             "o-", color="#A23B72", linewidth=1.5, markersize=6)
    ax2.set_xlabel("Repair Rate \u2192", color="#A23B72")
    ax2.tick_params(axis="x", colors="#A23B72")
    ax2.set_xlim(0, 1.1)

    # 修复率标注
    for i, (rr, cat) in enumerate(zip(repair_rates, categories)):
        ax2.text(rr + 0.03, i, f"{rr:.0%}", color="#A23B72",
                 va="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Error taxonomy saved to {output_path}")


# ═════════════════════════════════════════════════════════
# 图 4: Case Study 拼图
# ═════════════════════════════════════════════════════════

def plot_case_study(
    memory_path: str,
    gen_root: str,
    ref_root: str,
    output_path: str = "output/figures/case_study.png",
    n_cases: int = 6,
):
    """
    从 Memory 中选取有代表性的样本，生成参考图 + 各轮尝试的拼图。

    布局: n_cases 行 × 4 列 [Reference, Attempt 1, Attempt 2, Attempt 3]
    每张图下方标注 score。
    """
    if not os.path.exists(memory_path):
        raise FileNotFoundError(f"Memory file not found: {memory_path}")

    with open(memory_path) as f:
        memory = json.load(f)

    # 选取有多次尝试且被接受的图片
    candidates = []
    for cls_name, cls_data in memory.get("classes", {}).items():
        for img_name, img_data in cls_data.items():
            attempts = img_data.get("attempts", [])
            if len(attempts) >= 2 and img_data.get("accepted", False):
                candidates.append({
                    "class": cls_name,
                    "name": img_name,
                    "attempts": attempts,
                })

    if not candidates:
        print("No suitable candidates found in memory.")
        return

    # 选分数提升最大的
    candidates.sort(
        key=lambda x: x["attempts"][-1]["score"] - x["attempts"][0]["score"],
        reverse=True,
    )
    selected = candidates[:n_cases]

    max_cols = 4  # ref + 最多 3 次 attempt
    fig, axes = plt.subplots(
        len(selected), max_cols,
        figsize=(max_cols * 2.5, len(selected) * 2.5),
    )

    if len(selected) == 1:
        axes = axes.reshape(1, -1)

    for row_idx, case in enumerate(selected):
        # 参考图
        ref_dir = os.path.join(ref_root, case["class"])
        ref_candidates = sorted(Path(ref_dir).glob(f"{case['name']}.*"))
        ref_path = str(ref_candidates[0]) if ref_candidates else ""

        if ref_path and os.path.exists(ref_path):
            ref_img = Image.open(ref_path).convert("RGB")
            axes[row_idx][0].imshow(ref_img)
        axes[row_idx][0].set_title(f"Reference\n{case['class'][:20]}", fontsize=7)
        axes[row_idx][0].axis("off")

        # 各轮生成图
        gen_dir = os.path.join(gen_root, case["class"], "_attempts", case["name"])
        for col_idx, attempt in enumerate(case["attempts"]):
            if col_idx >= max_cols - 1:
                break
            col = col_idx + 1

            attempt_path = os.path.join(gen_dir, f"attempt_{attempt['attempt']}.png")
            if os.path.exists(attempt_path):
                img = Image.open(attempt_path).convert("RGB")
                axes[row_idx][col].imshow(img)
            else:
                axes[row_idx][col].text(0.5, 0.5, "N/A",
                                       ha="center", va="center", fontsize=8)

            score = attempt.get("score", 0)
            label = f"Attempt {attempt['attempt'] + 1}\nScore: {score:.1f}"
            if attempt.get("accepted", False):
                label += " \u2705"
            axes[row_idx][col].set_title(label, fontsize=7)
            axes[row_idx][col].axis("off")

        # 隐藏多余列
        for col in range(len(case["attempts"]) + 1, max_cols):
            axes[row_idx][col].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Case study saved to {output_path}")


# ═════════════════════════════════════════════════════════
# 图 5: Score 分布直方图
# ═════════════════════════════════════════════════════════

def plot_score_distribution(
    memory_path: str,
    output_path: str = "output/figures/score_distribution.png",
):
    """最终得分的分布直方图"""
    if not os.path.exists(memory_path):
        raise FileNotFoundError(f"Memory file not found: {memory_path}")

    with open(memory_path) as f:
        memory = json.load(f)

    final_scores = []
    all_scores = []
    for cls_name, cls_data in memory.get("classes", {}).items():
        for img_name, img_data in cls_data.items():
            attempts = img_data.get("attempts", [])
            for a in attempts:
                all_scores.append(a.get("score", 0))
            if attempts:
                final_scores.append(attempts[-1].get("score", 0))

    if not final_scores:
        print("No scores found.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # 最终分数分布
    ax1.hist(final_scores, bins=20, color="#2E86AB", edgecolor="white", alpha=0.8)
    ax1.axvline(x=7.0, color="#A23B72", linestyle="--", linewidth=1, label="Threshold=7")
    ax1.set_xlabel("Final Score")
    ax1.set_ylabel("Count")
    ax1.set_title(f"Final Score Distribution (n={len(final_scores)})")
    ax1.legend()

    # 所有分数分布
    ax2.hist(all_scores, bins=20, color="#F18F01", edgecolor="white", alpha=0.8)
    ax2.axvline(x=7.0, color="#A23B72", linestyle="--", linewidth=1)
    ax2.set_xlabel("Score")
    ax2.set_ylabel("Count")
    ax2.set_title(f"All Attempt Scores (n={len(all_scores)})")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Score distribution saved to {output_path}")


# ═════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Generate paper-ready figures")
    parser.add_argument("--memory", default="", help="Path to memory.json")
    parser.add_argument("--baselines-dir", default="", help="Baseline results dir")
    parser.add_argument("--gen-root", default="", help="Generated images root")
    parser.add_argument("--ref-root", default="", help="Reference images root")
    parser.add_argument("--output", default="output/figures", help="Output dir")
    parser.add_argument("--all", action="store_true", help="Generate all figures")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    if args.memory:
        print("\n--- Score Curves ---")
        plot_score_curves(args.memory, os.path.join(args.output, "score_curves.png"))

        print("\n--- Error Taxonomy ---")
        plot_error_taxonomy(args.memory, os.path.join(args.output, "error_taxonomy.png"))

        print("\n--- Score Distribution ---")
        plot_score_distribution(args.memory, os.path.join(args.output, "score_distribution.png"))

        if args.gen_root and args.ref_root:
            print("\n--- Case Study ---")
            plot_case_study(args.memory, args.gen_root, args.ref_root,
                           os.path.join(args.output, "case_study.png"))

    if args.baselines_dir:
        print("\n--- Metrics Comparison ---")
        plot_metrics_comparison(args.baselines_dir,
                               os.path.join(args.output, "metrics_comparison.png"))

    print(f"\nAll figures saved to {args.output}/")


if __name__ == "__main__":
    main()
