"""
实验分析模块

职责:
  1. 从 Memory JSON 中提取错误分类统计（Error Taxonomy）
  2. 绘制迭代分数曲线（Score vs Iteration）
  3. 绘制各 baseline 的 FID/CLIP 对比柱状图
  4. 生成可读的 Markdown 报告

用法:
  # 从已完成的实验中分析
  python experiments/analysis.py \
      --memory output/baselines/cub_bird/critic_threshold_7/memory.json \
      --baselines-dir output/baselines/cub_bird \
      --output output/report

  # 只看错误分类
  python experiments/analysis.py --memory <path> --error-taxonomy
"""

import os
import sys
import json
import argparse
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═════════════════════════════════════════════════════════
# 错误分类分析
# ═════════════════════════════════════════════════════════

# 预定义的错误类型关键词分类
ERROR_CATEGORIES = {
    "颜色/纹理不匹配": [
        "color", "texture", "pattern", "feather", "fur", "paint",
        "shade", "tone", "hue", "saturation",
    ],
    "视角变形": [
        "viewpoint", "rotation", "angle", "perspective", "flat",
        "2D", "3D", "pose", "stance", "orientation",
    ],
    "解剖结构畸形": [
        "anatomy", "limb", "wing", "leg", "beak", "head", "eye",
        "deform", "distort", "malformed", "extra", "missing",
        "fused", "merged", "duplicate",
    ],
    "图像质量": [
        "blur", "artifact", "pixelat", "noise", "low quality",
        "resolution", "grain", "jpeg",
    ],
    "背景不自然": [
        "background", "environment", "scene", "artificial",
        "unnatural", "plain", "empty", "floating", "pasted",
    ],
    "多主体/拼接": [
        "multiple", "collage", "split", "duplicated", "two birds",
        "three birds", "several",
    ],
    "身份不一致": [
        "identity", "species", "breed", "different", "mismatch",
        "incorrect", "wrong",
    ],
}


def classify_issue(issue_text: str) -> str:
    """将一条 issue 分类到预定义错误类型"""
    text = issue_text.lower()
    for category, keywords in ERROR_CATEGORIES.items():
        if any(kw in text for kw in keywords):
            return category
    return "其他"


def compute_error_taxonomy(memory_path: str) -> dict:
    """
    从 Memory JSON 中提取错误分类统计。

    返回:
      {
        "category_counts": {"颜色/纹理不匹配": 45, ...},
        "category_repair_rates": {"颜色/纹理不匹配": 0.78, ...},
        "per_class_top_errors": {"003.Sooty_Albatross": [...], ...},
        "total_issues": 200,
        "total_images": 6000,
      }
    """
    if not os.path.exists(memory_path):
        raise FileNotFoundError(f"Memory file not found: {memory_path}")

    with open(memory_path) as f:
        memory = json.load(f)

    all_issues = []
    # issues that appeared in first attempt AND were fixed by last attempt
    fixed_issues = []
    not_fixed_issues = []
    per_class_errors = defaultdict(list)

    for cls_name, cls_data in memory.get("classes", {}).items():
        for img_name, img_data in cls_data.items():
            attempts = img_data.get("attempts", [])
            accepted = img_data.get("accepted", False)

            # 收集所有 issue
            for attempt in attempts:
                for issue in attempt.get("issues", []):
                    if issue:
                        cat = classify_issue(issue)
                        all_issues.append(cat)

            # 修复率: 首轮 issue 是否在最终被修复
            if len(attempts) >= 2:
                first_issues = set()
                for issue in attempts[0].get("issues", []):
                    if issue:
                        first_issues.add(classify_issue(issue))

                last_issues = set()
                for issue in attempts[-1].get("issues", []):
                    if issue:
                        last_issues.add(classify_issue(issue))

                for cat in first_issues:
                    if cat not in last_issues and accepted:
                        fixed_issues.append(cat)
                    elif cat in last_issues:
                        not_fixed_issues.append(cat)
                    per_class_errors[cls_name].append(cat)

    # 统计
    category_counts = Counter(all_issues)
    category_repair_rates = {}
    for cat in category_counts:
        fixed_count = fixed_issues.count(cat)
        total = fixed_count + not_fixed_issues.count(cat)
        category_repair_rates[cat] = fixed_count / total if total > 0 else 0

    return {
        "category_counts": dict(category_counts.most_common()),
        "category_repair_rates": {k: round(v, 3) for k, v in category_repair_rates.items()},
        "per_class_top_errors": {
            cls: [e[0] for e in Counter(errs).most_common(3)]
            for cls, errs in per_class_errors.items()
        },
        "total_issues": len(all_issues),
        "total_images": sum(
            len(cls_data) for cls_data in memory.get("classes", {}).values()
        ),
    }


# ═════════════════════════════════════════════════════════
# 指标对比表
# ═════════════════════════════════════════════════════════

def load_baseline_results(baselines_dir: str) -> dict:
    """加载目录下所有 baseline 的指标结果"""
    results = {}
    baselines_dir = Path(baselines_dir)
    for f in sorted(baselines_dir.glob("*/metrics.json")):
        name = f.parent.name
        with open(f) as fh:
            results[name] = json.load(fh)
    # 也尝试加载 summary
    summary_path = baselines_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            results["_summary"] = json.load(f)
    return results


def generate_comparison_table(results: dict) -> str:
    """生成 Markdown 对比表"""
    lines = [
        "| Baseline | FID ↓ | CLIP ↑ | Identity ↑ | 接受率 | 平均重试 |",
        "|----------|-------|--------|------------|--------|----------|",
    ]
    for name, r in results.items():
        if name.startswith("_"):
            continue
        fid = f"{r.get('fid', 'N/A'):.2f}" if r.get('fid') else "N/A"
        clip = f"{r.get('clip_score', 'N/A'):.2f}" if r.get('clip_score') else "N/A"
        identity = f"{r.get('identity_score', 'N/A'):.2f}" if r.get('identity_score') else "N/A"
        lines.append(f"| {name} | {fid} | {clip} | {identity} | — | — |")
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════
# 报告生成
# ═════════════════════════════════════════════════════════

def generate_report(
    error_taxonomy: dict,
    baseline_results: dict,
    output_dir: str = "output/report",
) -> str:
    """生成完整实验报告的 Markdown"""
    os.makedirs(output_dir, exist_ok=True)

    lines = [
        "# 实验报告",
        "",
        "## 1. 指标对比",
        "",
    ]

    # 指标表
    if baseline_results:
        lines.append(generate_comparison_table(baseline_results))
        lines.append("")

    # 错误分类
    if error_taxonomy:
        lines.extend([
            "## 2. 错误分类统计",
            "",
            f"共分析 {error_taxonomy['total_images']} 张图片，"
            f"发现 {error_taxonomy['total_issues']} 个问题。",
            "",
            "| 错误类型 | 出现次数 | 占比 | 修复率 |",
            "|----------|---------|------|--------|",
        ])
        total = error_taxonomy["total_issues"]
        for cat, count in error_taxonomy["category_counts"].items():
            pct = count / total * 100 if total > 0 else 0
            repair = error_taxonomy["category_repair_rates"].get(cat, 0)
            repair_str = f"{repair:.0%}" if isinstance(repair, float) else "N/A"
            lines.append(f"| {cat} | {count} | {pct:.1f}% | {repair_str} |")

        lines.append("")

    report = "\n".join(lines)

    report_path = os.path.join(output_dir, "experiment_report.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved to {report_path}")
    return report


# ═════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze experiment results and generate reports"
    )
    parser.add_argument("--memory", default="",
                        help="Path to memory.json for error taxonomy")
    parser.add_argument("--baselines-dir", default="",
                        help="Directory containing baseline result subdirs")
    parser.add_argument("--output", default="output/report",
                        help="Output directory for report and plots")
    parser.add_argument("--error-taxonomy", action="store_true",
                        help="Only show error taxonomy")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    error_taxonomy = {}
    baseline_results = {}

    # 错误分类
    if args.memory or args.error_taxonomy:
        mem_path = args.memory or input("Memory JSON path: ")
        print(f"\nComputing error taxonomy from {mem_path}...")
        error_taxonomy = compute_error_taxonomy(mem_path)
        print(f"  Total images: {error_taxonomy['total_images']}")
        print(f"  Total issues: {error_taxonomy['total_issues']}")
        print(f"\n  Error breakdown:")
        for cat, count in error_taxonomy["category_counts"].items():
            pct = count / error_taxonomy["total_issues"] * 100
            repair = error_taxonomy["category_repair_rates"].get(cat, 0)
            print(f"    {cat:20s} {count:4d} ({pct:5.1f}%)  fix rate: {repair:.0%}")

        # 保存
        tax_path = os.path.join(args.output, "error_taxonomy.json")
        with open(tax_path, "w") as f:
            json.dump(error_taxonomy, f, indent=2)
        print(f"\nTaxonomy saved to {tax_path}")

    # 加载 baseline 结果
    if args.baselines_dir and os.path.exists(args.baselines_dir):
        print(f"\nLoading baseline results from {args.baselines_dir}...")
        baseline_results = load_baseline_results(args.baselines_dir)
        for name in baseline_results:
            if not name.startswith("_"):
                print(f"  {name}: FID={baseline_results[name].get('fid', 'N/A')}")

    # 生成报告
    if error_taxonomy or baseline_results:
        report = generate_report(error_taxonomy, baseline_results, args.output)
        print(f"\nFirst 500 chars of report:\n{report[:500]}")

    print("\nDone!")


if __name__ == "__main__":
    main()
