"""
基线对比实验

对每个 baseline 配置:
  1. 运行 pipeline（生成所有图片）
  2. 计算指标（FID / CLIP Score / Identity Score）
  3. 从 Memory 中提取统计信息（接受率、重试次数、分数提升）
  4. 汇总到 JSON 结果文件

用法:
  # 查看实验计划（不实际运行）
  CUDA_VISIBLE_DEVICES=2,3 python experiments/run_baselines.py --dry-run

  # 运行单个 baseline
  CUDA_VISIBLE_DEVICES=2,3 python experiments/run_baselines.py \
      --baseline no_feedback --task cub_bird

  # 运行所有 baselines
  CUDA_VISIBLE_DEVICES=2,3 python experiments/run_baselines.py \
      --all --task cub_bird

  # 只计算指标（跳过生成，适用于已有输出）
  CUDA_VISIBLE_DEVICES=2,3 python experiments/run_baselines.py \
      --all --metrics-only
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import OrchestratorAgent
from tasks import get_task, list_tasks
from experiments.configs import ALL_BASELINES, get_baseline_output_dir
from experiments.metrics import MetricsCalculator


# ── 默认 pipeline 配置（除 baseline 可变参数外的固定部分） ──

BASE_PIPELINE_CONFIG = {
    "steps": 28,
    "cfg": 3.5,
    "seed": 42,
    "device_map": "balanced",
    "max_memory": {0: "46GiB", 1: "46GiB"},
    "cache_options": {
        "cache_type": "DBCache",
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
    },
}


def build_pipeline_config(baseline, baseline_output_dir, eval_model_id):
    """为 baseline 构建 pipeline 配置"""
    return {
        **BASE_PIPELINE_CONFIG,
        "threshold": baseline.threshold,
        "max_retries": baseline.max_retries,
        "use_critic": baseline.use_critic,
        "eval_model_id": eval_model_id,
        "memory_path": os.path.join(baseline_output_dir, "memory.json"),
        "log_file": os.path.join(baseline_output_dir, "pipeline.log"),
    }


def run_one_baseline(task, baseline, gen_model_id="", eval_model_id=""):
    """运行单个 baseline 的生成"""
    output_dir = get_baseline_output_dir(task.output_root, baseline.name)
    os.makedirs(output_dir, exist_ok=True)

    # 保存 baseline 配置
    config_path = os.path.join(output_dir, "baseline_config.json")
    with open(config_path, "w") as f:
        json.dump({
            "name": baseline.name,
            "use_critic": baseline.use_critic,
            "max_retries": baseline.max_retries,
            "threshold": baseline.threshold,
            "description": baseline.description,
            "gen_model_id": gen_model_id,
            "eval_model_id": eval_model_id,
        }, f, indent=2)

    # 修改 task 的输出路径
    task.output_root = output_dir

    pipeline_config = build_pipeline_config(baseline, output_dir, eval_model_id)
    pipeline_config["gen_model_id"] = gen_model_id or task.gen_model_id if hasattr(task, "gen_model_id") else ""

    print(f"\n{'=' * 60}")
    print(f"Baseline: {baseline.name}")
    print(f"  {baseline.description}")
    print(f"  Output: {output_dir}")
    print(f"  use_critic={baseline.use_critic}, "
          f"max_retries={baseline.max_retries}, threshold={baseline.threshold}")
    print(f"{'=' * 60}")

    orchestrator = OrchestratorAgent(pipeline_config)
    orchestrator.run_pipeline(task)

    # 返回统计信息
    stats = orchestrator._stats.copy()
    mem_stats = orchestrator.memory.get_all_stats()
    return {**stats, "memory": mem_stats}


def compute_metrics_for_baseline(baseline, task, eval_model_id=""):
    """计算 baseline 的指标"""
    output_dir = get_baseline_output_dir(task.output_root, baseline.name)

    # 从 task config 中获取参考目录和 prompt
    ref_dir = task.input_root  # 参考图就是输入图
    gen_dir = output_dir       # 生成的图在 output_dir 下按类别组织

    # 将不同类别的生成图收集到一个目录（或按类别分别计算）
    # 简化：使用类别 000 的生成结果作为样本
    # 更好的方式：将所有类别合并后计算 FID

    print(f"\n  Computing metrics for {baseline.name}...")

    calc = MetricsCalculator(cache_dir=f"output/metrics/{baseline.name}")
    results = {"baseline": baseline.name}

    # FID — 使用真实图片目录 vs 生成图片
    try:
        results["fid"] = calc._run_with_time(
            "FID",
            compute_fid_simple, task.input_root, output_dir, calc.cache_dir,
        )
    except Exception as e:
        print(f"  FID failed: {e}")
        results["fid"] = None

    # CLIP Score — 采样子集
    try:
        results["clip_score"] = calc._run_with_time(
            "CLIP Score",
            compute_clip_score_simple, output_dir, task.task_description,
            sample_size=500,
        )
    except Exception as e:
        print(f"  CLIP failed: {e}")
        results["clip_score"] = None

    # Identity Score — 使用 Critic 采样
    if eval_model_id and baseline.use_critic:
        try:
            id_results = calc._run_with_time(
                "Identity Score",
                compute_identity_simple, output_dir, task.input_root,
                eval_model_id, task.category_name, task.task_description,
                sample_size=100,
            )
            results.update(id_results)
        except Exception as e:
            print(f"  Identity failed: {e}")

    # 保存
    calc.save_results(os.path.join(output_dir, "metrics.json"))
    return results


def compute_fid_simple(real_dir, gen_dir, cache_dir):
    from experiments.metrics import compute_fid
    return compute_fid(real_dir, gen_dir, cache_dir)


def compute_clip_score_simple(gen_dir, prompt, sample_size=500):
    from experiments.metrics import compute_clip_score
    return compute_clip_score(gen_dir, prompt, sample_size)


def compute_identity_simple(gen_dir, ref_dir, eval_model_id, category, task_desc, sample_size=100):
    from experiments.metrics import compute_identity_score
    return compute_identity_score(gen_dir, ref_dir, eval_model_id, sample_size, category, task_desc)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run baseline experiments for the multi-agent pipeline"
    )
    parser.add_argument("--task", default="cub_bird", choices=list_tasks(),
                        help="Task configuration")
    parser.add_argument("--baseline", default="",
                        help="Run a single baseline (name)")
    parser.add_argument("--all", action="store_true",
                        help="Run all baselines")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show experiment plan without running")
    parser.add_argument("--metrics-only", action="store_true",
                        help="Skip generation, only compute metrics from existing outputs")
    parser.add_argument("--gen-model", default="",
                        help="Generator model ID")
    parser.add_argument("--eval-model", default="",
                        help="Evaluator model ID")
    parser.add_argument("--output", default="output/baselines",
                        help="Results aggregation directory")
    parser.add_argument("--skip-upto", type=int, default=None,
                        help="Skip classes with numeric prefix <= N (e.g. --skip-upto 53 skips 001-053)")
    return parser.parse_args()


def main():
    args = parse_args()

    # 加载任务
    task_config_dict = {}
    if args.skip_upto is not None:
        task_config_dict["skip_upto"] = args.skip_upto
    task = get_task(args.task, **task_config_dict)

    # 确定要运行的 baselines
    if args.baseline:
        from experiments.configs import get_baseline
        baselines = [get_baseline(args.baseline)]
    elif args.all:
        baselines = ALL_BASELINES
    else:
        print("Specify --baseline NAME or --all")
        return

    if args.dry_run:
        print(f"\nExperiment Plan: {args.task}")
        print(f"{'=' * 60}")
        for b in baselines:
            output = get_baseline_output_dir(task.output_root, b.name)
            print(f"  [{b.name:25s}] → {output}")
            print(f"  {'':25s}  {b.description}")
            print()
        print(f"Total: {len(baselines)} baselines")
        return

    # 聚合结果
    all_results = {}
    aggregated_dir = os.path.join(args.output, args.task)
    os.makedirs(aggregated_dir, exist_ok=True)

    for baseline in baselines:
        print(f"\n{'#' * 60}")
        print(f"# Baseline: {baseline.name}")
        print(f"{'#' * 60}")

        if not args.metrics_only:
            run_one_baseline(task, baseline, args.gen_model, args.eval_model)

        results = compute_metrics_for_baseline(baseline, task, args.eval_model)
        all_results[baseline.name] = results

        # 保存中间结果
        with open(os.path.join(aggregated_dir, f"{baseline.name}.json"), "w") as f:
            json.dump(results, f, indent=2)

    # 保存汇总
    summary_path = os.path.join(aggregated_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"All baselines completed!")
    print(f"Results saved to {summary_path}")
    print(f"{'=' * 60}")

    # 打印汇总表
    print_table(all_results)


def print_table(results):
    """打印指标对比表"""
    print(f"\n{'─' * 60}")
    print(f"{'Baseline':25s} {'FID':>8s} {'CLIP':>8s} {'Identity':>8s}")
    print(f"{'─' * 60}")
    for name, r in results.items():
        fid = f"{r.get('fid', 'N/A'):.2f}" if r.get('fid') else "N/A"
        clip = f"{r.get('clip_score', 'N/A'):.2f}" if r.get('clip_score') else "N/A"
        identity = f"{r.get('identity_score', 'N/A'):.2f}" if r.get('identity_score') else "N/A"
        print(f"{name:25s} {fid:>8s} {clip:>8s} {identity:>8s}")
    print(f"{'─' * 60}")


if __name__ == "__main__":
    main()
