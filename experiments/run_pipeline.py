"""
实验入口：运行 Multi-Agent Pipeline

用法:
  # 使用 CUB 数据集（默认）
  python experiments/run_pipeline.py

  # 指定任务
  python experiments/run_pipeline.py --task cub_bird

  # 使用 Critic（需先设置 EVAL_MODEL_ID）
  python experiments/run_pipeline.py --task cub_bird --use-critic

  # 覆盖路径
  python experiments/run_pipeline.py \
      --task cub_bird \
      --input /path/to/data \
      --output /path/to/output

  # 覆盖阈值 / 重试次数
  python experiments/run_pipeline.py --threshold 8.0 --max-retries 5
"""

import os
import sys
import argparse

# 将项目根目录加入 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import OrchestratorAgent
from tasks import get_task, list_tasks


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Pipeline for FGVC Image Generation"
    )
    parser.add_argument("--task", type=str, default="cub_bird",
                        choices=list_tasks(),
                        help="Task configuration to use")
    parser.add_argument("--input", type=str, default=None,
                        help="Override input root path")
    parser.add_argument("--output", type=str, default=None,
                        help="Override output root path")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Minimum acceptable score (0-10)")
    parser.add_argument("--max-retries", type=int, default=None,
                        help="Maximum retry attempts per image")
    parser.add_argument("--use-critic", action="store_true",
                        help="Enable CriticAgent evaluation (requires evaluator model)")
    parser.add_argument("--memory-path", type=str, default=None,
                        help="Path to memory JSON file")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Path to log file")
    parser.add_argument("--skip-upto", type=int, default=None,
                        help="Skip first N classes")
    parser.add_argument("--gen-model", type=str, default=None,
                        help="Override generator model ID")
    parser.add_argument("--eval-model", type=str, default=None,
                        help="Override evaluator model ID")
    parser.add_argument("--list-tasks", action="store_true",
                        help="List all available tasks and exit")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_tasks:
        print("Available tasks:")
        for name in list_tasks():
            task_cls = get_task(name)
            print(f"  {name:20s} {task_cls.category_name}s, "
                  f"{task_cls.num_classes} classes")
        return

    # 加载任务配置
    task_config_dict = {}

    # 从命令行覆盖
    if args.input:
        task_config_dict["input_root"] = args.input
    if args.output:
        task_config_dict["output_root"] = args.output
    if args.skip_upto is not None:
        task_config_dict["skip_upto"] = args.skip_upto
    if args.gen_model:
        task_config_dict["gen_model_id"] = args.gen_model
    if args.eval_model:
        task_config_dict["eval_model_id"] = args.eval_model

    task = get_task(args.task, **task_config_dict)

    # 构建 pipeline 配置
    pipeline_config = {
        "gen_model_id": task.gen_model_id if hasattr(task, "gen_model_id") else "../Qwen-Image-Edit-2511",
        "eval_model_id": task.eval_model_id if hasattr(task, "eval_model_id") else "",
        "steps": 28,
        "cfg": 3.5,
        "seed": 42,
        "device_map": "balanced",
        "max_memory": {0: "46GiB", 1: "46GiB"},
        "threshold": args.threshold if args.threshold is not None else 7.0,
        "max_retries": args.max_retries if args.max_retries is not None else 3,
        "use_critic": args.use_critic,
        "memory_path": args.memory_path or f"output/memory/{task.name}.json",
        "log_file": args.log_file or f"output/logs/{task.name}.log",
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

    print(f"Task:         {task.name}")
    print(f"Input:        {task.input_root}")
    print(f"Output:       {task.output_root}")
    print(f"Threshold:    {pipeline_config['threshold']}")
    print(f"Max retries:  {pipeline_config['max_retries']}")
    print(f"Use critic:   {pipeline_config['use_critic']}")
    print()

    # 运行
    orchestrator = OrchestratorAgent(pipeline_config)
    orchestrator.run_pipeline(task)


if __name__ == "__main__":
    main()
