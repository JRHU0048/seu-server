"""
OrchestratorAgent — Pipeline 编排器

职责:
  1. 管理所有 Agent 的生命周期
  2. 控制 Generate → Critic → Refine → Retry 循环
  3. 通过 Memory 模块实现断点续跑和经验积累
  4. 记录统计日志

使用:
  orchestrator = OrchestratorAgent(config)
  orchestrator.run(task)
"""

import os
import time
from pathlib import Path
from PIL import Image

import torch

from .generator import GeneratorAgent
from .critic import CriticAgent
from .refiner import RefinerAgent
from .memory import MemoryModule
from .agent import BaseAgent

from tools import get_all_images, load_image


class OrchestratorAgent(BaseAgent):
    """Pipeline 编排器（无模型，纯控制逻辑）"""

    def __init__(self, config: dict):
        super().__init__(config)

        self.generator = GeneratorAgent(config)
        self.critic = None  # 惰性加载
        self.refiner = RefinerAgent(config)
        self.memory = MemoryModule(
            config.get("memory_path", "output/memory/memory.json")
        )

        self._log_file = config.get("log_file", "output/logs/pipeline.log")
        Path(self._log_file).parent.mkdir(parents=True, exist_ok=True)

        # 统计
        self._stats = {"total": 0, "accepted": 0, "skipped": 0, "attempts": 0}

    def _load_model(self):
        """加载 Generator（Orchestrator 启动时预热）"""
        self.generator.load()

    def _run_model(self, task):
        """运行完整 pipeline"""
        self._run_pipeline(task)

    # ── 主循环 ─────────────────────────────────────────

    def run_pipeline(self, task):
        """依次加载 Agent → 处理所有类别 → 卸载"""
        cfg = self.config

        self._log("=" * 60)
        self._log(f"Pipeline started — Task: {task.name}")
        self._log(f"  Threshold: {cfg.get('threshold', 7.0)}, "
                  f"Max retries: {cfg.get('max_retries', 3)}")
        self._log(f"  Input:  {task.input_root}")
        self._log(f"  Output: {task.output_root}")
        self._log("=" * 60)

        try:
            # 加载 Generator
            self._log("[Orchestrator] Loading Generator...")
            self.generator.load()

            # 可选择是否加载 Critic
            use_critic = cfg.get("use_critic", False)
            if use_critic:
                self._log("[Orchestrator] Loading Critic...")
                self.critic = CriticAgent(cfg)
                self.critic.load()

            # 遍历类别
            class_folders = sorted(os.listdir(task.input_root))
            for cls in class_folders:
                self._process_class(task, cls)

        finally:
            self._cleanup()

        self._log_stats()

    def _process_class(self, task, class_name: str):
        """处理单个类别"""
        # 跳过已处理类别
        if getattr(task, "skip_class", lambda _: False)(class_name):
            self._log(f"Skip class: {class_name}")
            return

        input_dir = os.path.join(task.input_root, class_name)
        output_dir = os.path.join(task.output_root, class_name)
        os.makedirs(output_dir, exist_ok=True)

        images = get_all_images(input_dir)
        if len(images) < task.min_input_images:
            self._log(f"Skip {class_name} (need >= {task.min_input_images} images)")
            return

        self._log(f"\n{'─' * 50}")
        self._log(f"Class: {class_name}  ({len(images)} images)")

        for inputs in task.iter_inputs(images):
            self._process_one(task, class_name, output_dir, inputs)

    def _process_one(self, task, class_name, output_dir, inputs):
        """处理单张/组图片"""
        img_name = inputs["name"]
        output_path = os.path.join(output_dir, f"{img_name}_{task.output_suffix}.png")

        self._stats["total"] += 1

        # Memory 检查：已成功则跳过
        if self.memory.is_done(class_name, img_name):
            self._log(f"  [{self._stats['total']}] {img_name} — already done, skip")
            self._stats["accepted"] += 1
            return

        self._log(f"\n  [{self._stats['total']}] {class_name}/{img_name}")

        # 准备输入
        input_pils = [load_image(p, task.image_size) for p in inputs["paths"]]
        ref_pil = load_image(inputs["paths"][0], task.image_size)
        neg = task.neg_prompt

        # 重试循环
        feedback_history = []
        accepted = False

        for attempt in range(1 + self.config.get("max_retries", 3)):
            prompt = task.build_prompt(feedback_history if attempt > 0 else None)

            self._log(f"    ── Attempt {attempt + 1} ──")

            # Step 1: 生成
            gen_img = self.generator.run(
                input_pils, prompt, neg,
                seed=self.config.get("seed", 42) + attempt,
            )

            # Step 2: 评估
            if self.critic is not None:
                result = self.critic.run(
                    gen_img, ref_pil,
                    criteria=task.evaluation_criteria,
                    category=task.category_name,
                    task_description=task.task_description,
                )
                score = result.global_score
                self._log(f"    Score: {score:.1f}/10")
                self._log(f"    Issues: {result.issues[:3]}")

                # 记录到 Memory
                self.memory.record_attempt(
                    class_name, img_name, attempt, score,
                    result.issues[:3], score >= self.config.get("threshold", 7.0),
                )
            else:
                # 无 Critic → 直接通过
                score = self.config.get("threshold", 7.0) + 1
                result = None

            # Step 3: 判断
            if score >= self.config.get("threshold", 7.0):
                gen_img.save(output_path)
                self._log(f"    \u2705 Accepted")
                self._stats["accepted"] += 1
                self._stats["attempts"] += attempt
                accepted = True
                break
            else:
                if attempt < self.config.get("max_retries", 3):
                    # 收集反馈
                    if result:
                        for issue in result.issues:
                            if issue not in feedback_history:
                                feedback_history.append(issue)
                        for sugg in result.suggestions:
                            if sugg not in feedback_history:
                                feedback_history.append(sugg)
                    self._log(f"    \u2b07 Below threshold, will retry")
                    torch.cuda.empty_cache()
                else:
                    self._log(f"    \u274c All attempts failed")

        if not accepted:
            self._stats["skipped"] += 1

        # 定期输出进度
        if self._stats["total"] % 20 == 0:
            self._log(
                f"\n  [Progress] {self._stats['total']} processed — "
                f"{self._stats['accepted']} accepted, {self._stats['skipped']} skipped"
            )

    # ── 辅助 ─────────────────────────────────────────────

    def _log(self, msg: str):
        msg = str(msg)
        print(msg)
        with open(self._log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def _log_stats(self):
        self._log("\n" + "=" * 60)
        self._log("Pipeline completed!")
        self._log(f"  Total:   {self._stats['total']}")
        self._log(f"  Accepted: {self._stats['accepted']}")
        self._log(f"  Skipped:  {self._stats['skipped']}")
        self._log(f"  Avg retries per accepted: "
                  f"{self._stats['attempts'] / max(self._stats['accepted'], 1):.2f}")
        self._log("=" * 60)

        # Memory 全局统计
        mem_stats = self.memory.get_all_stats()
        self._log(f"\nMemory stats:")
        self._log(f"  Total tracked: {mem_stats['total']}")
        self._log(f"  Persistent path: {self.memory.memory_path}")

    def _cleanup(self):
        if hasattr(self, "generator"):
            self.generator.unload()
        if hasattr(self, "critic") and self.critic is not None:
            self.critic.unload()
        torch.cuda.empty_cache()
        try:
            import cache_dit
            cache_dit.clear_cache()
        except ImportError:
            pass
