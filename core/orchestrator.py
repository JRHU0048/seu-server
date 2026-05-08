"""
OrchestratorAgent — Pipeline 编排器

职责:
  1. 管理所有 Agent 的生命周期
  2. 控制 Generate → Critic → Refine → Retry 循环
  3. 支持多种重试策略:
     - fixed:      固定次数重试（默认）
     - adaptive:   分数下降提前停止
     - aggressive: 每轮降低阈值 0.5
  4. 通过 Memory 模块实现断点续跑和经验积累

使用:
  orchestrator = OrchestratorAgent(config)
  orchestrator.run(task)
"""

import os
import time
from pathlib import Path

import torch

from .generator import GeneratorAgent
from .critic import CriticAgent, EvaluationResult
from .refiner import RefinerAgent
from .memory import MemoryModule
from .agent import BaseAgent

from tools import get_all_images, load_image


class OrchestratorAgent(BaseAgent):
    """Pipeline 编排器（无模型，纯控制逻辑）"""

    def __init__(self, config: dict):
        super().__init__(config)

        self.generator = GeneratorAgent(config)
        self.critic = None
        self.refiner = RefinerAgent(config)
        self.memory = MemoryModule(
            config.get("memory_path", "output/memory/memory.json")
        )

        self._log_file = config.get("log_file", "output/logs/pipeline.log")
        Path(self._log_file).parent.mkdir(parents=True, exist_ok=True)

        # 重试策略
        self._retry_strategy = config.get("retry_strategy", "fixed")
        self._save_attempts = config.get("save_attempts", False)

        # 统计
        self._stats = {
            "total": 0, "accepted": 0, "skipped": 0, "attempts_total": 0,
            "early_stops": 0,
        }

    def _load_model(self):
        self.generator.load()

    def _run_model(self, task):
        self._run_pipeline(task)

    # ── 主循环 ─────────────────────────────────────────

    def run_pipeline(self, task):
        cfg = self.config

        self._log("=" * 60)
        self._log(f"Pipeline started — Task: {task.name}")
        self._log(f"  Threshold: {cfg.get('threshold', 7.0)}, "
                  f"Max retries: {cfg.get('max_retries', 3)}")
        self._log(f"  Retry strategy: {self._retry_strategy}")
        self._log(f"  Save attempts: {self._save_attempts}")
        self._log(f"  Input:  {task.input_root}")
        self._log(f"  Output: {task.output_root}")
        self._log("=" * 60)

        try:
            self._log("[Orchestrator] Loading Generator...")
            self.generator.load()

            use_critic = cfg.get("use_critic", False)
            if use_critic:
                self._log("[Orchestrator] Loading Critic...")
                self.critic = CriticAgent(cfg)
                self.critic.load()

            class_folders = sorted(os.listdir(task.input_root))
            for cls in class_folders:
                self._process_class(task, cls)

        finally:
            self._cleanup()

        self._log_stats()

    # ── 类别处理 ───────────────────────────────────────

    def _process_class(self, task, class_name: str):
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

    # ── 单张处理 + 重试循环 ────────────────────────────

    def _process_one(self, task, class_name, output_dir, inputs):
        img_name = inputs["name"]
        suffix = task.output_suffix
        output_path = os.path.join(output_dir, f"{img_name}_{suffix}.png")

        self._stats["total"] += 1

        # Memory 检查
        if self.memory.is_done(class_name, img_name):
            self._log(f"  [{self._stats['total']}] {img_name} — already done, skip")
            self._stats["accepted"] += 1
            self._stats["attempts_total"] += len(
                self.memory.get_history(class_name, img_name)
            )
            return

        self._log(f"\n  [{self._stats['total']}] {class_name}/{img_name}")

        # 准备输入
        input_pils = [load_image(p, task.image_size) for p in inputs["paths"]]
        ref_pil = load_image(inputs["paths"][0], task.image_size)
        neg = task.neg_prompt

        max_retries = self.config.get("max_retries", 3)
        threshold = self.config.get("threshold", 7.0)

        feedback_history = []
        score_history = []
        accepted = False

        for attempt in range(1 + max_retries):
            # ── 计算实际阈值 ──
            effective_threshold = self._get_effective_threshold(
                threshold, attempt, score_history
            )

            # ── 构建 prompt ──
            prompt = task.build_prompt(feedback_history if attempt > 0 else None)

            self._log(f"    ── Attempt {attempt + 1} (threshold={effective_threshold:.1f}) ──")

            # Step 1: 生成
            gen_img = self.generator.run(
                input_pils, prompt, neg,
                seed=self.config.get("seed", 42) + attempt,
            )

            # Step 2: 保存中间结果（可选）
            if self._save_attempts:
                attempt_dir = os.path.join(output_dir, "_attempts", img_name)
                os.makedirs(attempt_dir, exist_ok=True)
                gen_img.save(os.path.join(attempt_dir, f"attempt_{attempt}.png"))

            # Step 3: 评估
            result = self._evaluate(
                gen_img, ref_pil, task, class_name, img_name, attempt
            )

            if result is not None:
                score = result.global_score
                score_history.append(score)
                self._log(f"    Score: {score:.1f}/10 | Issues: {result.issues[:2]}")
            else:
                score = effective_threshold + 1  # 无 Critic 直接通过

            # Step 4: 判断
            if score >= effective_threshold:
                gen_img.save(output_path)
                self._log(f"    \u2705 Accepted (score={score:.1f} >= {effective_threshold:.1f})")
                self._stats["accepted"] += 1
                self._stats["attempts_total"] += attempt
                accepted = True
                break

            # Step 5: 自适应提前停止
            if self._should_early_stop(score_history):
                self._log(f"    \u23f9 Early stop — score trend not improving")
                self._stats["early_stops"] += 1
                break

            # Step 6: 准备下次重试
            if attempt < max_retries:
                self._collect_feedback(result, feedback_history)
                self._log(f"    \u2b07 Retrying with {len(feedback_history)} feedback items")
                torch.cuda.empty_cache()
            else:
                self._log(f"    \u274c All attempts failed (final score={score:.1f})")

        if not accepted:
            self._stats["skipped"] += 1

        # 进度
        if self._stats["total"] % 20 == 0:
            self._log(f"\n  [Progress] total={self._stats['total']} "
                      f"accepted={self._stats['accepted']} "
                      f"skipped={self._stats['skipped']}")

    # ── 策略方法 ────────────────────────────────────────

    def _get_effective_threshold(self, base_threshold, attempt, score_history):
        """根据策略计算当前轮的有效阈值"""
        if self._retry_strategy == "aggressive":
            # 每轮降低 0.5
            return max(base_threshold - attempt * 0.5, 4.0)
        return base_threshold

    def _should_early_stop(self, score_history):
        """判断是否应该提前停止"""
        if self._retry_strategy != "adaptive":
            return False
        if len(score_history) < 3:
            return False
        # 连续两次分数下降或停滞
        if score_history[-1] <= score_history[-2] <= score_history[-3]:
            return True
        # 分数下降超过 0.5
        if score_history[-1] < score_history[-2] - 0.5:
            return True
        return False

    # ── 评估 ────────────────────────────────────────────

    def _evaluate(self, gen_img, ref_pil, task, class_name, img_name, attempt):
        """评估生成结果并记录到 Memory"""
        if self.critic is None:
            return None

        result = self.critic.run(
            gen_img, ref_pil,
            criteria=task.evaluation_criteria,
            category=task.category_name,
            task_description=task.task_description,
        )

        self.memory.record_attempt(
            class_name, img_name, attempt, result.global_score,
            result.issues[:3],
            result.global_score >= self.config.get("threshold", 7.0),
        )
        return result

    def _collect_feedback(self, result, feedback_history):
        """从评估结果中收集反馈"""
        if result is None:
            return
        for issue in result.issues:
            if issue and issue not in feedback_history:
                feedback_history.append(issue)
        for sugg in result.suggestions:
            if sugg and sugg not in feedback_history:
                feedback_history.append(sugg)

    # ── 辅助 ────────────────────────────────────────────

    def _log(self, msg: str):
        msg = str(msg)
        print(msg)
        with open(self._log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def _log_stats(self):
        self._log("\n" + "=" * 60)
        self._log("Pipeline completed!")
        self._log(f"  Total:          {self._stats['total']}")
        self._log(f"  Accepted:       {self._stats['accepted']}")
        self._log(f"  Skipped:        {self._stats['skipped']}")
        self._log(f"  Early stops:    {self._stats['early_stops']}")
        self._log(f"  Avg attempts:   "
                  f"{self._stats['attempts_total'] / max(self._stats['accepted'], 1):.2f}")
        self._log("=" * 60)

        mem_stats = self.memory.get_all_stats()
        self._log(f"\nMemory stats:")
        self._log(f"  Total tracked:  {mem_stats['total']}")
        self._log(f"  Accepted:       {mem_stats['accepted']}")
        self._log(f"  Skipped:        {mem_stats['skipped']}")
        self._log(f"  Path:           {self.memory.memory_path}")

    def _cleanup(self):
        if hasattr(self, "generator"):
            self.generator.unload()
        if hasattr(self, "critic") and self.critic is not None:
            self.critic.unload()
        torch.cuda.empty_cache()
        try:
            import cache_dit
            # cache_dit.clear_cache()
            if hasattr(cache_dit, "clear_cache"):
                cache_dit.clear_cache()
        except ImportError:
            pass


# ═══════════════════════════════════════════════════════════════
# SerialPipelineOrchestrator
# ═══════════════════════════════════════════════════════════════

class SerialPipelineOrchestrator(BaseAgent):
    """
    Agent级串行Pipeline — Generator和Critic分时占用显存，不同时加载。

    流程（多轮agent级迭代）:
      Round 1:
        Phase 1: GENERATE 全部图片（仅 Generator 在显存）
        Phase 2: EVALUATE 全部生成图（仅 Critic 在显存）
        Phase 3: 聚合跨图片反馈 → 更新生成 prompt
      Round 2..N:
        重复 Phase 1→Phase 2→Phase 3，携带上一轮聚合的全局反馈

    与 OrchestratorAgent 的关键区别:
    - Generator 和 Critic 永不共存于显存
    - 反馈以 agent 级别跨所有图片聚合，而非逐图即时反馈
    - 多轮迭代在数据集级别进行，同一 prompt 用于所有图片
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.generator = GeneratorAgent(config)
        self.critic = None
        self.refiner = RefinerAgent(config)
        self.memory = MemoryModule(
            config.get("memory_path", "output/memory/memory.json")
        )
        self._log_file = config.get("log_file", "output/logs/pipeline.log")
        Path(self._log_file).parent.mkdir(parents=True, exist_ok=True)
        self._num_rounds = config.get("num_rounds", 3)
        self._threshold = config.get("threshold", 7.0)

        self._stats = {
            "total": 0, "accepted": 0,
            "round_stats": [],
        }

    def _load_model(self):
        self.generator.load()

    def _run_model(self, task):
        self.run_pipeline(task)

    # ── 主入口 ─────────────────────────────────────────

    def run_pipeline(self, task):
        self._log("=" * 60)
        self._log("Serial Pipeline started" + f" — Task: {task.name}")
        self._log(f"  Num rounds:  {self._num_rounds}")
        self._log(f"  Threshold:   {self._threshold}")
        self._log(f"  Use critic:  {self.config.get('use_critic', False)}")
        self._log(f"  Generator:   {self.config.get('gen_model_id', '')}")
        self._log(f"  Critic:      {self.config.get('eval_model_id', '')}")
        self._log("=" * 60)

        use_critic = self.config.get("use_critic", False)

        # Step 1: 扫描所有待处理图片
        all_items = self._collect_items(task)
        self._stats["total"] = len(all_items)

        if not all_items:
            self._log("No items to process. Exiting.")
            return

        self._log(f"\nTotal images to process: {len(all_items)}")
        self._log(f"Rounds of agent-level iteration: {self._num_rounds}")

        # 跨轮次聚合反馈（agent-level, 非 per-image）
        global_feedback = []

        # 最佳结果追踪: {(cls, img_name): {"round": r, "score": s, "image": PIL}}
        best_per_image = {}

        # ── 多轮迭代 ──
        for round_idx in range(self._num_rounds):
            self._log(f"\n{'=' * 60}")
            self._log(f"  ROUND {round_idx + 1} / {self._num_rounds}")
            self._log(f"{'=' * 60}")

            # ===== Phase 1: 统一生成（仅 Generator 在显存） =====
            self._log(f"\n>>> [Phase 1] Generating all {len(all_items)} images ...")
            self.generator.load()

            round_generated = []  # [(cls, img_name, ref_pil, gen_img)]
            for idx, item in enumerate(all_items):
                cls, img_name, input_pils, ref_pil, neg = item
                prompt = task.build_prompt(global_feedback if round_idx > 0 else None)

                self._log(f"  Gen [{idx + 1}/{len(all_items)}] {cls}/{img_name}")
                gen_img = self.generator.run(
                    input_pils, prompt, neg,
                    seed=self.config.get("seed", 42) + round_idx * 10000 + idx,
                )
                round_generated.append((cls, img_name, ref_pil, gen_img))

            # 卸载 Generator → 释放显存
            self.generator.unload()
            self._log("  [Phase 1] Generator unloaded — VRAM released.")

            # ===== 无 Critic 模式 =====
            if not use_critic:
                self._log(">>> Critic disabled — saving all images directly.")
                for cls, img_name, _, gen_img in round_generated:
                    out_dir = os.path.join(task.output_root, cls)
                    os.makedirs(out_dir, exist_ok=True)
                    gen_img.save(os.path.join(
                        out_dir, f"{img_name}_{task.output_suffix}_r{round_idx}.png"
                    ))
                self._stats["accepted"] = len(all_items)
                break  # 一轮就够了

            # ===== Phase 2: 统一评估（仅 Critic 在显存） =====
            self._log(f">>> [Phase 2] Evaluating all {len(round_generated)} images ...")
            if self.critic is None:
                self.critic = CriticAgent(self.config)
            self.critic.load()

            round_evaluations = []  # [(cls, img_name, gen_img, EvaluationResult)]
            all_issues = []
            scores = []

            for idx, (cls, img_name, ref_pil, gen_img) in enumerate(round_generated):
                self._log(f"  Eval [{idx + 1}/{len(round_generated)}] {cls}/{img_name}")
                result = self.critic.run(
                    gen_img, ref_pil,
                    criteria=task.evaluation_criteria,
                    category=task.category_name,
                    task_description=task.task_description,
                )
                round_evaluations.append((cls, img_name, gen_img, result))
                all_issues.extend(result.issues)
                scores.append(result.global_score)

                # 更新全局最佳
                key = (cls, img_name)
                if (key not in best_per_image or
                        result.global_score > best_per_image[key]["score"]):
                    best_per_image[key] = {
                        "round": round_idx,
                        "score": result.global_score,
                        "image": gen_img,
                    }

            self.critic.unload()
            self._log("  [Phase 2] Critic unloaded — VRAM released.")

            # ===== 保存本轮中间结果 & 记录 Memory =====
            accepted_count = 0
            for cls, img_name, gen_img, result in round_evaluations:
                rounds_dir = os.path.join(task.output_root, cls, "_rounds")
                os.makedirs(rounds_dir, exist_ok=True)
                gen_img.save(os.path.join(rounds_dir, f"{img_name}_r{round_idx}.png"))

                is_accepted = result.global_score >= self._threshold
                if is_accepted:
                    accepted_count += 1
                self.memory.record_attempt(
                    cls, img_name, round_idx, result.global_score,
                    result.issues[:3], is_accepted,
                )

            # ===== 本轮统计 =====
            avg_score = sum(scores) / len(scores) if scores else 0.0
            self._log(f"\n--- Round {round_idx + 1} Summary ---")
            self._log(f"  Avg score:    {avg_score:.2f} / 10")
            self._log(f"  Accepted:     {accepted_count} / {len(scores)}  (threshold={self._threshold})")
            if all_issues:
                from collections import Counter
                top3 = Counter(all_issues).most_common(3)
                self._log(f"  Top issues:   {[t[0][:60] for t in top3]}")

            self._stats["round_stats"].append({
                "round": round_idx,
                "avg_score": round(avg_score, 2),
                "accepted": accepted_count,
                "total": len(scores),
            })

            # ===== Phase 3: 聚合跨图反馈（agent-level） =====
            if round_idx < self._num_rounds - 1:
                self._log(">>> [Phase 3] Aggregating cross-image feedback ...")
                global_feedback = self._aggregate_feedback(round_evaluations, self._threshold)
                self._log(f"  → {len(global_feedback)} feedback items for round {round_idx + 2}")
                for fb in global_feedback[:3]:
                    self._log(f"    - {fb[:120]}")

        # ── 保存最终最佳结果 ──
        self._log(f"\n>>> Saving best results across {len(best_per_image)} images ...")
        saved = 0
        for (cls, img_name), best in best_per_image.items():
            out_dir = os.path.join(task.output_root, cls)
            os.makedirs(out_dir, exist_ok=True)
            best["image"].save(os.path.join(
                out_dir, f"{img_name}_{task.output_suffix}_best.png"
            ))
            saved += 1

        self._stats["accepted"] = saved
        self._log_stats()

    # ── 数据收集 ────────────────────────────────────────

    def _collect_items(self, task):
        """扫描数据集，收集所有待处理的 (cls, img_name, input_pils, ref_pil, neg)"""
        items = []
        class_folders = sorted(os.listdir(task.input_root))
        for cls in class_folders:
            if getattr(task, "skip_class", lambda _: False)(cls):
                self._log(f"  Skip class: {cls}")
                continue

            input_dir = os.path.join(task.input_root, cls)
            output_dir = os.path.join(task.output_root, cls)
            os.makedirs(output_dir, exist_ok=True)

            images = get_all_images(input_dir)
            if len(images) < task.min_input_images:
                self._log(f"  Skip {cls} ({len(images)} < {task.min_input_images} images)")
                continue

            for inputs in task.iter_inputs(images):
                img_name = inputs["name"]
                if self.memory.is_done(cls, img_name):
                    self._log(f"  {cls}/{img_name} — already done, skip")
                    continue

                input_pils = [load_image(p, task.image_size) for p in inputs["paths"]]
                ref_pil = load_image(inputs["paths"][0], task.image_size)
                items.append((cls, img_name, input_pils, ref_pil, task.neg_prompt))

        return items

    # ── 反馈聚合 ────────────────────────────────────────

    def _aggregate_feedback(self, round_evaluations, threshold):
        """
        跨所有图片聚合 agent-level 反馈。
        策略:
          1. 提取所有 issues 中的高频模式
          2. 从低分图片中收集具体的改进建议
        """
        from collections import Counter

        all_issues = []
        low_score_results = []

        for cls, img_name, gen_img, result in round_evaluations:
            all_issues.extend(result.issues)
            if result.global_score < threshold:
                low_score_results.append(result)

        if not low_score_results:
            return []  # 全部达标 → 无需反馈

        # 高频失败模式
        issue_counts = Counter(all_issues)
        top_issues = [iss for iss, _ in issue_counts.most_common(5) if iss]

        feedback = []
        if top_issues:
            feedback.append(
                "Common problems from previous round across ALL images: "
                + "; ".join(top_issues)
            )

        # 低分图片的具体反馈（去重）
        seen = set()
        for r in low_score_results:
            for iss in r.issues:
                if iss and iss not in seen:
                    feedback.append(f"- {iss}")
                    seen.add(iss)
                    if len(feedback) >= 8:
                        break
            if len(feedback) >= 8:
                break

        return feedback[:8]

    # ── 工具 ────────────────────────────────────────────

    def _log(self, msg: str):
        msg = str(msg)
        print(msg)
        with open(self._log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def _log_stats(self):
        self._log("\n" + "=" * 60)
        self._log("Serial Pipeline completed!")
        self._log(f"  Total items:  {self._stats['total']}")
        self._log(f"  Best saved:   {self._stats['accepted']}")
        rounds = self._stats.get("round_stats", [])
        if rounds:
            self._log("  Rounds:")
            for s in rounds:
                self._log(f"    Round {s['round'] + 1}: avg={s['avg_score']:.2f}, "
                          f"accepted={s['accepted']}/{s['total']}")
        self._log("=" * 60)

    def _cleanup(self):
        if hasattr(self, "generator"):
            self.generator.unload()
        if hasattr(self, "critic") and self.critic is not None:
            self.critic.unload()
        torch.cuda.empty_cache()
        try:
            import cache_dit
            if hasattr(cache_dit, "clear_cache"):
                cache_dit.clear_cache()
        except ImportError:
            pass
