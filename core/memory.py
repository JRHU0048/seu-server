"""
MemoryModule — 经验记忆模块

职责:
  1. 记录每张图片的生成历史（尝试次数、分数、错误类型）
  2. 支持断点续跑（已成功的不重复生成）
  3. 同类经验迁移（同一类别的失败模式可用于同类其他实例）
  4. 跨 session 持久化（JSON 文件）

使用:
  memory = MemoryModule("output/memory/cub_bird.json")
  memory.record_attempt(class_name, img_name, attempt, score, issues)
  if memory.is_done(class_name, img_name): continue
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Optional


class MemoryModule:
    """经验记忆模块"""

    def __init__(self, memory_path: str = "output/memory/memory.json"):
        self.memory_path = Path(memory_path)
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict = self._load()

    # ── 持久化 ────────────────────────────────────────────

    def _load(self) -> dict:
        if self.memory_path.exists():
            with open(self.memory_path, "r") as f:
                return json.load(f)
        return {"classes": {}}

    def _save(self):
        with open(self.memory_path, "w") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    # ── 记录 ──────────────────────────────────────────────

    def record_attempt(
        self,
        class_name: str,
        img_name: str,
        attempt: int,
        score: float,
        issues: list[str],
        accepted: bool,
    ):
        """记录一次生成尝试"""
        cls_data = self._data["classes"].setdefault(class_name, {})
        img_data = cls_data.setdefault(img_name, {"attempts": [], "accepted": False})

        img_data["attempts"].append({
            "attempt": attempt,
            "score": score,
            "issues": issues,
        })
        if accepted:
            img_data["accepted"] = True
            img_data["final_score"] = score

        self._save()

    # ── 查询 ──────────────────────────────────────────────

    def is_done(self, class_name: str, img_name: str) -> bool:
        """检查某张图是否已成功生成"""
        img_data = self._data["classes"].get(class_name, {}).get(img_name)
        return img_data is not None and img_data.get("accepted", False)

    def get_history(self, class_name: str, img_name: str) -> list:
        """获取某张图的生成历史"""
        return self._data["classes"].get(class_name, {}).get(img_name, {}).get("attempts", [])

    def get_attempt_count(self, class_name: str, img_name: str) -> int:
        """获取某张图已尝试次数"""
        return len(self.get_history(class_name, img_name))

    def get_class_stats(self, class_name: str) -> dict:
        """获取某个类别的统计信息"""
        cls_data = self._data["classes"].get(class_name, {})
        if not cls_data:
            return {"total": 0, "accepted": 0, "skipped": 0}

        total = len(cls_data)
        accepted = sum(1 for v in cls_data.values() if v.get("accepted"))
        return {
            "total": total,
            "accepted": accepted,
            "skipped": total - accepted,
        }

    def get_all_stats(self) -> dict:
        """获取全局统计"""
        total = accepted = skipped = 0
        for cls_name, cls_data in self._data["classes"].items():
            for img_name, img_data in cls_data.items():
                total += 1
                if img_data.get("accepted"):
                    accepted += 1
                else:
                    skipped += 1
        return {"total": total, "accepted": accepted, "skipped": skipped}

    def get_common_issues(self, class_name: str, top_k: int = 5) -> list[tuple[str, int]]:
        """获取某个类别最常见的失败原因"""
        issue_counter = defaultdict(int)
        for img_data in self._data["classes"].get(class_name, {}).values():
            for attempt in img_data.get("attempts", []):
                for issue in attempt.get("issues", []):
                    issue_counter[issue] += 1
        return sorted(issue_counter.items(), key=lambda x: -x[1])[:top_k]
