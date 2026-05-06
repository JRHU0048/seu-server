"""
TaskConfig — 任务配置基类

职责:
  定义数据集无关的接口，所有具体任务（CUB / Cars / Dogs / NABirds）继承此类。

使用:
  from tasks.base import TaskConfig

  @register("my_task")
  class MyTask(TaskConfig):
      ...
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

from tools import get_all_images


@dataclass
class TaskConfig:
    """任务配置基类"""

    # ── 标识 ──
    name: str = "base"
    category_name: str = "object"
    task_description: str = ""

    # ── 路径 ──
    input_root: str = ""
    output_root: str = ""

    # ── 数据 ──
    num_classes: int = 0
    min_input_images: int = 1
    skip_upto: int = 0

    # ── 生成参数 ──
    image_size: int = 384
    neg_prompt: str = (
        "collage, multiple images, split image, duplicated subject, "
        "blurry, low quality, deformed, malformed, bad anatomy, "
        "extra parts, fused parts"
    )

    # ── 输出 ──
    output_suffix: str = "generated"

    # ── 评估标准 ──
    evaluation_criteria: list = field(default_factory=lambda: [
        {"name": "identity", "description": "Does the subject's identity match the reference?", "weight": 0.35},
        {"name": "quality", "description": "Any artifacts, blurring, deformities?", "weight": 0.35},
        {"name": "background", "description": "Is the background realistic and natural?", "weight": 0.30},
    ])

    # ── 数据迭代 ────────────────────────────────────

    def get_classes(self) -> List[str]:
        return sorted(os.listdir(self.input_root))

    def skip_class(self, class_name: str) -> bool:
        if self.skip_upto <= 0:
            return False
        from tools import is_skipped_class
        return is_skipped_class(class_name, max_class=self.skip_upto)

    def iter_inputs(self, image_paths: List[str]):
        raise NotImplementedError

    # ── Prompt 构建 ────────────────────────────────

    def build_prompt(self, feedback_history: Optional[List[str]] = None) -> str:
        raise NotImplementedError
