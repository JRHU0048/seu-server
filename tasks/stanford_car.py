"""
Stanford Cars 视角生成任务（预留）

数据集:
  Stanford Cars — 196 类汽车，共 16,185 张图片

TODO:
  1. 上传数据后填写 input_root / output_root
  2. 调整 prompt 模板（汽车描述词）
  3. 调整 evaluation_criteria（关注车灯/轮毂等）
"""

from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path

from .base import TaskConfig
from . import register


@register("stanford_car")
@dataclass
class StanfordCarConfig(TaskConfig):
    name: str = "stanford_car"
    category_name: str = "car"
    task_description: str = "Rotate the car 60 degrees and replace the background."

    # FIXME: 上传数据后修改
    input_root: str = "/path/to/stanford_cars/train"
    output_root: str = "/path/to/stanford_cars/output"

    num_classes: int = 196
    min_input_images: int = 3
    output_suffix: str = "view_60"

    evaluation_criteria: list = field(default_factory=lambda: [
        {"name": "identity",
         "description": "Does the car model, color, shape, and details match the reference?",
         "weight": 0.35},
        {"name": "viewpoint",
         "description": "Is the rotation realistic with proper 3D perspective?",
         "weight": 0.25},
        {"name": "quality",
         "description": "Any artifacts, distortion, or unrealistic elements?",
         "weight": 0.25},
        {"name": "background",
         "description": "Is the background a realistic street/nature scene?",
         "weight": 0.15},
    ])

    gen_prompt_template: str = (
        "You are given THREE reference images of a car.\n"
        "The FIRST image is the MAIN reference.\n"
        "Task:\n"
        "- Rotate the camera horizontally about 60 degrees around the car.\n"
        "- Replace the background with a realistic environment.\n"
        "Important: Generate ONLY ONE car. Maintain exact model identity.\n"
        "{feedback_section}"
    )

    def iter_inputs(self, image_paths: List[str]):
        total = len(image_paths)
        for i in range(total):
            yield {
                "name": Path(image_paths[i]).stem,
                "paths": [
                    image_paths[i],
                    image_paths[(i + 1) % total],
                    image_paths[(i + 2) % total],
                ],
            }

    def build_prompt(self, feedback_history: Optional[List[str]] = None) -> str:
        feedback_section = ""
        if feedback_history:
            items = "\n".join(f"- {fb}" for fb in feedback_history if fb)
            feedback_section = "\n\nPrevious issues to fix:\n" + items
        return self.gen_prompt_template.format(feedback_section=feedback_section)
