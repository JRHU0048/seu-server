"""
NABirds 视角生成任务（预留）

数据集:
  NABirds (North American Birds) — 555 类，约 48,000 张图片
  与 CUB 同类任务（细粒度鸟类分类），可复用大部分 prompt

TODO:
  1. 上传数据后填写 input_root / output_root
  2. 确认类别数量
"""

from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path

from .base import TaskConfig
from . import register


@register("nabird")
@dataclass
class NABirdConfig(TaskConfig):
    name: str = "nabird"
    category_name: str = "bird"
    task_description: str = "Rotate the bird 60 degrees and replace the background with a natural outdoor scene."

    # FIXME: 上传数据后修改
    input_root: str = "/path/to/nabirds/train"
    output_root: str = "/path/to/nabirds/output"

    num_classes: int = 555
    min_input_images: int = 3
    output_suffix: str = "view_bg_edit_v1"

    evaluation_criteria: list = field(default_factory=lambda: [
        {"name": "identity",
         "description": "Does the bird's species, feather colors, beak shape match the reference?",
         "weight": 0.35},
        {"name": "viewpoint",
         "description": "Is the 60-degree rotation realistic?",
         "weight": 0.25},
        {"name": "quality",
         "description": "Any artifacts, blurring, deformities?",
         "weight": 0.25},
        {"name": "background",
         "description": "Is the background a realistic natural environment?",
         "weight": 0.15},
    ])

    gen_prompt_template: str = (
        "You are given THREE reference images of a bird.\n"
        "The FIRST image is the MAIN reference and defines the bird's identity.\n"
        "Task:\n"
        "- Rotate the camera horizontally about 60 degrees.\n"
        "- Replace the background with a realistic natural environment.\n"
        "Generate ONLY ONE bird. Do not deform or distort.\n"
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
