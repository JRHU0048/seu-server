"""
Stanford Dogs 视角生成任务（预留）

数据集:
  Stanford Dogs — 120 类犬种，共 20,580 张图片

TODO:
  1. 上传数据后填写 input_root / output_root
  2. 调整 prompt 模板
"""

from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path

from .base import TaskConfig
from . import register


@register("stanford_dog")
@dataclass
class StanfordDogConfig(TaskConfig):
    name: str = "stanford_dog"
    category_name: str = "dog"
    task_description: str = "Rotate the dog 60 degrees and replace the background."

    # FIXME: 上传数据后修改
    input_root: str = "/path/to/stanford_dogs/train"
    output_root: str = "/path/to/stanford_dogs/output"

    num_classes: int = 120
    min_input_images: int = 3
    output_suffix: str = "view_60"

    evaluation_criteria: list = field(default_factory=lambda: [
        {"name": "identity",
         "description": "Does the dog breed, fur pattern, face shape match the reference?",
         "weight": 0.35},
        {"name": "viewpoint",
         "description": "Is the rotation realistic?",
         "weight": 0.25},
        {"name": "quality",
         "description": "Any artifacts, deformities, or anatomical issues?",
         "weight": 0.25},
        {"name": "background",
         "description": "Is the background realistic?",
         "weight": 0.15},
    ])

    gen_prompt_template: str = (
        "You are given THREE reference images of a dog.\n"
        "The FIRST image is the MAIN reference.\n"
        "Task:\n"
        "- Rotate the camera horizontally about 60 degrees around the dog.\n"
        "- Replace the background with a realistic natural scene.\n"
        "Important: Generate ONLY ONE dog. Maintain exact breed identity.\n"
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
