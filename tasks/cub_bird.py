"""
CUB-200-2011 鸟类视角生成任务

数据集:
  Caltech-UCSD Birds-200-2011 (CUB-200-2011)
  200 类鸟类，约 6000 张训练图片

输入策略:
  滑动窗口 3 图（第一张为主图，后两张为辅助参考）

生成目标:
  60° 水平视角旋转 + 自然背景替换
"""

from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path

from .base import TaskConfig
from . import register


@register("cub_bird")
@dataclass
class CUBBirdConfig(TaskConfig):
    """CUB-200-2011 鸟类视角生成"""

    name: str = "cub_bird"
    category_name: str = "bird"
    task_description: str = "Rotate the bird 60 degrees and replace the background with a natural outdoor scene."

    # ── 路径（默认值，可在构造时覆盖） ──
    input_root: str = "/home/huyanhan/data/CUB_200_2011/images/train"
    output_root: str = "/home/huyanhan/data/CUB_200_2011/qwen_output/train"

    num_classes: int = 200
    min_input_images: int = 3
    output_suffix: str = "view_bg_edit_v1"

    # ── 评估标准 ──
    evaluation_criteria: list = field(default_factory=lambda: [
        {"name": "identity",
         "description": "Does the bird's species, feather colors, beak shape, "
                        "and body structure match the reference?",
         "weight": 0.35},
        {"name": "viewpoint",
         "description": "Is the 60-degree horizontal rotation realistic "
                        "and does it look like a genuine 3D rotation?",
         "weight": 0.25},
        {"name": "quality",
         "description": "Any artifacts, blurring, deformities, malformed anatomy?",
         "weight": 0.25},
        {"name": "background",
         "description": "Is the new background a realistic natural environment "
                        "with consistent lighting?",
         "weight": 0.15},
    ])

    # ── Prompt 模板 ──
    gen_prompt_template: str = (
        "You are given THREE reference images of a bird.\n"
        "The FIRST image is the MAIN reference and defines the bird's identity: "
        "species, exact feather colors, beak shape, texture, pattern, body proportions, "
        "and all visual characteristics. The other two images are AUXILIARY ONLY.\n"
        "STRICT RULE: The resulting bird MUST EXACTLY match the FIRST image's appearance. "
        "Do NOT use any identity attribute from the auxiliary images; "
        "if they differ, ignore them entirely.\n"
        "Task:\n"
        "- Rotate the camera horizontally about 60 degrees around the bird.\n"
        "- Replace the background with a realistic natural outdoor environment.\n"
        "- Adjust pose, lighting, shadow, and perspective so the result "
        "looks like a natural wildlife photograph.\n"
        "Important: Generate ONLY ONE bird. Do NOT deform or distort. "
        "Do NOT generate collage or multiple birds.\n"
        "{feedback_section}"
    )

    # ── 输入迭代：滑动窗口 3 图 ──

    def iter_inputs(self, image_paths: List[str]):
        """滑动窗口三图输入"""
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

    # ── Prompt 构建 ──

    def build_prompt(self, feedback_history: Optional[List[str]] = None) -> str:
        feedback_section = ""
        if feedback_history:
            items = "\n".join(f"- {fb}" for fb in feedback_history if fb)
            feedback_section = (
                "\n\nNote — Previous generation had issues. "
                "Please fix them this time:\n" + items
            )
        return self.gen_prompt_template.format(feedback_section=feedback_section)
