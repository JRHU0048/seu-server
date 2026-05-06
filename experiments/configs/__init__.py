"""
Baseline 配置定义

每个 baseline 是一个 dict，包含:
  - name: 唯一标识
  - use_critic: 是否启用 CriticAgent
  - max_retries: 最大重试次数
  - threshold: 接受阈值
  - description: 该 baseline 控制了什么变量

所有 baselines 共享相同的生成参数（steps, cfg, seed 等），
只在 use_critic / max_retries / threshold 三个维度上变化。
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class BaselineConfig:
    """单个 baseline 的配置"""
    name: str
    use_critic: bool
    max_retries: int
    threshold: float
    description: str


# ── Baseline 列表 ──────────────────────────────────────

ALL_BASELINES = [
    BaselineConfig(
        name="no_feedback",
        use_critic=False,
        max_retries=0,
        threshold=7.0,
        description="单次生成，无反馈无重试 — 当前方法的质量基线",
    ),
    BaselineConfig(
        name="multi_attempt",
        use_critic=False,
        max_retries=3,
        threshold=7.0,
        description="重复 3 次相同 prompt，无反馈 — 纯随机性的影响",
    ),
    BaselineConfig(
        name="critic_threshold_7",
        use_critic=True,
        max_retries=3,
        threshold=7.0,
        description="完整 pipeline，阈值 7.0 — 我们的默认方法",
    ),
    BaselineConfig(
        name="critic_threshold_6",
        use_critic=True,
        max_retries=3,
        threshold=6.0,
        description="低阈值 — 降低接受标准的效果",
    ),
    BaselineConfig(
        name="critic_threshold_8",
        use_critic=True,
        max_retries=3,
        threshold=8.0,
        description="高阈值 — 提高接受标准的效果",
    ),
]


def get_baseline_names() -> List[str]:
    return [b.name for b in ALL_BASELINES]


def get_baseline(name: str) -> BaselineConfig:
    for b in ALL_BASELINES:
        if b.name == name:
            return b
    raise KeyError(f"Unknown baseline: {name}. Available: {get_baseline_names()}")


def get_baseline_output_dir(task_output_root: str, baseline_name: str) -> str:
    """获取 baseline 的输出目录"""
    return os.path.join(task_output_root, "baselines", baseline_name)
