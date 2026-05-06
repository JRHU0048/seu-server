"""
tasks 包 — 任务配置注册

通过 register 装饰器注册新任务，供 experiments/run_pipeline.py 加载。

使用:
  from tasks import get_task
  task = get_task("cub_bird")
"""

from typing import Dict, Type
from .base import TaskConfig

_registry: Dict[str, Type[TaskConfig]] = {}


def register(name: str):
    """装饰器：注册任务"""
    def wrapper(cls):
        _registry[name] = cls
        return cls
    return wrapper


def get_task(name: str, **kwargs) -> TaskConfig:
    """根据名称获取任务配置实例"""
    if name not in _registry:
        available = ", ".join(_registry.keys())
        raise KeyError(f"Unknown task '{name}'. Available: {available}")
    return _registry[name](**kwargs)


def list_tasks() -> list[str]:
    return list(_registry.keys())


# ── 注册具体任务（导入即自动触发装饰器） ──
from . import cub_bird
from . import stanford_car
from . import stanford_dog
from . import nabird
