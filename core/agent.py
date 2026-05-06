"""
BaseAgent — 所有 Agent 的抽象基类

职责:
  定义统一的加载/运行/卸载生命周期,
  支持 with 语句自动管理显存。
"""

from abc import ABC, abstractmethod
import torch


class BaseAgent(ABC):
    """Agent 基类，所有 Agent 继承此接口"""

    def __init__(self, config: dict):
        self.config = config
        self._model = None
        self._loaded = False

    # ── 生命周期 ──────────────────────────────────────────

    @abstractmethod
    def _load_model(self):
        """子类实现: 加载模型到 GPU"""
        ...

    @abstractmethod
    def _run_model(self, *args, **kwargs):
        """子类实现: 用加载好的模型执行推理"""
        ...

    def load(self):
        """加载模型（幂等）"""
        if not self._loaded:
            self._load_model()
            self._loaded = True
        return self

    def run(self, *args, **kwargs):
        """执行推理（自动加载）"""
        if not self._loaded:
            self.load()
        return self._run_model(*args, **kwargs)

    def unload(self):
        """卸载模型，释放显存"""
        if self._model is not None:
            del self._model
            self._model = None
        self._loaded = False
        torch.cuda.empty_cache()

    # ── 上下文管理 ────────────────────────────────────────

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()
