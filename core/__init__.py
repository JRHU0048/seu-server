"""
core 包 — Multi-Agent 框架核心
"""

from .agent import BaseAgent
from .generator import GeneratorAgent
from .critic import CriticAgent, EvaluationResult
from .refiner import RefinerAgent
from .memory import MemoryModule
from .orchestrator import OrchestratorAgent, SerialPipelineOrchestrator

__all__ = [
    "BaseAgent",
    "GeneratorAgent",
    "CriticAgent",
    "EvaluationResult",
    "RefinerAgent",
    "MemoryModule",
    "OrchestratorAgent",
    "SerialPipelineOrchestrator",
]
