"""
CriticAgent — 结构化评估 Agent

职责:
  用 Qwen VLM 对生成结果进行多维度量化评估,
  输出结构化评分卡（identity / viewpoint / quality / background 等维度）。

使用:
  agent = CriticAgent(task.config)
  agent.load()
  result = agent.run(gen_image, ref_image, criteria)
  print(result.global_score, result.issues)
  agent.unload()
"""

import re
import json
from dataclasses import dataclass, field
from typing import Optional

import torch

from .agent import BaseAgent


@dataclass
class EvaluationResult:
    """结构化评估结果"""
    identity_score: float = 0.0
    viewpoint_score: Optional[float] = None
    quality_score: float = 0.0
    background_score: Optional[float] = None
    global_score: float = 0.0
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    fix_priority: list[str] = field(default_factory=list)
    raw_output: str = ""


# ── 评估 Prompt 模板 ──────────────────────────────────

EVAL_PROMPT_TEMPLATE = """\
You are an expert evaluator of AI-generated {category} images.

You will see TWO images:
1. REFERENCE image — the original {category} photo.
2. GENERATED image — the AI-generated new viewpoint of the same {category}.

The generation task was: "{task_description}"

Evaluate the GENERATED image on these criteria (each score 0-10):

{criteria_text}

IMPORTANT: Respond ONLY with a valid JSON object, no other text:
{{
    "identity_score": <0-10>,
    "viewpoint_score": <0-10>,
    "quality_score": <0-10>,
    "background_score": <0-10>,
    "global_score": <average of above>,
    "issues": ["issue1", "issue2"],
    "suggestions": ["suggestion1", "suggestion2"],
    "fix_priority": ["identity", "background"]
}}
"""


class CriticAgent(BaseAgent):
    """结构化评估 Agent"""

    def __init__(self, config: dict, model_id: str = None):
        super().__init__(config)
        self._model_id = model_id or config.get("eval_model_id", "")
        self._processor = None

    def _load_model(self):
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            self._model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self._processor = AutoProcessor.from_pretrained(
            self._model_id, trust_remote_code=True
        )

    def _run_model(self, gen_image, ref_image, criteria: list[dict] = None,
                   category: str = "bird", task_description: str = "") -> EvaluationResult:
        """
        评估生成图像

        参数:
          gen_image: PIL.Image — 生成的图像
          ref_image: PIL.Image — 参考图像
          criteria: list[dict] — 评估标准列表
          category: str — 类别名词（bird / car / dog）
          task_description: str — 任务描述

        返回:
          EvaluationResult — 结构化评估结果
        """
        # 构建 criteria 文本
        if criteria is None:
            criteria = [
                {"name": "identity", "description": "Does the subject's identity match the reference?", "weight": 1},
                {"name": "viewpoint", "description": "Is the rotation/viewpoint realistic?", "weight": 1},
                {"name": "quality", "description": "Any artifacts, blurring, deformities?", "weight": 1},
                {"name": "background", "description": "Is the background realistic?", "weight": 1},
            ]

        criteria_text = "\n".join(
            f"{i+1}. {c['name']}: {c['description']}" for i, c in enumerate(criteria)
        )

        prompt = EVAL_PROMPT_TEMPLATE.format(
            category=category,
            task_description=task_description,
            criteria_text=criteria_text,
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": ref_image},
                    {"type": "image", "image": gen_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text],
            images=[ref_image, gen_image],
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs, max_new_tokens=512, do_sample=False,
            )

        raw = self._processor.decode(
            output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()

        return self._parse(raw)

    def _parse(self, raw_text: str) -> EvaluationResult:
        """从 VLM 输出中解析结构化结果"""
        result = EvaluationResult(raw_output=raw_text)

        # 尝试从 JSON 代码块中提取
        json_match = re.search(r"```(?:json)?\s*\n?({.*?})\s*\n?```", raw_text, re.DOTALL)
        if json_match:
            raw_text = json_match.group(1)

        # 尝试直接解析 JSON
        try:
            data = json.loads(raw_text)
            result.identity_score = float(data.get("identity_score", 0))
            result.viewpoint_score = float(data.get("viewpoint_score", 0))
            result.quality_score = float(data.get("quality_score", 0))
            result.background_score = float(data.get("background_score", 0))
            result.global_score = float(data.get("global_score", 0))
            result.issues = data.get("issues", [])
            result.suggestions = data.get("suggestions", [])
            result.fix_priority = data.get("fix_priority", [])
            return result
        except (json.JSONDecodeError, TypeError):
            pass

        # 备选: 从文本中正则提取各维度的分数
        for dim in ["identity", "viewpoint", "quality", "background", "global"]:
            m = re.search(rf'"{dim}_score"\s*:\s*([\d.]+)', raw_text)
            if m:
                setattr(result, f"{dim}_score", float(m.group(1)))

        # 提取 issues
        issues_match = re.search(r'"issues"\s*:\s*\[(.*?)\]', raw_text, re.DOTALL)
        if issues_match:
            result.issues = re.findall(r'"([^"]+)"', issues_match.group(1))

        # 提取 suggestions
        sugg_match = re.search(r'"suggestions"\s*:\s*\[(.*?)\]', raw_text, re.DOTALL)
        if sugg_match:
            result.suggestions = re.findall(r'"([^"]+)"', sugg_match.group(1))

        return result
