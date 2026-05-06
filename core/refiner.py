"""
RefinerAgent — Prompt 改写 Agent

职责:
  将 Critic 的结构化反馈翻译成 Generator 能理解的 prompt 改进,
  精准修改 prompt 的对应段落，而非简单 append。

使用:
  agent = RefinerAgent(config)
  improved = agent.run(original_prompt, eval_result, feedback_history)
"""

from .agent import BaseAgent


# ── 反馈注入模板 ─────────────────────────────────────

def _build_section_fix(evaluation_result) -> str:
    """根据评估结果生成针对性的改进指令"""
    fixes = []

    if evaluation_result.identity_score is not None and evaluation_result.identity_score < 7:
        fixes.append(
            "- IDENTITY: The previous output did not perfectly match the reference's appearance. "
            "Pay extremely close attention to colors, patterns, and morphology of the reference. "
            "Do NOT blend features from auxiliary images."
        )

    if evaluation_result.viewpoint_score is not None and evaluation_result.viewpoint_score < 7:
        fixes.append(
            "- VIEWPOINT: The previous rotation appeared flat or unnatural. "
            "Ensure the camera rotates around the subject in 3D space, "
            "preserving natural perspective and proportions."
        )

    if evaluation_result.quality_score is not None and evaluation_result.quality_score < 7:
        fixes.append(
            "- QUALITY: The previous output had artifacts or deformities. "
            "Avoid extra limbs, distorted anatomy, blurring, or pixelation."
        )

    if evaluation_result.background_score is not None and evaluation_result.background_score < 7:
        fixes.append(
            "- BACKGROUND: The previous background was not realistic. "
            "Generate a natural outdoor scene with proper lighting and shadows "
            "that match the subject's environment."
        )

    # 额外 issues
    for issue in evaluation_result.issues:
        fixes.append(f"- FIX: {issue}")

    return "\n".join(fixes)


class RefinerAgent(BaseAgent):
    """
    Prompt 改写 Agent（无模型，纯逻辑改写 + 可选 LLM 改写）
    当前实现: 基于规则的段落替换
    未来可升级: 用 LLM 做 prompt 重写
    """

    def __init__(self, config: dict, use_llm: bool = False):
        super().__init__(config)
        self.use_llm = use_llm

    def _load_model(self):
        if self.use_llm:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model_id = self.config.get("refiner_model_id", "")
            if model_id:
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_id, torch_dtype="auto", device_map="auto"
                )
                self._tokenizer = AutoTokenizer.from_pretrained(model_id)

    def _run_model(self, original_prompt: str, evaluation_result=None,
                   feedback_history: list[str] = None) -> str:
        """
        改进生成 prompt

        参数:
          original_prompt: str — 原始生成 prompt
          evaluation_result: EvaluationResult | None — 评估结果
          feedback_history: list[str] — 历史反馈文本

        返回:
          str — 改进后的 prompt
        """
        parts = [original_prompt]

        if self.use_llm and self._model is not None:
            # LLM 改写模式（预留接口）
            return self._llm_rewrite(original_prompt, evaluation_result, feedback_history)

        # 规则改写模式
        if evaluation_result is not None:
            section_fixes = _build_section_fix(evaluation_result)
            if section_fixes:
                parts.append("\n\n[IMPROVEMENTS NEEDED]\n" + section_fixes)

        if feedback_history:
            items = "\n".join(f"- {fb}" for fb in feedback_history if fb)
            parts.append(f"\n\n[PREVIOUS FEEDBACK]\n{items}")

        return "\n\n".join(parts)

    def _llm_rewrite(self, original_prompt, evaluation_result, feedback_history) -> str:
        """LLM 改写模式（预留，暂未实现）"""
        # TODO: 使用 LLM 重写整个 prompt，而非简单拼接
        return original_prompt
