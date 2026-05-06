"""
GeneratorAgent — 生成 Agent

职责:
  封装 Qwen-Image-Edit-2511 的加载、推理、显存管理,
  支持 cache-dit 加速推理。

使用:
  agent = GeneratorAgent(task.config)
  agent.load()
  img = agent.run(input_imgs, prompt, neg_prompt)
  agent.unload()
"""

import torch
from diffusers import QwenImageEditPlusPipeline
import cache_dit

from .agent import BaseAgent


class GeneratorAgent(BaseAgent):
    """图像生成 Agent（Qwen-Image-Edit-2511）"""

    def _load_model(self):
        cfg = self.config
        self._model = QwenImageEditPlusPipeline.from_pretrained(
            cfg["gen_model_id"],
            torch_dtype=getattr(torch, cfg.get("dtype", "bfloat16")),
            device_map=cfg.get("device_map", "balanced"),
            max_memory=cfg.get("max_memory", {0: "46GiB", 1: "46GiB"}),
        )

        # cache-dit 加速
        cache_dit.enable_cache(self._model, **cfg.get("cache_options", {
            "cache_type": cache_dit.DBCache,
            "warmup_steps": 8,
            "max_cached_steps": -1,
            "Fn_compute_blocks": 8,
            "Bn_compute_blocks": 0,
            "residual_diff_threshold": 0.12,
            "do_separate_classifier_free_guidance": True,
            "cfg_compute_first": False,
            "enable_taylorseer": True,
            "enable_encoder_taylorseer": True,
            "taylorseer_cache_type": "residual",
            "taylorseer_kwargs": {"n_derivatives": 4},
        }))

        self._model.set_progress_bar_config(disable=None)

    def _run_model(self, input_imgs, prompt, neg_prompt=None, seed=42):
        """
        生成一张图像

        参数:
          input_imgs: list[PIL.Image] — 输入参考图
          prompt: str — 生成 prompt
          neg_prompt: str | None — 负面 prompt
          seed: int — 随机种子

        返回:
          PIL.Image — 生成结果
        """
        cfg = self.config

        if neg_prompt is None:
            neg_prompt = "collage, multiple birds, split image, duplicated bird, blurry, low quality, deformed"

        with torch.inference_mode():
            result = self._model(
                image=input_imgs,
                prompt=prompt,
                negative_prompt=neg_prompt,
                true_cfg_scale=cfg.get("cfg", 3.5),
                num_inference_steps=cfg.get("steps", 28),
                guidance_scale=1.0,
                generator=torch.manual_seed(seed),
            )

        torch.cuda.empty_cache()
        return result.images[0]

    def run_batch(self, input_list, prompt, neg_prompt=None, seeds=None):
        """
        批量生成（多次调用 run，但模型已加载在显存中）

        参数:
          input_list: list[list[PIL.Image]]
          prompt: str
          neg_prompt: str | None
          seeds: list[int] | None

        返回:
          list[PIL.Image]
        """
        if seeds is None:
            seeds = [42] * len(input_list)
        return [
            self._run_model(imgs, prompt, neg_prompt, seed)
            for imgs, seed in zip(input_list, seeds)
        ]
