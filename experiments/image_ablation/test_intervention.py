#!/usr/bin/env python3
"""CPU check that every condition feeds the selected pixels through one encoder call."""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

os.environ.setdefault("AGIMA_CONDITION", "original")

import evaluation.runtime as G
from evaluation.environments import _reset_preserving_init_state
from evaluation.actions import _jpeg_roundtrip
from experiments.image_ablation import run_ablation as runner
from experiments.image_ablation.oracle import OracleReplica, render_rgb
from experiments.image_ablation.test_oracle import first_row


class FakeInferencer:
    def __init__(self):
        self.parts = iter(("<think>('move', 1)<|vision_start|>",
                           "</think>\n('move', 1)"))
        self.images = []
        self.image_calls = 0
        self.tokenizer = SimpleNamespace(encode=lambda text, **_: list(text))
        self.vae_transform = lambda image: torch.zeros(3, 256, 256)
        self.vit_transform = lambda image: torch.zeros(3, 252, 252)

    def init_gen_context(self):
        return {"tokens": []}

    def update_context_text(self, text, context):
        context["tokens"].append(text)
        return context

    def update_context_image(self, image, context, vae=True, vit=True):
        assert vae and vit
        self.images.append(np.asarray(image).copy())
        return context

    def gen_text(self, context, **kwargs):
        assert kwargs["do_sample"] is False
        assert kwargs["max_length"] == 600
        return next(self.parts)

    def gen_image(self, **kwargs):
        assert kwargs["num_timesteps"] == 50
        assert kwargs["cfg_text_scale"] == kwargs["cfg_img_scale"] == 1.0
        self.image_calls += 1
        return Image.new("RGB", (256, 256), (17, 34, 51))


def main() -> None:
    args = SimpleNamespace(profile="sokoban_v8_res256",
                           sokoban_action_mode_override="move_only",
                           sokoban_success_mode="stop_then_solved",
                           sim2real=False, action_preset=None, action_name_map=None)
    row = G.prepare_sokoban_episode(first_row("sokoban"), args)
    env, _ = G.make_profiled_env(row, args)
    try:
        _reset_preserving_init_state(env, seed=row["episode_seed"], init_state=row["init_state"])
        real = _jpeg_roundtrip(render_rgb(env), 95, "JPEG")
        clone = OracleReplica(env)
        try:
            oracle = _jpeg_roundtrip(clone.step("('move', 1)"), 95, "JPEG")
        finally:
            clone.close()
        generated = _jpeg_roundtrip(Image.new("RGB", (256, 256), (17, 34, 51)), 95, "JPEG")
        blank = _jpeg_roundtrip(Image.new("RGB", (256, 256), (128, 128, 128)), 95, "JPEG")
        expected = {"original": generated, "blank": blank, "stale": real, "oracle": oracle}
        for condition, image in expected.items():
            runner.CONDITION = condition
            fake = FakeInferencer()
            output = runner._generate_chunk(fake, ["instruction", real], env, real, 95, "JPEG")
            assert fake.image_calls == 1, condition
            assert len(fake.images) == 2, condition  # unchanged real observation; one imagined frame
            assert np.array_equal(fake.images[0], np.asarray(real)), condition
            assert np.array_equal(fake.images[1], np.asarray(image)), condition
            assert output["audit"]["identical"] and output["imagined_actions"] == ["('move', 1)"]
            assert output["used"][0].size == (256, 256)
            if condition == "original":
                assert len(output["mse"]) == 1
        print("Four imagined-frame interventions encode exactly the selected pixels")
    finally:
        env.close()


if __name__ == "__main__":
    main()
