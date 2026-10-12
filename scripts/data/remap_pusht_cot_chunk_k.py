#!/usr/bin/env python3
"""SHARED PushT instruction builder for the K-action-chunk feedback-interval study.

PushT analog of the sokoban remap_sokoban_chunk_k.py + remap_sokoban_cot_chunk_k.py instruction
builders. This module is import-only (no remap main): the CoT-envfb / non-CoT-envfb / single-shot
PushT remaps and the eval runner all import these so train==eval. PushT-specific vs sokoban:
  - `_rules_prefix` does NOT inject the sokoban two-crate rule (PushT has no crates).
  - FRAME0_LABEL = "Starting frame:" (PushT shows a rendered frame, not a board).
  - `_max_steps` default 30 (PushT env_kwargs.max_steps=30) vs sokoban 20.
The cadence/think wording is task-agnostic ('move' actions; K only sets the action-chunk size;
imagination is per-step). `rules` = the FULL step-0 source prompt; the per-step counter is stripped.
"""


def _rules_prefix(step0_prompt: str) -> str:
    """PushT rules = step-0 prompt with the per-step counter stripped (cut at 'This is step 1.')."""
    cut = step0_prompt.find("This is step 1.")
    return (step0_prompt[:cut] if cut != -1 else step0_prompt).rstrip()


def _max_steps(row: dict) -> int:
    for src in (row.get("env_kwargs"), row.get("init_state")):
        if isinstance(src, dict) and src.get("max_steps") is not None:
            return int(src["max_steps"])
    return 30   # PushT default (env_kwargs.max_steps=30)


def interval_sentence(K: int) -> str:
    return (f"You will be shown a new frame after every {K} actions: at each frame, output up to "
            f"{K} `('move', (x, y))` actions (one per line), then you will see the resulting frame and "
            f"continue. Finish with `('stop', 'stop')` once the gray T covers the green target.")


def chunk_counter(step_idx: int, max_steps: int) -> str:
    return f"This is step {step_idx + 1}. You are allowed to take {max_steps - step_idx - 1} more steps."


# Per-chunk <think> open/close templates: EVERY chunk opens its own imagination and closes before committing.
def think_open(n_actions: int) -> str:
    return ("Let me imagine the next move step by step." if n_actions == 1
            else f"Let me imagine the next {n_actions} moves step by step.")


THINK_CLOSE = "Done imagining; committing these actions:"
FRAME0_LABEL = "Starting frame:"   # labels the initial frame image (frame0), which immediately follows


def cot_instruction(rules: str, max_steps: int, K: int) -> str:
    """User instruction = PushT rules + ONE minimal cadence line (action-chunk size) + the frame0 label;
    the initial frame image follows this text. The rules already state the goal + stop requirement; the env
    enforces the step limit. Imagination is ALWAYS per-step; K only sets the action-chunk size. `max_steps`
    kept for signature compatibility. `rules` is the FULL step-0 source prompt; the per-step counter is stripped."""
    cadence = ("You act one move at a time." if K == 1 else f"You act in chunks of {K} actions.")
    return _rules_prefix(rules).rstrip() + "\n\n" + cadence + "\n\n" + FRAME0_LABEL
