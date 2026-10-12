#!/usr/bin/env python3
"""SHARED Pacman-2D instruction builder for the K-action-chunk feedback-interval study.

Pacman analog of remap_pusht_cot_chunk_k.py. Import-only (no remap main): the CoT-envfb /
non-CoT-envfb / single-shot / cot-inf pacman remaps AND the eval runner import these so train==eval.
Pacman-specific vs pushT:
  - `rules` = the pacman env `get_prompt()` text (already documents the `('move', action_id)` /
    `('stop','stop')` action format, the ghost rule, and the all-food-then-stop success). It has NO
    per-step counter, so `_rules_prefix` returns it unchanged.
  - FRAME0_LABEL = "Starting frame:" (a rendered maze frame follows).
  - `_max_steps` default 80 (pacman easy env_kwargs.max_steps=80; hard=120; rows carry their own).
The cadence wording is task-agnostic (K only sets the action-chunk size; imagination is per-step).
"""


def _rules_prefix(step0_prompt: str) -> str:
    """Pacman rules = the env get_prompt() text (no per-step counter to strip; cut is a no-op)."""
    cut = step0_prompt.find("This is step 1.")
    return (step0_prompt[:cut] if cut != -1 else step0_prompt).rstrip()


def _max_steps(row: dict) -> int:
    for src in (row.get("env_kwargs"), row.get("init_state")):
        if isinstance(src, dict) and src.get("max_steps") is not None:
            return int(src["max_steps"])
    return 20   # pacman easy default (env_kwargs.max_steps=20; hard=40)


def budget_sentence(max_steps: int) -> str:
    """TOTAL step budget, stated once in the instruction (adopts the maze2d convention,
    remap_maze2d_cot_chunk_k_envfb.py:103). Gives the model a horizon cue so it does not infer a
    stop-time from the (short) training-episode length -> reduces premature stopping on longer boards.
    easy=20, hard=40 come from the row's env_kwargs.max_steps."""
    return f"You are allowed to take at most {max_steps} steps in total."


# Per-chunk <think> open/close templates: every chunk opens its own imagination and closes before committing.
def think_open(n_actions: int) -> str:
    return ("Let me imagine the next move step by step." if n_actions == 1
            else f"Let me imagine the next {n_actions} moves step by step.")


THINK_CLOSE = "Done imagining; committing these actions:"
FRAME0_LABEL = "Starting frame:"   # labels the initial rendered maze frame (frame0), which immediately follows


def cot_instruction(rules: str, max_steps: int, K: int) -> str:
    """User instruction = pacman rules + TOTAL step budget + ONE cadence line (action-chunk size)
    + the frame0 label; the initial frame image follows this text. The budget sentence (maze2d-style)
    conveys the horizon (easy=20/hard=40) since the pacman board carries no step counter."""
    cadence = ("You act one move at a time." if K == 1 else f"You act in chunks of {K} actions.")
    return (_rules_prefix(rules).rstrip() + "\n\n"
            + budget_sentence(max_steps) + " " + cadence + "\n\n" + FRAME0_LABEL)


def cadence_inf() -> str:
    return ("You will see ONLY this initial frame and receive no further frames or feedback; "
            "imagine every step and commit the whole plan in one shot.")


def cot_instruction_inf(rules: str, max_steps: int) -> str:
    """K=inf cadence variant (single open-loop chunk), with the total step budget."""
    return (_rules_prefix(rules).rstrip() + "\n\n"
            + budget_sentence(max_steps) + " " + cadence_inf() + "\n\n" + FRAME0_LABEL)
