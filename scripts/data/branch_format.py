#!/usr/bin/env python3
"""Shared counterfactual branch format for CoT-inf training rows.

Branching splices ONE counterfactual detour into a CoT-inf single-turn row
at a critical step: inside the model's <think>, right before it commits the expert
action a_j, it first imagines an ALTERNATIVE action, renders the resulting frame,
recognizes the alternative is bad, and backtracks to the committed line. The
post-</think> committed plan is IDENTICAL to the unbranched row — only the private
imagination gains a rejected detour.

    ...expert imagination up to step j-1...  a_{j-1}<img_{j-1}>
    <BRANCH_OPEN>
      alt_a_j <alt_img_j>            <- MSE targets (loss=True), CE on the alt text
    <BACKTRACK> <BRANCH_CLOSE>
    a_j<img_j>  ...rest of episode...          </think>
    {full committed plan, UNCHANGED}

WHY THE THREE MARKERS ARE PLAIN NATURAL LANGUAGE (not special tokens):
Every assistant text run is fed to the loader via _add_text(..., need_loss=True),
so the marker phrases are CE-trained as ordinary text. No tokenizer/vocab change is
needed; the model learns "open a hypothetical / it failed / return to the plan"
purely from these recurring phrases.

LOADER CONTRACT (data/interleave_datasets/edit_dataset_envfb.py :: parse_row),
which this format is built to satisfy for a CoT-inf SINGLE-TURN open-loop row:
  * An assistant image with loss=False is DROPPED (parse_row ~:69-74). Therefore
    every alt-branch imagined frame MUST carry loss=True (it becomes an MSE target).
  * A loss-0 CONTEXT image is only honored in a USER turn (~:114-119). A single-turn
    open-loop row has exactly ONE such image: frame0 in the user turn. The assistant
    turn must contain NO loss-0 image.
  * parse_row concatenates consecutive text runs, prepends "<think>" to the first
    text seg of the assistant turn, and strips a leading "<think>" only from the
    LAST seg (~:94-98). Splicing a branch strictly BETWEEN the think-opener seg and
    the committed-plan seg leaves both of those untouched, so <think> handling and
    the verbatim committed plan are preserved.

This module is pure data-shaping (no torch / no loader import) so remap workers can
import it cheaply. Callers: the Sokoban and PushT branch remappers.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# (a) The three fixed natural-language markers. CE-trained automatically (they
#     are just assistant text). Keep them stable across the whole Exp2 study so
#     sokoban and pusht share one vocabulary of branch cues.
# ---------------------------------------------------------------------------
BRANCH_OPEN = "Let me first consider an alternative move:"
BACKTRACK = "This alternative does not work out, so I abandon it."
BRANCH_CLOSE = "Returning to the main line, I commit to:"

# ---------------------------------------------------------------------------
# (c) Metadata: schema version + the row-level keys this format stamps.
# ---------------------------------------------------------------------------
SCHEMA_VERSION = "exp2_branch_v1"
META_KEYS: Tuple[str, ...] = (
    "has_branch",
    "branch_step",
    "n_branch_steps",
    "n_branch_frames",
    "schema_version",
)

# Type of a single counterfactual step supplied by the caller: (alt_action_text,
# alt_frame_b64_or_None). The frame, when present, is emitted as a loss=True image.
AltStep = Tuple[str, Optional[str]]


def count_branch(alt_text_frames: Sequence[AltStep]) -> Tuple[int, int]:
    """Return (n_branch_steps, n_branch_frames) for an alt segment.

    n_branch_steps = number of counterfactual (action) steps.
    n_branch_frames = number of loss=True imagined frames those steps add.
    """
    n_steps = len(alt_text_frames)
    n_frames = sum(1 for item in alt_text_frames if item[1] is not None)
    return n_steps, n_frames


def splice_branch(
    assistant_content_list: List[dict],
    at_index: int,
    alt_text_frames: Sequence[AltStep],
    open_text: Optional[str] = None,
    close_text: Optional[str] = None,
) -> List[dict]:
    """Splice a counterfactual branch into a CoT-inf assistant `content` list.

    `open_text`/`close_text` override the branch wrapper wording (default: the
    detour-style BRANCH_OPEN / BACKTRACK+BRANCH_CLOSE constants). Pass e.g. a
    candidate-COMPARISON wording ("let me weigh a few candidate moves" /
    "none covers as much, so I choose:") for multi-candidate branches — the
    splice mechanics, loss flags, and committed-plan preservation are identical.

    Args:
      assistant_content_list: the assistant message's `content` list — the
        interleaved [text, image, text, image, ..., committed-plan-text] built by
        remap_{sokoban,pusht}_cot_chunk_inf. Element 0 carries the <think> opener;
        the LAST element is the committed post-</think> plan.
      at_index: index in that list of the content dict that BEGINS the critical
        step (the text run holding the expert action a_j). The branch is inserted
        immediately BEFORE this index. Must be in [1, len-1] so it lands strictly
        after the think-opener seg (index 0) and before the committed-plan seg
        (the last element), both of which must stay verbatim. Use
        `step_action_index` to map a step number to this index on a clean row.
      alt_text_frames: the counterfactual rollout as a list of
        (alt_action_text, alt_frame_b64_or_None) tuples. Each tuple emits a text
        run and, if a frame is given, a loss=True image (MSE target). Must be
        non-empty.

    Returns:
      A NEW content list: assistant_content_list[:at_index] + <branch> +
      assistant_content_list[at_index:]. The branch is wrapped BRANCH_OPEN ...
      alt steps ... BACKTRACK + BRANCH_CLOSE. The input list is not mutated and
      every existing content dict (including the committed-plan tail) is reused by
      reference, i.e. left byte-for-byte identical.

    Raises:
      TypeError / ValueError on a malformed content list, out-of-range at_index,
      or empty alt_text_frames.
    """
    if not isinstance(assistant_content_list, list):
        raise TypeError("assistant_content_list must be a list of content dicts")
    n = len(assistant_content_list)
    if n < 2:
        raise ValueError(
            "assistant_content_list too short: need at least a think-opener seg "
            "and a committed-plan seg")
    if not (1 <= at_index <= n - 1):
        raise ValueError(
            f"at_index={at_index} out of range [1, {n - 1}]: the branch must land "
            f"after the <think> opener (index 0) and before the committed plan "
            f"(index {n - 1}), which must stay unchanged")
    if not alt_text_frames:
        raise ValueError("alt_text_frames is empty; nothing to splice")

    branch: List[dict] = [
        {"type": "text", "text": "\n" + (BRANCH_OPEN if open_text is None else str(open_text))}
    ]
    for item in alt_text_frames:
        if not (isinstance(item, (tuple, list)) and len(item) == 2):
            raise ValueError(
                "each alt_text_frames item must be a (alt_text, alt_frame_b64|None) pair")
        alt_text, alt_frame = item
        if alt_text:
            branch.append({"type": "text", "text": "\n" + str(alt_text)})
        if alt_frame is not None:
            # loss=True is MANDATORY: the envfb loader DROPS assistant images with
            # loss=False, so a loss-0 alt frame would silently vanish. As loss=True
            # it becomes an MSE target (the imagined counterfactual board).
            branch.append({"type": "image", "image": alt_frame, "loss": True})
    branch.append({
        "type": "text",
        "text": "\n" + (BACKTRACK + "\n" + BRANCH_CLOSE if close_text is None else str(close_text)),
    })

    return assistant_content_list[:at_index] + branch + assistant_content_list[at_index:]


def step_action_index(assistant_content_list: Sequence[dict], step: int) -> int:
    """Content index of `step`'s expert-action text run, for a CLEAN CoT-inf row.

    On an unbranched CoT-inf content list every step is exactly (action-text,
    imagined-frame), so step j's action text is at index 2*j and its frame at
    2*j+1, with the committed plan as the final element. Pass the result as
    `at_index` to `splice_branch` to branch BEFORE that step. Validates that the
    computed slot is in fact a text dict (guards against a step with no frame,
    which would break the 2*j stride).
    """
    idx = 2 * step
    if not (0 <= idx < len(assistant_content_list)):
        raise ValueError(f"step={step} -> index {idx} out of range for content of "
                         f"length {len(assistant_content_list)}")
    if assistant_content_list[idx].get("type") != "text":
        raise ValueError(
            f"content[{idx}] is not a text run (type="
            f"{assistant_content_list[idx].get('type')!r}); the row is not a clean "
            f"one-frame-per-step CoT-inf list, compute at_index explicitly")
    return idx


def branch_metadata(
    has_branch: bool,
    branch_step: Optional[int] = None,
    n_branch_steps: int = 0,
    n_branch_frames: int = 0,
) -> dict:
    """Build the Exp2 branch metadata dict (keys == META_KEYS).

    branch_step is the 0-based STEP index the branch was spliced before (-1 when
    there is no branch). n_branch_steps / n_branch_frames come from count_branch.
    """
    return {
        "has_branch": bool(has_branch),
        "branch_step": (int(branch_step) if branch_step is not None else -1),
        "n_branch_steps": int(n_branch_steps),
        "n_branch_frames": int(n_branch_frames),
        "schema_version": SCHEMA_VERSION,
    }


def no_branch_metadata() -> dict:
    """Metadata for a control / unbranched row (has_branch=False, branch_step=-1)."""
    return branch_metadata(False)
