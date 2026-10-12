#!/usr/bin/env python3
# Modified for Agima: portable paths and explicit dataset arguments.
"""Full streaming audit for Pacman shared-prefix forward-thinking K=inf data."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Optional


_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


from scripts.data.remap_pacman_cot_chunk_inf_forward_shared import (
    SCHEMA_VERSION,
    STOP_ACTION,
    TOKEN_BUDGET,
    _assistant_images,
    _committed_actions,
    _estimate_tokens,
    _move_action,
    _parse_action,
    _source_episode,
    _write_json,
)


def _rows(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _frame_actions(row: dict[str, Any]) -> list[Optional[str]]:
    actions: list[Optional[str]] = []
    pending_text = ""
    assistant = next(message for message in row["messages"] if message.get("role") == "assistant")
    for item in assistant.get("content") or []:
        if item.get("type") == "text":
            pending_text += str(item.get("text", ""))
            continue
        if item.get("type") != "image":
            continue
        action: Optional[str] = None
        for line in pending_text.splitlines():
            line = line.strip()
            if not line.startswith("("):
                continue
            try:
                kind, payload = _parse_action(line)
            except Exception:
                continue
            if kind == "move":
                action = _move_action(int(payload))
        actions.append(action)
        pending_text = ""
    return actions


def _text(row: dict[str, Any]) -> str:
    return "".join(
        str(item.get("text", ""))
        for message in row.get("messages") or []
        if message.get("role") == "assistant"
        for item in message.get("content") or []
        if item.get("type") == "text"
    )


def _error(errors: Counter, examples: list[dict[str, Any]], code: str, context: dict[str, Any]) -> None:
    errors[code] += 1
    if len(examples) < 50:
        examples.append({"code": code, **context})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )
    parser.add_argument("--expected-rows", type=int, default=100_000)
    parser.add_argument("--max-imagined-frames", type=int, default=40)
    parser.add_argument("--min-prefix-steps", type=int, default=2)
    parser.add_argument("--max-prefix-steps", type=int, default=10)
    parser.add_argument("--min-regret-moves", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_shards = sorted((args.source_root / "data" / "train").glob("*.jsonl.gz"))
    output_shards = sorted((args.output_root / "training").glob("*.jsonl.gz"))
    if not source_shards or len(source_shards) != len(output_shards):
        raise RuntimeError(
            f"source/output shard mismatch: {len(source_shards)} vs {len(output_shards)}"
        )
    if [p.name for p in source_shards] != [p.name for p in output_shards]:
        raise RuntimeError("source/output shard names differ")

    counts: Counter = Counter()
    errors: Counter = Counter()
    examples: list[dict[str, Any]] = []
    prefix_hist: Counter = Counter()
    regret_hist: Counter = Counter()
    frame_hist: Counter = Counter()
    expert_len_hist: Counter = Counter()
    alternate_total_hist: Counter = Counter()
    output_hashes: set[str] = set()
    output_seeds: set[int] = set()
    shard_summaries: list[dict[str, Any]] = []

    for source_path, output_path in zip(source_shards, output_shards):
        source_iter = _rows(source_path)
        output_iter = _rows(output_path)
        shard_rows = 0
        while True:
            try:
                source = next(source_iter)
            except StopIteration:
                source = None
            try:
                output = next(output_iter)
            except StopIteration:
                output = None
            if source is None and output is None:
                break
            shard_rows += 1
            counts["rows"] += 1
            context = {
                "shard": source_path.name,
                "row": shard_rows - 1,
                "init_state_hash": (output or source or {}).get("init_state_hash"),
            }
            if source is None or output is None:
                _error(errors, examples, "source_output_row_count_mismatch", context)
                break
            try:
                rules, initial_frame, moves, move_actions, _ = _source_episode(source)
                all_actions = [*move_actions, STOP_ACTION]
                expert_len = len(moves)
                if expert_len > 27:
                    raise AssertionError(f"source expert length {expert_len} exceeds 27")
                expert_len_hist[str(expert_len)] += 1
                if output.get("init_state_hash") != source.get("init_state_hash"):
                    raise AssertionError("source/output init_state_hash order mismatch")
                if output.get("episode_seed") != source.get("episode_seed"):
                    raise AssertionError("source/output episode_seed order mismatch")
                init_hash = str(output["init_state_hash"])
                seed = int(output["episode_seed"])
                if init_hash in output_hashes:
                    raise AssertionError("duplicate output init_state_hash")
                if seed in output_seeds:
                    raise AssertionError("duplicate output episode_seed")
                output_hashes.add(init_hash)
                output_seeds.add(seed)

                if "history" in output:
                    raise AssertionError("output unexpectedly retains raw history")
                if output.get("schema_version") != SCHEMA_VERSION:
                    raise AssertionError("wrong schema version")
                if output.get("chunk_size") != "inf" or output.get("n_chunks") != 1:
                    raise AssertionError("not a single K=inf chunk")
                if output.get("n_feedback_turns") != 0 or output.get("n_feedback_frames") != 0:
                    raise AssertionError("K=inf output unexpectedly contains feedback")
                messages = output.get("messages") or []
                if len(messages) != 2 or [m.get("role") for m in messages] != ["user", "assistant"]:
                    raise AssertionError("bad message roles")
                user_images = [x for x in messages[0].get("content") or [] if x.get("type") == "image"]
                if len(user_images) != 1 or user_images[0].get("image") != initial_frame:
                    raise AssertionError("initial user frame differs from source")
                if bool(user_images[0].get("loss", True)):
                    raise AssertionError("initial user frame is not loss=False")
                if str(args.max_imagined_frames) not in str(messages[0]["content"][0].get("text", "")):
                    raise AssertionError("instruction does not state the imagined-frame budget")

                assistant_images = _assistant_images(output)
                image_values = [str(item.get("image", "")) for item in assistant_images]
                if any(not image or not bool(item.get("loss")) for image, item in zip(image_values, assistant_images)):
                    raise AssertionError("assistant image is empty or not loss=True")
                if len(image_values) != int(output.get("n_imagined_frames", -1)):
                    raise AssertionError("assistant image count mismatch")
                if len(image_values) > args.max_imagined_frames:
                    raise AssertionError("assistant image budget exceeded")
                frame_hist[str(len(image_values))] += 1
                counts["max_imagined_frames"] = max(counts["max_imagined_frames"], len(image_values))
                frame_actions = _frame_actions(output)
                if len(frame_actions) != len(image_values):
                    raise AssertionError("could not align frame actions")
                if _committed_actions(output) != all_actions:
                    raise AssertionError("committed actions differ from source expert")
                if int(output.get("estimated_tokens", TOKEN_BUDGET)) != _estimate_tokens(messages):
                    raise AssertionError("estimated token count is stale")
                if int(output["estimated_tokens"]) >= TOKEN_BUDGET:
                    raise AssertionError("token budget exceeded")
                counts["max_estimated_tokens"] = max(
                    counts["max_estimated_tokens"], int(output["estimated_tokens"])
                )

                text = _text(output)
                if output.get("has_plan_comparison"):
                    counts["pair_rows"] += 1
                    if int(output.get("n_candidates", 0)) != 2:
                        raise AssertionError("pair row does not have two candidates")
                    decision = int(output.get("decision_step", -1))
                    if not args.min_prefix_steps <= decision <= args.max_prefix_steps:
                        raise AssertionError("decision step outside configured window")
                    regret = int(output.get("regret_moves", 0))
                    if regret < args.min_regret_moves:
                        raise AssertionError("pair regret below threshold")
                    prefix_hist[str(decision)] += 1
                    regret_hist[str(regret)] += 1
                    alternate_total_hist[str(int(output["alternate_full_total_moves"]))] += 1
                    candidates = output.get("candidates") or []
                    if sum(c.get("source") == "expert_shortest" for c in candidates) != 1:
                        raise AssertionError("pair does not contain one expert candidate")
                    if sum(c.get("source") == "alternate_shortest_after_divergence" for c in candidates) != 1:
                        raise AssertionError("pair does not contain one alternate candidate")
                    chosen_index = int(output.get("chosen_candidate", 0)) - 1
                    if chosen_index not in {0, 1} or candidates[chosen_index]["source"] != "expert_shortest":
                        raise AssertionError("chosen candidate is not the expert")
                    counts[f"chosen_candidate:{chosen_index + 1}"] += 1
                    expected_swap = hashlib.sha256(
                        f"{init_hash}|{decision}|{SCHEMA_VERSION}".encode("utf-8")
                    ).digest()[0] % 2 == 1
                    if bool(output.get("candidate_order_shuffled")) != expected_swap:
                        raise AssertionError("candidate order shuffle is not deterministic")
                    if any(c.get("full_status") != "solved" for c in candidates):
                        raise AssertionError("comparison uses a non-solved full candidate")
                    alternate = next(c for c in candidates if c["source"].startswith("alternate"))
                    expert = next(c for c in candidates if c["source"] == "expert_shortest")
                    if int(expert["full_total_moves"]) != expert_len:
                        raise AssertionError("expert candidate total length mismatch")
                    if (
                        expert.get("status") != "solved"
                        or expert.get("full_status") != "solved"
                        or int(expert.get("displayed_continuation_moves", -1))
                        != int(expert.get("full_continuation_moves", -2))
                        or int(expert.get("displayed_total_moves", -1))
                        != int(expert.get("full_total_moves", -2))
                    ):
                        raise AssertionError("expert candidate is not complete and visibly solved")
                    counts["rows_with_complete_visible_expert"] += 1
                    if int(alternate["full_total_moves"]) - expert_len != regret:
                        raise AssertionError("alternate regret arithmetic mismatch")
                    if alternate.get("status") == "comparison_budget_reached":
                        counts["alternate_comparison_budget_reached"] += 1
                    elif alternate.get("status") != "solved":
                        raise AssertionError("unsupported visible alternate status")

                    layout = output.get("frame_layout") or {}
                    n_shared = int(layout.get("shared_prefix", -1))
                    n_first = int(layout.get("candidate_1", -1))
                    n_reset = int(layout.get("checkpoint_reset", -1))
                    n_second = int(layout.get("candidate_2", -1))
                    if n_shared != decision or n_reset != 1:
                        raise AssertionError("bad shared-prefix/reset layout")
                    if n_shared + n_first + n_reset + n_second != len(image_values):
                        raise AssertionError("frame layout does not sum to images")
                    if image_values[:decision] != [
                        str(step.get("image_next"))
                        for step in source["history"]
                        if _parse_action(str(step.get("action") or step.get("vlm_output")))[0] == "move"
                    ][:decision]:
                        raise AssertionError("shared prefix frames differ from source")
                    if frame_actions[:decision] != move_actions[:decision]:
                        raise AssertionError("shared prefix actions differ from source")
                    reset_index = decision + n_first
                    if image_values[reset_index] != image_values[decision - 1] or frame_actions[reset_index] is not None:
                        raise AssertionError("checkpoint reset does not restore the decision frame")
                    segments = [
                        (frame_actions[decision : decision + n_first], image_values[decision : decision + n_first]),
                        (frame_actions[reset_index + 1 :], image_values[reset_index + 1 :]),
                    ]
                    for candidate, (actions, frames) in zip(candidates, segments):
                        if len(actions) != int(candidate["displayed_continuation_moves"]):
                            raise AssertionError("candidate displayed action count mismatch")
                        if not actions or actions[0] != candidate["first_action"]:
                            raise AssertionError("candidate first action mismatch")
                        if any(action is None for action in actions):
                            raise AssertionError("candidate contains a frame without a move action")
                        if candidate["source"] == "expert_shortest":
                            if actions != move_actions[decision:]:
                                raise AssertionError("expert continuation actions differ from source")
                            source_move_frames = [
                                str(step.get("image_next"))
                                for step in source["history"]
                                if _parse_action(str(step.get("action") or step.get("vlm_output")))[0] == "move"
                            ]
                            if frames != source_move_frames[decision:]:
                                raise AssertionError("expert continuation frames differ from source")
                    if text.count("Candidate 1:") != 1 or text.count("Candidate 2:") != 1:
                        raise AssertionError("pair candidate headers missing")
                    if text.count("Return to the exact decision checkpoint") != 1:
                        raise AssertionError("pair reset marker missing")
                else:
                    counts["single_rows"] += 1
                    if int(output.get("n_candidates", 0)) != 1 or not output.get("single_candidate_fallback"):
                        raise AssertionError("bad single fallback metadata")
                    candidates = output.get("candidates") or []
                    if len(candidates) != 1:
                        raise AssertionError("single fallback does not contain exactly one candidate")
                    expert = candidates[0]
                    if (
                        expert.get("source") != "expert_shortest"
                        or expert.get("status") != "solved"
                        or expert.get("full_status") != "solved"
                        or int(expert.get("displayed_continuation_moves", -1))
                        != int(expert.get("full_continuation_moves", -2))
                        or int(expert.get("displayed_total_moves", -1))
                        != int(expert.get("full_total_moves", -2))
                        or int(expert.get("full_total_moves", -1)) != expert_len
                    ):
                        raise AssertionError("single fallback expert is not complete and visibly solved")
                    counts["rows_with_complete_visible_expert"] += 1
                    if frame_actions != move_actions:
                        raise AssertionError("single fallback imagined actions differ from expert")
                    source_move_frames = [
                        str(step.get("image_next"))
                        for step in source["history"]
                        if _parse_action(str(step.get("action") or step.get("vlm_output")))[0] == "move"
                    ]
                    if image_values != source_move_frames:
                        raise AssertionError("single fallback frames differ from expert")
                    if "Availability check:" not in text:
                        raise AssertionError("single fallback lacks availability text")
                counts["successful_rows"] += 1
            except Exception as exc:
                _error(errors, examples, type(exc).__name__ + ":" + str(exc), context)

        shard_summaries.append(
            {
                "source": str(source_path),
                "output": str(output_path),
                "rows": shard_rows,
                "source_compressed_bytes": source_path.stat().st_size,
                "output_compressed_bytes": output_path.stat().st_size,
            }
        )

    if counts["rows"] != args.expected_rows:
        errors["wrong_total_rows"] += 1
    if len(output_hashes) != counts["rows"]:
        errors["wrong_unique_hash_count"] += 1
    summary_path = args.output_root / "metadata" / "forward_summary.json"
    generation_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
    if generation_summary and int(generation_summary.get("totals", {}).get("rows", -1)) != counts["rows"]:
        errors["generation_summary_row_count_mismatch"] += 1

    audit = {
        "schema_version": "pacman_forward_shared_compare2_kinf_full_audit_v1",
        "dataset_schema_version": SCHEMA_VERSION,
        "source_root": str(args.source_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "passed": not errors,
        "errors": dict(errors),
        "error_examples": examples,
        "counts": dict(counts),
        "unique_init_state_hashes": len(output_hashes),
        "unique_episode_seeds": len(output_seeds),
        "prefix_histogram": dict(sorted(prefix_hist.items(), key=lambda item: int(item[0]))),
        "regret_histogram": dict(sorted(regret_hist.items(), key=lambda item: int(item[0]))),
        "imagined_frame_histogram": dict(sorted(frame_hist.items(), key=lambda item: int(item[0]))),
        "expert_move_histogram": dict(sorted(expert_len_hist.items(), key=lambda item: int(item[0]))),
        "alternate_full_total_histogram": dict(
            sorted(alternate_total_hist.items(), key=lambda item: int(item[0]))
        ),
        "compressed_output_bytes": sum(item["output_compressed_bytes"] for item in shard_summaries),
        "shards": shard_summaries,
        "checks": [
            "all source/output gzip streams read to EOF in row-aligned order",
            "100k unique init-state hashes and episode seeds",
            "committed actions exactly equal each source expert trajectory including STOP",
            "initial user frame exactly equals source frame0 and has loss=False",
            "all assistant images are populated loss=True targets",
            "shared prefix, expert continuation, and checkpoint reset frames equal source bytes",
            "candidate order shuffle recomputed from init_state_hash, decision step, and schema",
            "only complete BFS-solved longer alternatives appear in comparisons",
            "every row contains exactly one complete visible solved expert candidate",
            f"imagined frames never exceed {args.max_imagined_frames} and estimated tokens remain below 50k",
            "single-plan fallbacks preserve the complete source expert rollout",
        ],
    }
    output_path = args.output_root / "metadata" / "full_corpus_audit.json"
    _write_json(output_path, audit)
    print(json.dumps({"audit": str(output_path.resolve()), "passed": audit["passed"], "counts": dict(counts), "errors": dict(errors)}, indent=2))
    return 0 if audit["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
