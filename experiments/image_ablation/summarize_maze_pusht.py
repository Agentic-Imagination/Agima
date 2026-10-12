"""Summarize complete Maze and PushT K=5 paired frame-ablation runs."""

import ast
import csv
import json
import math
from collections import Counter
from pathlib import Path

from experiments.image_ablation.preflight_maze_pusht import CONFIG, ROOT, archived_stats
from experiments.image_ablation.summarize import mcnemar


OUT = ROOT / "output"
CONDITIONS = ("No-Imagine", "Original", "Blank", "Stale", "Oracle GT")
RUN_NAMES = {"Original": "original", "Blank": "blank", "Stale": "stale", "Oracle GT": "oracle"}


def _rows(task, condition):
    if condition == "No-Imagine":
        path = ROOT / "prepared" / "maze_pusht_baseline_metrics.jsonl"
        return [row for row in map(json.loads, path.read_text().splitlines()) if row["task"] == task]
    paths = sorted((ROOT / "runs" / f"{task}_{RUN_NAMES[condition]}").rglob("episode_stats.json"))
    return [json.loads(path.read_text())["result"] for path in paths]


def main():
    OUT.mkdir(exist_ok=True)
    rows_by = {}
    all_rows = []
    summary = []
    tests = []
    terminations = []
    mse_rows = []
    alignment = []
    for task, spec in CONFIG.items():
        arms = {}
        for condition in CONDITIONS:
            rows = _rows(task, condition)
            if len(rows) != 100:
                raise RuntimeError(f"{task} {condition}: only {len(rows)}/100 episodes completed")
            by_id = {row["episode_id"]: row for row in rows}
            if len(by_id) != 100:
                raise RuntimeError(f"{task} {condition}: duplicate episode identifiers")
            if arms and set(by_id) != set(next(iter(arms.values()))):
                raise RuntimeError(f"{task} {condition}: episode set differs from the staged baseline")
            arms[condition] = by_id
            all_rows.extend(sorted(rows, key=lambda row: row["episode_id"]))
            successes = sum(bool(row["success"]) for row in rows)
            parses = sum(bool(row["parse_failure"]) for row in rows)
            invalids = sum(int(row["num_invalid_actions"]) for row in rows)
            moves = sum(int(row["num_executed_actions"]) - int(bool(row["stop_emitted"]))
                        for row in rows)
            summary.append({"task": task, "condition": condition, "successes": successes,
                            "n": 100, "success_pct": successes,
                            "parse_failures": parses, "parse_failure_pct": parses,
                            "invalid_actions": invalids, "attempted_moves": moves,
                            "invalid_action_pct": round(100 * invalids / moves, 3) if moves else None})
            terminations.append({"task": task, "condition": condition,
                                 "counts": dict(Counter(row["termination_reason"] for row in rows))})
        for condition in CONDITIONS:
            if condition != "Original":
                tests.append({"task": task, "condition": condition,
                              **mcnemar(arms["Original"], arms[condition])})
        published = archived_stats(spec["cot"])
        paper = {f"{task}:{int(row['global_index']):07d}:{int(row['episode_seed'])}":
                 {"success": bool(row["stats"]["success"])} for row in published}
        if set(arms["Original"]) != set(paper):
            raise RuntimeError(f"{task}: Original episode set differs from the staged archive")
        alignment.append({"task": task,
                          "measured_successes": sum(row["success"] for row in arms["Original"].values()),
                          "archive_successes": sum(row["success"] for row in paper.values()),
                          **mcnemar(arms["Original"], paper)})
        for episode_id, row in sorted(arms["Original"].items()):
            if len(row["pixel_mse"]) != row["n_imagined"]:
                raise RuntimeError(f"{episode_id}: missing Oracle MSE frame")
            for index, value in enumerate(row["pixel_mse"], start=1):
                mse_rows.append({"task": task, "episode_id": episode_id,
                                 "imagined_frame_index": index, "pixel_mse_0_255": value})
        rows_by[task] = arms

    with (OUT / "maze_pusht_results.jsonl").open("w") as stream:
        for row in all_rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    for name, records, fields in (
        ("maze_pusht_summary.csv", summary, list(summary[0])),
        ("maze_pusht_world_model_pixel_mse.csv", mse_rows,
         ["task", "episode_id", "imagined_frame_index", "pixel_mse_0_255"]),
    ):
        with (OUT / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    (OUT / "maze_pusht_mcnemar.json").write_text(json.dumps(tests, indent=2) + "\n")
    (OUT / "maze_pusht_termination_reasons.json").write_text(json.dumps(terminations, indent=2) + "\n")

    lookup = {(row["task"], row["condition"]): row for row in summary}
    report = ["# Maze2D hard and PushT in-distribution imagined-frame ablation", "",
              "`maze_pusht_summary.csv` contains counts from the supplied episode records.",
              "No-Imagine uses replayed archive records; Original, Blank, Stale and Oracle GT",
              "use the current run directories. Archive counts are shown separately below.", "",
              "## Measured Original versus archive", ""]
    for item in alignment:
        report.append(f"- {item['task']}: measured {item['measured_successes']}/100; "
                      f"archive {item['archive_successes']}/100; paired discordance "
                      f"({item['b_original_only']}, {item['c_other_only']}), "
                      f"McNemar p={item['p_exact_two_sided']:.4g}.")
    report.extend(["", "## Measured condition differences", ""])
    for task in CONFIG:
        original = lookup[(task, "Original")]["successes"]
        for condition in ("Blank", "Stale", "Oracle GT"):
            successes = lookup[(task, condition)]["successes"]
            report.append(
                f"- {task} / {condition}: {successes}/100 versus measured Original "
                f"{original}/100; difference {successes - original:+d} percentage points."
            )
    pusht = next(item for item in alignment if item["task"] == "pusht")
    report.append(
        f"PushT measured Original differs from the archive by "
        f"{pusht['measured_successes'] - pusht['archive_successes']:+d} percentage points."
    )
    if abs(pusht["measured_successes"] - pusht["archive_successes"]) > 10:
        paper_actions = {int(row["global_index"]): [step["action"] for step in row["history"]]
                         for row in archived_stats(CONFIG["pusht"]["cot"])}
        measured_actions = {
            int(row["global_index"]): [step["action"] for step in row["history"]]
            for path in (ROOT / "runs/pusht_original").rglob("episode_stats.json")
            for row in [json.loads(path.read_text())]
        }
        first_same = sum(paper_actions[i][:1] == measured_actions[i][:1]
                         for i in paper_actions)
        first_five_same = sum(paper_actions[i][:5] == measured_actions[i][:5]
                              for i in paper_actions)
        first_move_distances = []
        for index in paper_actions:
            if not paper_actions[index] or not measured_actions[index]:
                continue
            published_first = ast.literal_eval(paper_actions[index][0])
            measured_first = ast.literal_eval(measured_actions[index][0])
            if published_first[0] == measured_first[0] == "move":
                first_move_distances.append(math.dist(published_first[1], measured_first[1]))
        distance = (f"{max(first_move_distances):.6f}" if first_move_distances
                    else "not available (no paired initial move actions)")
        report.append(
            f"At the first action, {first_same}/100 action prefixes matched the archive; "
            f"the largest target-coordinate difference among paired move actions was "
            f"{distance}. After five actions, {first_five_same}/100 action prefixes matched."
        )
    report.extend(["", "## Success and diagnostics", "",
                   "| Task | Condition | Success | Parse failure | Invalid actions / moves |",
                   "|---|---|---:|---:|---:|"])
    for task in CONFIG:
        for condition in CONDITIONS:
            row = lookup[(task, condition)]
            report.append(f"| {task} | {condition} | {row['successes']}/100 | "
                          f"{row['parse_failures']}/100 | {row['invalid_actions']}/{row['attempted_moves']} |")
    report.extend(["", "## Exact paired McNemar tests versus measured Original", "",
                   "| Task | Condition | b: Original only | c: Other only | p |",
                   "|---|---|---:|---:|---:|"])
    for row in tests:
        report.append(f"| {row['task']} | {row['condition']} | {row['b_original_only']} | "
                      f"{row['c_other_only']} | {row['p_exact_two_sided']:.4g} |")
    report.extend(["", "## Termination reasons", ""])
    for row in terminations:
        report.append(f"- {row['task']} / {row['condition']}: " +
                      ", ".join(f"{key}={value}" for key, value in sorted(row["counts"].items())))
    report.extend(["", "## Original world-model pixel MSE", ""])
    for task in CONFIG:
        values = [row["pixel_mse_0_255"] for row in mse_rows if row["task"] == task]
        report.append(f"- {task}: {len(values)} frames; mean MSE "
                      f"{sum(values) / len(values):.6f} on 0–255 RGB." if values else
                      f"- {task}: no imagined frames.")
    report.extend(["", "## Trace locations", "",
                   "Episode records are read from `runs/{task}_{condition}/`.",
                   "When generated, context-image checks are saved in",
                   "`output/{task}_{condition}_first10_audit.json`.", ""])
    (OUT / "maze_pusht_report.md").write_text("\n".join(report))
    print(json.dumps({"alignment": alignment, "summary": summary, "tests": tests}, indent=2))


if __name__ == "__main__":
    main()
