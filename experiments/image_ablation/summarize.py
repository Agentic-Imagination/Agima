#!/usr/bin/env python3
"""Make the paired paper outputs only after all four conditions finish."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

from experiments.image_ablation.preflight import EXPECTED, ROOT, archived_stats


CONDITIONS = ("No-Imagine", "Original", "Blank", "Stale", "Oracle GT")
RUN_NAMES = {"Original": "original", "Blank": "blank", "Stale": "stale", "Oracle GT": "oracle"}


def wilson(successes: int, n: int) -> tuple[float, float]:
    if n < 1:
        return float("nan"), float("nan")
    z = 1.959963984540054
    p = successes / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return center - radius, center + radius


def mcnemar(original: dict[str, dict], other: dict[str, dict]) -> dict:
    b = sum(bool(original[key]["success"]) and not bool(other[key]["success"])
            for key in original)
    c = sum(not bool(original[key]["success"]) and bool(other[key]["success"])
            for key in original)
    n = b + c
    p = min(1.0, 2.0 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2 ** n)) if n else 1.0
    return {"b_original_only": b, "c_other_only": c, "p_exact_two_sided": p}


def read_rows(task: str, condition: str) -> list[dict]:
    if condition == "No-Imagine":
        path = ROOT / "prepared" / "baseline_metrics.jsonl"
        return [row for row in map(json.loads, path.read_text().splitlines()) if row["task"] == task]
    run_dir = ROOT / "runs" / f"{task}_{RUN_NAMES[condition]}"
    paths = sorted(run_dir.rglob("episode_stats.json"))
    return [json.loads(path.read_text())["result"] for path in paths]


def summarize(tasks: tuple[str, ...], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    rows_by = {}
    all_results = []
    csv_rows = []
    tests = []
    terminations = []
    for task in tasks:
        n = EXPECTED[task]["n"]
        task_rows = {}
        for condition in CONDITIONS:
            rows = read_rows(task, condition)
            if len(rows) != n:
                raise RuntimeError(f"{task} {condition}: {len(rows)}/{n} episodes; complete this cell first")
            by_id = {row["episode_id"]: row for row in rows}
            if len(by_id) != n:
                raise RuntimeError(f"{task} {condition}: duplicate episode_id")
            if task_rows and set(by_id) != set(next(iter(task_rows.values()))):
                raise RuntimeError(f"{task} {condition}: episode set differs from the staged baseline")
            task_rows[condition] = by_id
            all_results.extend(sorted(rows, key=lambda row: row["episode_id"]))
            successes = sum(bool(row["success"]) for row in rows)
            low, high = wilson(successes, n)
            parse_failures = sum(bool(row["parse_failure"]) for row in rows)
            invalid = sum(int(row["num_invalid_actions"]) for row in rows)
            attempted_moves = sum(int(row["num_executed_actions"]) - int(bool(row["stop_emitted"]))
                                  for row in rows)
            record = {
                "task": task, "condition": condition, "successes": successes, "n": n,
                "success_rate": successes / n, "wilson95_low": low, "wilson95_high": high,
                "parse_failure_episodes": parse_failures, "parse_failure_rate": parse_failures / n,
                "invalid_actions": invalid, "attempted_moves": attempted_moves,
                "invalid_action_rate": invalid / attempted_moves if attempted_moves else float("nan"),
            }
            csv_rows.append(record)
            counts = Counter(row["termination_reason"] for row in rows)
            terminations.append({"task": task, "condition": condition, "counts": dict(counts)})
        original = task_rows["Original"]
        for condition in CONDITIONS:
            if condition != "Original":
                tests.append({"task": task, "condition": condition,
                              **mcnemar(original, task_rows[condition])})
        rows_by[task] = task_rows

    with (outdir / "results.jsonl").open("w") as stream:
        for row in all_results:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    with (outdir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    (outdir / "mcnemar.json").write_text(json.dumps(tests, indent=2) + "\n")
    (outdir / "termination_reasons.json").write_text(json.dumps(terminations, indent=2) + "\n")
    with (outdir / "world_model_pixel_mse.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["task", "episode_id", "imagined_frame_index",
                                                     "pixel_mse_0_255", "pixel_mse_0_1"])
        writer.writeheader()
        for task in tasks:
            for episode_id, row in sorted(rows_by[task]["Original"].items()):
                errors = row["pixel_mse"]
                if len(errors) != row["n_imagined"]:
                    raise RuntimeError(f"{episode_id}: MSE/frame count mismatch")
                for index, error in enumerate(errors, start=1):
                    writer.writerow({"task": task, "episode_id": episode_id,
                                     "imagined_frame_index": index, **error})

    lookup = {(row["task"], row["condition"]): row for row in csv_rows}
    def presentation_success(task: str, condition: str) -> int | None:
        if task not in tasks:
            return None
        return lookup[(task, condition)]["successes"]

    with (outdir / "paper_table.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["condition", "sokoban_success_pct",
                                                     "pacman_success_pct", "source"])
        writer.writeheader()
        for condition in CONDITIONS:
            writer.writerow({
                "condition": condition,
                "sokoban_success_pct": presentation_success("sokoban", condition),
                "pacman_success_pct": presentation_success("pacman", condition),
                "source": "replayed_baseline" if condition == "No-Imagine"
                          else "this_ablation",
            })

    latex = [r"% Requires \usepackage{booktabs}", r"\begin{table}[t]", r"\centering",
             r"\begin{tabular}{lcc}", r"\toprule",
             r"Condition & Sokoban 4-box hard & Pacman hard \\", r"\midrule"]
    for condition in CONDITIONS:
        cells = [f"{value}\\%" if value is not None else "--"
                 for task in ("sokoban", "pacman")
                 for value in [presentation_success(task, condition)]]
        latex.append(condition.replace("-", "--") + " & " + " & ".join(cells) + r" \\")
    latex += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Success rate on 100 paired episodes per task at $K=5$. All rates are computed from the supplied episode records. No-Imagine uses the replayed baseline; the other conditions use the current runs.}",
              r"\label{tab:agima-k5-image-ablation}", r"\end{table}"]
    (outdir / "table.tex").write_text("\n".join(latex) + "\n")

    report = ["# Imagined-frame ablation (K=5)", "",
              "`summary.csv` contains the counts computed from the supplied episode records.",
              "No-Imagine uses replayed archive records; Original, Blank, Stale and Oracle GT",
              "use the current run directories. `paper_table.csv` uses these same counts.", "",
              "## Measured Original versus archive", ""]
    for task in tasks:
        published = archived_stats(task, "cot")
        published_by_id = {
            f"{task}:{item['global_index']:07d}:{item['episode_seed']}":
            {"success": bool(item["stats"]["success"])} for item in published
        }
        original = rows_by[task]["Original"]
        if set(original) != set(published_by_id):
            raise RuntimeError(f"{task}: Original episode set differs from the staged baseline")
        alignment = mcnemar(original, published_by_id)
        report.append(
            f"- {task}: Original {sum(row['success'] for row in original.values())}/100 versus archive "
            f"{sum(row['success'] for row in published_by_id.values())}/100; "
            f"paired new-only/archive-only = ({alignment['b_original_only']}, "
            f"{alignment['c_other_only']}), exact p={alignment['p_exact_two_sided']:.4g}."
        )
    audits = []
    for task in tasks:
        for condition, run_name in RUN_NAMES.items():
            audit_path = outdir / f"{task}_{run_name}_first10_audit.json"
            checked = json.loads(audit_path.read_text())
            if len(checked) != 10 or not all(item["passed"] for item in checked):
                raise RuntimeError(f"{task} {condition}: first-ten trace audit incomplete")
            audits.extend(checked)
    report += [
        "",
        f"All {len(audits)} saved first-ten traces passed context-image and real-boundary pixel checks",
        f"({sum(item['imagined_frames'] for item in audits)} imagined frames; "
        f"{sum(item['real_boundaries'] for item in audits)} real boundaries).",
        "",
              "## Final success table", "",
              "Each task has N=100. All rates below are computed from the supplied records.", "",
              "| Condition | Sokoban 4-box hard | Pacman hard |",
              "|---|---:|---:|"
    ]
    for condition in CONDITIONS:
        report.append(f"| {condition} | " + " | ".join(
            f"{presentation_success(task, condition)}%" if task in tasks else "--"
            for task in ("sokoban", "pacman")) + " |")
    measured_original = ", ".join(
        f"{task} {lookup[(task, 'Original')]['successes']}/100" for task in tasks
    )
    report += ["", "## Measured run diagnostics", "",
               f"Newly measured Original success: {measured_original};",
               "they supply the Original side of the paired tests below. `summary.csv` keeps",
               "the supplied-record rates and underlying counts for every condition, including the",
               "replayed No-Imagine archive baseline.", "",
               "Invalid move rate uses non-STOP executed moves as denominator; a Pacman wall bump",
               "counts as invalid even when the ghost advances. Parse failure rate counts episodes",
               "with an imagined or committed action parse failure.", "",
               "| Task | Condition | Parse failures | Invalid moves |",
               "|---|---|---:|---:|"]
    for row in csv_rows:
        report.append(f"| {row['task']} | {row['condition']} | "
                      f"{100*row['parse_failure_rate']:.1f}% | {100*row['invalid_action_rate']:.1f}% |")
    report += ["", "## Paired McNemar exact tests", "",
               "The reference is the measured Original run.",
               "b = measured Original succeeds and comparator fails; c = comparator succeeds",
               "and measured Original fails.", "",
               "| Task | Comparator | b | c | p |", "|---|---|---:|---:|---:|"]
    for test in tests:
        report.append(f"| {test['task']} | {test['condition']} | {test['b_original_only']} | "
                      f"{test['c_other_only']} | {test['p_exact_two_sided']:.4g} |")
    report += ["", "## Termination reasons", ""]
    for item in terminations:
        report.append(f"- {item['task']} / {item['condition']}: " +
                      ", ".join(f"{key}={value}" for key, value in sorted(item["counts"].items())))
    report += ["", "## Original world-model pixel error", "",
               "Pixel MSE values below are read from the Original episode records on RGB [0,1].", ""]
    for task in tasks:
        errors = [x["pixel_mse_0_1"] for row in rows_by[task]["Original"].values()
                  for x in row["pixel_mse"]]
        report.append(f"- {task}: {len(errors)} paired frames, mean MSE={sum(errors)/len(errors):.6f}." if errors
                      else f"- {task}: no paired frames recorded.")
    report += ["", "Every frame's MSE is in `world_model_pixel_mse.csv` in episode order.",
               "", "## Measured condition differences", ""]
    for task in tasks:
        original = lookup[(task, "Original")]["successes"]
        n = EXPECTED[task]["n"]
        for condition in ("Blank", "Stale", "Oracle GT"):
            successes = lookup[(task, condition)]["successes"]
            difference = 100 * (successes - original) / n
            report.append(
                f"- {task} / {condition}: {successes}/{n} versus measured Original "
                f"{original}/{n}; difference {difference:+.1f} percentage points."
            )
    report.append("")
    (outdir / "report.md").write_text("\n".join(report))
    print(f"Wrote {len(all_results)} episode rows and paper outputs to {outdir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", choices=tuple(EXPECTED), default=list(EXPECTED))
    parser.add_argument("--outdir", type=Path, default=ROOT / "output")
    args = parser.parse_args()
    summarize(tuple(args.tasks), args.outdir)
