#!/usr/bin/env python3
"""Recount unique evaluation episodes and report incomplete or inconsistent results.

Success follows the existing current and historical episode schemas. Invalid
records and conflicting duplicates are reported explicitly and make the command
fail. Use --expected-episodes to check the intended benchmark denominator.
"""

import argparse
import glob
import json
import math
from pathlib import Path
import sys


def episode_success(stats: dict) -> bool:
    """Read current success flags, then historical history/reward fallbacks."""
    st = stats.get("stats") or {}
    success = st.get("success")
    if success is not None:
        return bool(success)
    for step in reversed(stats.get("history") or []):
        info = step.get("info") or {}
        if info.get("is_success") is not None:
            return bool(info.get("is_success"))
        if info.get("success") is not None:
            return bool(info.get("success"))
    return float(st.get("reward") or 0.0) > 0.0


def validated_success(record):
    if not isinstance(record, dict):
        raise ValueError("episode record must be a JSON object")
    stats = record.get("stats")
    history = record.get("history")
    stats = {} if stats is None else stats
    history = [] if history is None else history
    if not isinstance(stats, dict) or not isinstance(history, list):
        raise ValueError("stats must be an object and history must be a list")
    flags = [stats.get("success")]
    for step in history:
        if not isinstance(step, dict) or (step.get("info") is not None and not isinstance(step["info"], dict)):
            raise ValueError("history steps and their info fields must be objects")
        info = step.get("info") or {}
        flags.extend((info.get("success"), info.get("is_success")))
    if any(value is not None and not isinstance(value, bool) for value in flags):
        raise ValueError("success flags must be JSON booleans")
    if not any(value is not None for value in flags) and stats.get("reward") is None:
        raise ValueError("episode has no success flag or historical reward result")
    if not math.isfinite(float(stats.get("reward") or 0.0)):
        raise ValueError("reward must be finite")
    return episode_success(record)


def read_json(path):
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"non-finite JSON value: {value}")

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            reject_constant(value)
        return number

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_keys, parse_constant=reject_constant,
                      parse_float=finite_float)


def retally_run(run_dir: str, expected_episodes=None) -> dict:
    root = Path(run_dir)
    by_index = {}
    issues = []
    duplicates = 0
    nested_runs = sorted(path.parent for path in root.rglob("run-contract.json") if path.parent != root)
    if nested_runs:
        issues.append("directory contains nested runs; pass each run directory or a quoted glob separately")
    for path in sorted(root.rglob("episode_stats.json")):
        try:
            record = read_json(path)
            success = validated_success(record)
            index = record.get("global_index")
            if index is not None and (type(index) is not int or index < 0):
                raise ValueError("global_index must be a nonnegative integer")
            key = ("index", index) if index is not None else ("path", str(path))
            identity = {field: record.get(field) for field in ("env_id", "seed", "episode_seed")}
            value = (success, identity)
            if key in by_index:
                duplicates += 1
                if by_index[key] != value:
                    raise ValueError(f"conflicting records for global_index {index}")
            else:
                by_index[key] = value
        except (OSError, ValueError, TypeError) as exc:
            issues.append(f"{path.relative_to(root)}: {exc}")
    total = len(by_index)
    successes = sum(value[0] for value in by_index.values())
    if not total:
        issues.append("no valid episode records found")
    if expected_episodes is not None and total != expected_episodes:
        issues.append(f"expected {expected_episodes} unique episodes, found {total}")
    result = {
        "run": str(root), "unique_episodes": total, "successes": successes,
        # A partial/conflicting tally must never look like a certified success rate.
        "success_rate": successes / total if total and not issues else None,
        "duplicate_records": duplicates, "issues": issues,
    }
    summary_path = root / "summary.json"
    if summary_path.exists():
        try:
            summary = read_json(summary_path)
            if not isinstance(summary, dict):
                raise ValueError("summary must be a JSON object")
            for field in ("episodes", "completed", "skipped", "errors", "successes"):
                if field in summary and (type(summary[field]) is not int or summary[field] < 0):
                    raise ValueError(f"summary {field} must be a nonnegative integer")
            result["summary_json"] = {key: summary.get(key) for key in
                                      ("episodes", "completed", "skipped", "errors", "status", "successes", "success_rate")}
            if summary.get("errors"):
                issues.append(f"summary.json reports {summary['errors']} episode errors")
            planned = summary.get("episodes")
            if expected_episodes is None and type(planned) is int and planned != total:
                issues.append(f"summary.json planned {planned} episodes, found {total} unique records")
            if issues:
                result["success_rate"] = None
            result["summary_disagrees"] = (summary.get("successes") != successes or
                (result["success_rate"] is not None and
                 not math.isclose(float(summary.get("success_rate") or 0.0), result["success_rate"], abs_tol=1e-6)))
        except (OSError, ValueError, TypeError) as exc:
            result["summary_json"] = "unreadable"
            issues.append(f"summary.json is invalid: {exc}")
            result["success_rate"] = None
    return result


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="Run directories or quoted glob patterns, e.g. 'runs/sokoban-*'.")
    parser.add_argument("--expected-episodes", type=positive_int, help="Required unique episode count for each run.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable tallies and diagnostics.")
    args = parser.parse_args(argv)
    run_dirs = []
    for pattern in args.runs:
        matches = [Path(path).resolve() for path in sorted(glob.glob(pattern)) if Path(path).is_dir()]
        if not matches:
            parser.error(f"no run directories matched: {pattern}")
        run_dirs.extend(matches)
    results = [retally_run(str(path), args.expected_episodes) for path in dict.fromkeys(run_dirs)]
    if args.json:
        print(json.dumps(results, indent=2, allow_nan=False))
    else:
        print(f"{'successes/unique':>18}  {'rate':>10}  run")
        for result in results:
            rate = f"{100 * result['success_rate']:.1f}%" if result["success_rate"] is not None else "INVALID"
            print(f"{result['successes']:>8}/{result['unique_episodes']:<8} {rate:>10}  {result['run']}")
            if result.get("summary_disagrees"):
                print("  summary.json disagrees with the unique-episode tally.")
    for result in results:
        for issue in result["issues"]:
            print(f"{result['run']}: {issue}", file=sys.stderr)
    return 1 if any(result["issues"] for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
