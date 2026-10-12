"""Write a four-task table from measured ablation summaries."""

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
TASKS = ("sokoban", "pacman", "maze", "pusht")
CONDITIONS = ("No-Imagine", "Original", "Blank", "Stale", "Oracle GT")


def _read(path: Path):
    with path.open(newline="") as stream:
        records = {}
        for row in csv.DictReader(stream):
            key = row["task"], row["condition"]
            if key in records:
                raise ValueError(f"{path}: duplicate task/condition {key}")
            if int(row["n"]) != 100 or not 0 <= int(row["successes"]) <= 100:
                raise ValueError(f"{path}: expected 100 episodes and a valid success count for {key}")
            records[key] = row
        return records


def main():
    sokoban_pacman = _read(OUT / "summary.csv")
    maze_pusht = _read(OUT / "maze_pusht_summary.csv")
    if any((task, condition) not in sokoban_pacman for task in TASKS[:2] for condition in CONDITIONS):
        raise RuntimeError("Sokoban/Pacman summary is incomplete")
    if any((task, condition) not in maze_pusht for task in TASKS[2:] for condition in CONDITIONS):
        raise RuntimeError("Maze/PushT summary is incomplete")

    values = {}
    for condition in CONDITIONS:
        values[condition] = tuple(
            int((sokoban_pacman if task in TASKS[:2] else maze_pusht)[(task, condition)]["successes"])
            for task in TASKS
        )

    with (OUT / "paper_table.csv").open("w") as stream:
        stream.write("condition,sokoban_success_pct,pacman_success_pct,maze_success_pct,"
                     "pusht_success_pct,source\n")
        for condition in CONDITIONS:
            stream.write(",".join([condition, *(str(value) for value in values[condition]),
                                   "replayed_baseline" if condition == "No-Imagine"
                                   else "this_ablation"]) + "\n")

    rows = ["% Requires \\usepackage{booktabs}",
            "\\begin{table}[t]", "\\centering", "\\small", "\\begin{tabular}{lcccc}",
            "\\toprule", "Condition & Sokoban & Pacman & Maze2D & PushT \\\\",
            "\\midrule"]
    for condition in CONDITIONS:
        name = "No--Imagine" if condition == "No-Imagine" else condition
        cells = " & ".join(f"{value}\\%" for value in values[condition])
        rows.append(f"{name} & {cells} \\\\")
    rows.extend(["\\bottomrule", "\\end{tabular}",
                 "\\caption{Success rate on 100 paired episodes per task at $K=5$ "
                 "(Sokoban 4-box hard, Pacman hard, Maze2D hard, PushT in-distribution). "
                 "All rates are computed from the supplied episode records. No-Imagine "
                 "uses the replayed baseline; the other conditions use the current runs. "
                 "Paired tests use each measured Original run.}",
                 "\\label{tab:agima-k5-image-ablation}", "\\end{table}", ""])
    (OUT / "table.tex").write_text("\n".join(rows))
    print("Updated", OUT / "paper_table.csv", "and", OUT / "table.tex")


if __name__ == "__main__":
    main()
