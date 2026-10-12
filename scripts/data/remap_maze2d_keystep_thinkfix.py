#!/usr/bin/env python3
# Modified for Agima: portable paths and explicit dataset arguments.
"""F5 fix: maze2d CoT key-step assistant turn is missing its opening <think> (it starts with
bare prose and only emits a closing </think>). This remaps the EXISTING lastbranch_v3 CoT data
(easy+hard, training+testing) to prepend "<think>" to the FIRST text segment of the key-step
turn (the single assistant turn that carries the imagined-branch images). Trivial/stop turns,
user turns, __bagel_order_key, and every other field are byte-identical. Multiprocess over
shards; writes NEW repos (originals untouched). After this, every key step is balanced
<think>...</think>, matching the trivial/stop steps."""
import argparse, json, gzip, glob, os, shutil
from multiprocessing import Pool

SRC_TMPL = "{hf}/maze2d_{diff}_native256_stopreq_stepmsg_fixedstart_cot_lastbranch_ordered_20260606_lastbranch_v3"
OUT_TMPL = "{hf}/maze2d_{diff}_native256_stopreq_stepmsg_fixedstart_cot_lastbranch_ordered_{tag}"

def fix_row(row):
    """Prepend <think> to the first text segment of each image-carrying (key) assistant turn
    that lacks it. Idempotent. Returns (row, fixed_turns)."""
    msgs = row.get("messages")
    fixed = 0
    if isinstance(msgs, list):
        for m in msgs:
            if m.get("role") != "assistant":
                continue
            c = m.get("content")
            if not isinstance(c, list):
                continue
            if not any(isinstance(it, dict) and it.get("type") == "image" for it in c):
                continue  # only the key (branch-image) turn; trivial/stop are text-only & already correct
            for it in c:
                if isinstance(it, dict) and it.get("type") == "text":
                    t = it.get("text", "")
                    if not t.lstrip().startswith("<think>"):
                        it["text"] = "<think>" + t
                        fixed += 1
                    break  # only the first text segment
    return row, fixed

def process_shard(args_tuple):
    in_path, out_path = args_tuple
    n = key_turns = balanced = 0
    tmp = out_path + ".tmp"
    with gzip.open(in_path, "rt") as fin, gzip.open(tmp, "wt") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            row, fixed = fix_row(row)
            n += 1
            key_turns += fixed
            # post-check: the image turn(s) now balanced
            for m in row.get("messages", []):
                if m.get("role") == "assistant" and isinstance(m.get("content"), list) and \
                   any(isinstance(it, dict) and it.get("type") == "image" for it in m["content"]):
                    txt = "".join(it.get("text", "") for it in m["content"] if isinstance(it, dict) and it.get("type") == "text")
                    if txt.count("<think>") == txt.count("</think>") == 1:
                        balanced += 1
            fout.write(json.dumps(row) + "\n")
    os.replace(tmp, out_path)
    return (in_path, n, key_turns, balanced)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="20260611_lastbranch_v3_thinkfix")
    ap.add_argument("--workers", type=int, default=min(96, os.cpu_count() or 8))
    ap.add_argument("--input-base", required=True, help="Parent directory of the source datasets")
    ap.add_argument("--output-base", required=True, help="Parent directory for the corrected datasets")
    args = ap.parse_args()

    tasks = []
    out_roots = {}
    for diff in ("easy", "hard"):
        src = SRC_TMPL.format(hf=args.input_base, diff=diff)
        out = OUT_TMPL.format(hf=args.output_base, diff=diff, tag=args.tag)
        out_roots[diff] = out
        for sub in ("training", "testing"):
            os.makedirs(f"{out}/{sub}", exist_ok=True)
            for sh in sorted(glob.glob(f"{src}/{sub}/*.jsonl.gz")):
                tasks.append((sh, f"{out}/{sub}/{os.path.basename(sh)}"))
        # copy metadata as provenance
        if os.path.isdir(f"{src}/metadata"):
            shutil.copytree(f"{src}/metadata", f"{out}/metadata", dirs_exist_ok=True)
        # note file
        with open(f"{out}/THINKFIX_NOTE.txt", "w") as f:
            f.write(f"F5 fix of {src}\nOnly change: prepended '<think>' to the key-step assistant turn's first "
                    f"text segment so it is balanced <think>...</think> like trivial/stop steps. "
                    f"All other content (trivial/stop turns, user turns, branch images, __bagel_order_key, "
                    f"trajectory_steps, init_state, metadata) is byte-identical.\n")

    print(f"cores={os.cpu_count()} workers={args.workers} shards={len(tasks)}", flush=True)
    with Pool(min(args.workers, len(tasks))) as p:
        results = p.map(process_shard, tasks)

    tot_n = tot_key = tot_bal = 0
    for path, n, kt, bal in sorted(results):
        tot_n += n; tot_key += kt; tot_bal += bal
        print(f"  {os.path.basename(os.path.dirname(os.path.dirname(path)))}/{os.path.basename(path)}: "
              f"rows={n} keysteps_fixed={kt} keysteps_balanced_after={bal}", flush=True)
    print(f"\nDONE rows={tot_n} keysteps_fixed={tot_key} keysteps_balanced_after={tot_bal}", flush=True)
    print(f"-> {'OK (every row fixed + balanced)' if tot_key == tot_bal == tot_n else 'CHECK: counts differ'}", flush=True)
    for diff, out in out_roots.items():
        print(f"   {diff} -> {out}", flush=True)

if __name__ == "__main__":
    main()
