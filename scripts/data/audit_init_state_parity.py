#!/usr/bin/env python3
# Modified for Agima: portable paths and explicit dataset arguments.
"""Cross-interval / cross-variant init-state parity audit (sampled).
For every task (sokoban/pusht/maze2d) and every dataset variant (cot K1/3/5/10/inf, noncot K*,
branch), sample the head of every shard and verify BY __bagel_order_key that the underlying
episodes are identical: same seed/episode_seed, same canonical init_state, same frame0 bytes.
Forward (sokoban) is included as an EXPECTED-DIFFERENT control (different generated levels).
"""
import argparse, gzip, glob, json, hashlib, os, re, time
from multiprocessing import Pool

N_PER_SHARD = 300

RE_ORD = re.compile(r'"__bagel_order_key":\s*"?([^",}]+)"?')
RE_SEED = re.compile(r'"seed":\s*(\d+)')
RE_EPS = re.compile(r'"episode_seed":\s*(\d+)')
RE_IMG = re.compile(r'"image":\s*"([A-Za-z0-9+/=]{200,})"')

def variants(task):
    if task == "sokoban":
        d = {f"cot_k{k}": f"sokoban_easy_v8_cot_chunk_k{k}_world_model_20260622_perseg" for k in (1,3,5,10)}
        d.update({f"noncot_k{k}": f"sokoban_easy_v8_noncot_chunk_k{k}_world_model_20260622_perseg" for k in (1,3,5,10)})
        d["cot_kinf"] = "sokoban_easy_v8_cot_chunk_kinf_world_model_20260707_perseg"
        d["branch_kinf"] = "sokoban_easy_v8_cot_chunk_kinf_branch_20260709_perseg"
        d["forward_EXPECT_DIFF"] = "sokoban_easy_v8_cot_chunk_kinf_forward_20260710_fwdpure"
    elif task == "pusht":
        d = {f"cot_k{k}": f"pusht_96_norm4_cot_chunk_k{k}_20260622_perseg" for k in (1,3,5,10)}
        d.update({f"noncot_k{k}": f"pusht_96_norm4_noncot_chunk_k{k}_20260622_perseg" for k in (1,3,5,10)})
        d["cot_kinf"] = "pusht_96_norm4_cot_chunk_kinf_20260707_perseg"
        d["branchcmp_kinf"] = "pusht_96_norm4_cot_chunk_kinf_branchcmp_20260709_perseg"
    else:
        d = {f"cot_k{k}": f"maze2d_easy_native256_cot_chunk_k{k}_20260707_perseg" for k in (1,3,5,10,"inf")}
        d.update({f"noncot_k{k}": f"maze2d_easy_native256_noncot_chunk_k{k}_20260707_perseg" for k in (1,3,5,10,"inf")})
    return d

def extract_init(line):
    i = line.find('"init_state"')
    if i < 0: return None
    j = line.find("{", i)
    if j < 0: return None
    depth = 0; in_str = False; esc = False
    for k in range(j, len(line)):
        c = line[k]
        if in_str:
            if esc: esc = False
            elif c == "\\": esc = True
            elif c == '"': in_str = False
        else:
            if c == '"': in_str = True
            elif c == "{": depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(line[j:k+1])
                        return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",",":")).encode()).hexdigest()[:16]
                    except Exception:
                        return "PARSE_ERR"
    return None

def scan(args):
    task, var, shard = args
    rows = []
    try:
        with gzip.open(shard, "rt") as f:
            for i, line in enumerate(f):
                if i >= N_PER_SHARD: break
                ordk = RE_ORD.search(line); seed = RE_SEED.search(line); eps = RE_EPS.search(line)
                img = RE_IMG.search(line)
                rows.append((ordk.group(1) if ordk else f"NOORD:{os.path.basename(shard)}:{i}",
                             seed.group(1) if seed else None,
                             eps.group(1) if eps else None,
                             extract_init(line),
                             hashlib.sha256(img.group(1).encode()).hexdigest()[:16] if img else None))
    except Exception as e:
        rows.append((f"SCAN_ERR:{e}", None, None, None, None))
    return task, var, rows

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="Parent directory of the dataset variants")
    args = parser.parse_args()
    H = args.data_root
    t0 = time.time()
    jobs = []
    for task in ("sokoban", "pusht", "maze2d"):
        for var, name in variants(task).items():
            shards = sorted(glob.glob(f"{H}/{name}/training/*.jsonl.gz"))
            if not shards:
                print(f"!! NO SHARDS for {task}/{var} ({name})", flush=True)
            for sh in shards:
                jobs.append((task, var, sh))
    print(f"scanning {len(jobs)} shard-heads x {N_PER_SHARD} rows on {os.cpu_count()} cores", flush=True)
    data = {}
    with Pool(min(96, os.cpu_count() or 8)) as p:
        for task, var, rows in p.imap_unordered(scan, jobs):
            data.setdefault(task, {}).setdefault(var, {})
            for ordk, seed, eps, ish, fsh in rows:
                data[task][var][ordk] = (seed, eps, ish, fsh)
    print(f"[{time.time()-t0:.0f}s] scan done; comparing...", flush=True)

    overall_bad = 0
    for task, vars_ in data.items():
        ref_name = "cot_k1"
        ref = vars_.get(ref_name, {})
        print(f"\n===== {task} (reference = {ref_name}, {len(ref)} sampled keys) =====", flush=True)
        for var, rows in sorted(vars_.items()):
            if var == ref_name: continue
            common = set(rows) & set(ref)
            mism_seed = mism_init = mism_f0 = init_avail = 0
            for k in common:
                s1, e1, i1, f1 = ref[k]; s2, e2, i2, f2 = rows[k]
                if (s1, e1) != (s2, e2): mism_seed += 1
                if i1 and i2:
                    init_avail += 1
                    if i1 != i2: mism_init += 1
                if f1 and f2 and f1 != f2: mism_f0 += 1
            expect_diff = "EXPECT_DIFF" in var
            verdict = "OK" if (mism_seed == mism_init == mism_f0 == 0 and len(common) > 0) else ("EXPECTED (control)" if expect_diff else "MISMATCH!!")
            if verdict == "MISMATCH!!": overall_bad += 1
            print(f"  {var:22s} sampled={len(rows):5d} common-with-ref={len(common):5d} "
                  f"| seed/eps-mism={mism_seed} init-mism={mism_init}/{init_avail} frame0-mism={mism_f0}  -> {verdict}", flush=True)
    print(f"\nRESULT: {'ALL CLEAN — intervals & baselines share identical episodes' if overall_bad==0 else f'{overall_bad} variant(s) MISMATCHED — ablation contaminated, inspect!'}", flush=True)
