"""Standard-library helpers for portable, single-node training and evaluation."""

import argparse
import datetime
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
INTERRUPT_GRACE_SECONDS = (30, 10, 5)
RECORDED_ENV_KEYS = {
    "PYTHONPATH", "VISGYM_ROOT", "SDL_VIDEODRIVER", "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER", "OMP_NUM_THREADS", "PYTHONHASHSEED", "WANDB_MODE",
    "CUBLAS_WORKSPACE_CONFIG", "NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
    "BAGEL_CHUNK_K", "BAGEL_BASE_PATH", "BAGEL_NO_REGROUND", "BAGEL_COT_FRAME_MODE",
    "BAGEL_COT_CFG_TEXT", "BAGEL_COT_CFG_IMG", "BAGEL_COT_NUM_TIMESTEPS",
    "BAGEL_COT_MAX_TEXT_TOKENS", "BAGEL_ENV_FEEDBACK_LABEL", "BAGEL_PUSHT_INJECT_GT_FRAME",
    "BAGEL_PUSHT_BICUBIC_ROUNDTRIP", "BAGEL_WAIT_ALL_RANK_SUMMARIES", "BAGEL_DECODER",
}


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def absolute(value):
    return Path(value).expanduser().resolve()


def require_file(path):
    if not path.is_file():
        raise ValueError(f"Required file not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Required file is empty: {path}")


def reject_nonfinite_json(value):
    raise ValueError(f"non-finite JSON number: {value}")


def finite_json_float(value):
    number = float(value)
    if not math.isfinite(number):
        reject_nonfinite_json(value)
    return number


def read_json_object(path):
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_keys,
                           parse_constant=reject_nonfinite_json, parse_float=finite_json_float)
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        return value
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def require_weight_file(path):
    require_file(path)
    with path.open("rb") as stream:
        if stream.read(128).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError(f"Weight file is a Git LFS pointer, not downloaded weights: {path}")


def validate_base(path, training=False):
    files = []
    for name in ("llm_config.json", "vit_config.json", "ae.safetensors",
                 "tokenizer_config.json", "vocab.json", "merges.txt"):
        require_file(path / name)
        if name.endswith(".json"):
            read_json_object(path / name)
        files.append(path / name)
    require_weight_file(path / "ae.safetensors")
    if training:
        # The preserved training loader reads this single file, unlike the eval loader.
        require_weight_file(path / "ema.safetensors")
        files.append(path / "ema.safetensors")
    # The tokenizer may prefer this optional file when it is present.
    for name in ("tokenizer.json", "special_tokens_map.json", "added_tokens.json"):
        if (path / name).exists():
            require_file(path / name)
            read_json_object(path / name)
            files.append(path / name)
    return files


def validate_checkpoint(path):
    if (path / "ema.safetensors").is_file():
        require_weight_file(path / "ema.safetensors")
        return [path / "ema.safetensors"]
    index = path / "ema.safetensors.index.json"
    require_file(index)
    try:
        weight_map = read_json_object(index)["weight_map"]
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("weight_map must be a nonempty object")
        if any(not key or not isinstance(name, str) or not name
               for key, name in weight_map.items()):
            raise ValueError("weight_map must map nonempty tensor names to nonempty filenames")
        files = [index]
        for name in sorted(set(weight_map.values())):
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts or "\\" in name:
                raise ValueError("checkpoint index references a file outside its directory")
            if relative.suffix != ".safetensors":
                raise ValueError("checkpoint index must reference .safetensors shards")
            # Safe relative names may be HF snapshot symlinks into the local blob cache.
            shard = path / relative
            require_weight_file(shard)
            files.append(shard)
        return files
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid checkpoint index {index}: {exc}") from exc


def validate_episodes(path, directory_only=False):
    if not directory_only and path.is_file() and str(path).endswith((".jsonl", ".jsonl.gz")):
        files = [path]
    elif path.is_dir():
        # Training's JsonLIterableDataset uses os.listdir; evaluation is recursive.
        find = path.glob if directory_only else path.rglob
        files = sorted(p for pattern in ("*.jsonl", "*.jsonl.gz") for p in find(pattern) if p.is_file())
    else:
        files = []
    if not files:
        hint = " Training shards must be direct children of --data-dir." if directory_only else ""
        raise ValueError(f"No JSONL episodes found at: {path}.{hint}")
    for file in files:
        require_file(file)
    return files


def validate_rows(files, *, task=None, minimum=1, limit=None):
    """Check the rows that a short run will consume, without decoding images or importing loaders."""
    checked = 0
    for path in files:
        opener = gzip.open if str(path).endswith(".gz") else open
        try:
            with opener(path, "rt", encoding="utf-8") as stream:
                for number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    location = f"{path}:{number}"
                    try:
                        row = json.loads(line, parse_constant=reject_nonfinite_json,
                                         parse_float=finite_json_float)
                    except ValueError as exc:
                        raise ValueError(f"Invalid JSONL row at {location}: {exc}") from exc
                    if not isinstance(row, dict):
                        raise ValueError(f"Expected an episode object at {location}")
                    if task is None:
                        messages = row.get("messages")
                        if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
                            raise ValueError(f"Training requires prepared messages rows at {location}; remap recorded episodes first")
                        if not all(any(m.get("role") == role and isinstance(m.get("content"), list)
                                       and m["content"] for m in messages) for role in ("user", "assistant")):
                            raise ValueError(f"Training row needs nonempty user and assistant content lists at {location}")
                    else:
                        prefix = {"sokoban": "sokoban/", "pusht": "pusht/", "maze2d": "maze_2d/"}[task]
                        if not str(row.get("env_id", "")).startswith(prefix):
                            raise ValueError(f"Expected a {task} evaluation episode with env_id starting {prefix!r} at {location}")
                        if not isinstance(row.get("init_state"), dict) or not row["init_state"]:
                            raise ValueError(f"Evaluation requires a nonempty init_state object at {location}")
                        if row.get("env_kwargs") is not None and not isinstance(row["env_kwargs"], dict):
                            raise ValueError(f"env_kwargs must be an object at {location}")
                    checked += 1
                    if checked >= (limit if limit is not None else minimum):
                        return checked
        except (OSError, EOFError, UnicodeError) as exc:
            raise ValueError(f"Cannot read JSONL data {path}: {exc}") from exc
    if checked < minimum:
        raise ValueError(f"Found only {checked} data rows; need at least {minimum} (one per training GPU, or one evaluation episode)")
    return checked


def validate_output_location(output, inputs):
    output = output.resolve()
    for path in inputs:
        path = path.resolve()
        if output == path or output in path.parents or (path.is_dir() and path in output.parents):
            raise ValueError(f"--output-dir must not overlap an input asset path: {path}")


def file_fingerprints(files, *, hash_all=False):
    """Hash small metadata and selected episode pools; record size/mtime for large weights/data."""
    records = []
    for path in sorted(set(files)):
        stat = path.stat()
        record = {"path": str(path.absolute()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if hash_all or (stat.st_size <= 1024 * 1024 and path.suffix != ".safetensors"):
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            record["sha256"] = digest.hexdigest()
        records.append(record)
    return records


def source_provenance():
    """Record Git identity and hash runtime sources; documentation-only edits do not invalidate resume."""
    result = {"git_commit": None, "git_dirty": None}
    try:
        top = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, timeout=10)
        is_checkout = top.returncode == 0 and Path(top.stdout.strip()).resolve() == ROOT.resolve()
    except (OSError, subprocess.TimeoutExpired):
        is_checkout = False
    for key, args in (("git_commit", ["rev-parse", "HEAD"]),
                      ("git_dirty", ["status", "--porcelain", "--untracked-files=normal"])):
        if not is_checkout:
            break  # An archive inside another checkout must not claim its Git identity.
        try:
            command = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True,
                                     text=True, timeout=10)
            if command.returncode == 0:
                result[key] = bool(command.stdout.strip()) if key == "git_dirty" else command.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass  # Source archives need not contain Git metadata.
    paths = [ROOT / "inferencer.py", ROOT / "requirements.txt"]
    for directory in ("scripts", "training", "evaluation", "data", "modeling", "gymnasium"):
        paths.extend((ROOT / directory).rglob("*.py"))
    for path in (ROOT / "gymnasium").rglob("*"):
        if path.is_file() and path.suffix.lower() in {".png", ".gif", ".jpg", ".jpeg", ".ttf", ".otf"}:
            paths.append(path)
    for directory, pattern in (("configs", "*.yaml"), ("requirements", "*.txt")):
        paths.extend((ROOT / directory).rglob(pattern))
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        digest.update(path.relative_to(ROOT).as_posix().encode() + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    result["runtime_source_sha256"] = digest.hexdigest()
    return result


def validate_run_contract(output, contract):
    if not output.exists():
        return
    existing = [path for path in output.iterdir() if path.name != ".launch.lock"]
    if not existing:
        return
    path = output / "run-contract.json"
    if not path.is_file():
        raise ValueError("Evaluation output is nonempty but has no run-contract.json. Use a new --output-dir; legacy results cannot be resumed safely.")
    previous = read_json_object(path)
    if previous != contract:
        changed = sorted(key for key in set(previous) | set(contract) if previous.get(key) != contract.get(key))
        raise ValueError("Evaluation output belongs to a different run (changed: " + ", ".join(changed) +
                         "). Use a new --output-dir; --overwrite only reruns the same experiment.")


def write_json_atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def runtime_environment(settings=None):
    # Old experiment knobs must not silently change a documented recipe.
    env = {key: value for key, value in os.environ.items() if not key.startswith("BAGEL_")}
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["VISGYM_ROOT"] = str(ROOT)
    env.setdefault("SDL_VIDEODRIVER", "dummy")
    env.update(settings or {})
    return env


def torchrun(gpus, runner, module=False):
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
               f"--nproc_per_node={gpus}"]
    command.extend(("-m", runner) if module else (str(ROOT / runner),))
    return command


def validate_runtime(gpus, env):
    """Check the selected interpreter and visible GPUs before creating run artifacts."""
    if importlib.util.find_spec("torch") is None:
        raise ValueError(
            f"PyTorch is not installed for {sys.executable}. "
            "Activate the CUDA environment described in the installation instructions."
        )
    probe = (
        "import json, sys, torch; from torch.distributed import run; "
        "print(json.dumps({'python': sys.version.split()[0], 'torch': torch.__version__, "
        "'cuda_available': torch.cuda.is_available(), "
        "'visible_gpus': torch.cuda.device_count()}))"
    )
    try:
        result = subprocess.run([sys.executable, "-c", probe], cwd=ROOT, env=env,
                                capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired as exc:
        raise ValueError("PyTorch CUDA preflight timed out after 60 seconds; no run files were written") from exc
    if result.returncode:
        detail = (result.stderr.strip() or result.stdout.strip()).splitlines()
        raise ValueError("PyTorch CUDA preflight failed: " + (detail[-1] if detail else "interpreter exited without diagnostics"))
    try:
        runtime = json.loads(result.stdout.strip().splitlines()[-1])
        cuda_available = runtime["cuda_available"]
        visible_gpus = int(runtime["visible_gpus"])
    except (ValueError, IndexError, KeyError, TypeError) as exc:
        raise ValueError("PyTorch CUDA preflight returned an invalid result; no run files were written") from exc
    if not cuda_available or visible_gpus < gpus:
        raise ValueError(
            f"Requested {gpus} CUDA GPU(s), but {visible_gpus if cuda_available else 0} are available. "
            "Check CUDA_VISIBLE_DEVICES and your CUDA installation, or use --dry-run."
        )
    return runtime


def stop_after_interrupt(process):
    """Give torchrun a bounded opportunity to stop its independently sessioned workers."""
    interrupt_grace, terminate_grace, kill_grace = INTERRUPT_GRACE_SECONDS
    for requested_signal, grace in ((signal.SIGINT, interrupt_grace),
                                    (signal.SIGTERM, terminate_grace)):
        if process.poll() is not None:
            return
        try:
            process.send_signal(requested_signal)
            process.wait(timeout=grace)
            return
        except ProcessLookupError:
            return
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            print(f"torchrun did not finish cleanup after {requested_signal.name}; escalating.",
                  file=sys.stderr, flush=True)
    if process.poll() is None:
        print("Killing unresponsive torchrun; its independent workers may still be running.",
              file=sys.stderr, flush=True)
        try:
            process.kill()
            process.wait(timeout=kill_grace)
        except (ProcessLookupError, subprocess.TimeoutExpired, KeyboardInterrupt):
            print("Could not confirm torchrun and worker termination.", file=sys.stderr, flush=True)


def run_command(command, env):
    # A separate session lets this launcher forward terminal Ctrl-C exactly once.
    # Avoid subprocess.call / a Popen context manager: their interrupt cleanup can
    # kill torchrun before it has stopped workers, or perform an unbounded wait.
    process = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True)
    try:
        return process.wait()
    except KeyboardInterrupt:
        print("Interrupted: asking torchrun to stop its workers (up to 45 seconds).",
              file=sys.stderr, flush=True)
        try:
            stop_after_interrupt(process)
        except BaseException as exc:
            print(f"Worker cleanup could not be confirmed: {type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)
        raise


def launch(command, env, output, dry_run, metadata, *, gpus, assets=None, evaluation=False):
    visible_env = {key: value for key, value in env.items() if key in RECORDED_ENV_KEYS}
    print(f"Working directory: {ROOT}", flush=True)
    print("Environment: " + json.dumps(visible_env, sort_keys=True), flush=True)
    print(shlex.join(command), flush=True)
    if dry_run:
        print("Dry run: no files written; assets, dependencies and GPUs were not checked.")
        return 0
    if output.exists() and not output.is_dir():
        raise ValueError(f"--output-dir must be a directory, not an existing file: {output}")
    runtime = validate_runtime(gpus, env)
    source = source_provenance()
    contract = {
        "version": 1, "command": [arg for arg in command if arg != "--overwrite"],
        "environment": visible_env, "parameters": metadata, "assets": assets or {},
        "runtime_source_sha256": source["runtime_source_sha256"],
        "runtime": {key: runtime.get(key) for key in ("python", "torch")},
    }
    if evaluation:
        validate_run_contract(output, contract)
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".launch.lock"
    try:
        handle = lock.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise ValueError(f"Output is locked by another launcher: {lock}. If a prior launcher crashed, verify its job has stopped before removing this lock.") from exc
    keep_lock = False
    try:
        with handle:
            json.dump({"pid": os.getpid()}, handle)
        if evaluation:
            validate_run_contract(output, contract)  # Check again after acquiring the exclusive lock.
            write_json_atomic(output / "run-contract.json", contract)
            (output / "summary.json").unlink(missing_ok=True)
            # Remove previous summaries; workers also verify the current job ID
            # when waiting and aggregating, including in direct research runs.
            for summary in (output / "_rank_summaries").glob("rank_*.json"):
                summary.unlink()
        elif any(path.name != ".launch.lock" for path in output.iterdir()):
            # A second cold-start launcher may have passed its CLI check before
            # another invocation created artifacts and released this same lock.
            raise ValueError("Training output is no longer empty; choose a new --output-dir for this cold-start recipe")
        stamp = datetime.datetime.now(datetime.timezone.utc)
        path = output / f"launch-{stamp.strftime('%Y%m%dT%H%M%S%fZ')}.json"
        manifest = dict(metadata, command=command, environment=visible_env, cwd=str(ROOT),
                        runtime=runtime, source=source, assets=assets or {}, status="running",
                        started_at=stamp.isoformat(), returncode=None)
        write_json_atomic(path, manifest)
        try:
            code = run_command(command, env)
            # A signal can kill torchrun before it reaps its independently
            # sessioned workers. wait() returns a negative code, not an exception.
            keep_lock = code < 0
            status = "interrupted" if keep_lock else ("succeeded" if code == 0 else "failed")
            manifest.update(returncode=code, status=status)
            return code
        except KeyboardInterrupt:
            # A torchrun exit code does not prove all independently sessioned
            # workers stopped. Retain exclusion even after graceful interruption.
            keep_lock = True
            manifest["status"] = "interrupted"
            raise
        except BaseException:
            keep_lock = True
            manifest["status"] = "interrupted"
            raise
        finally:
            manifest["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            write_json_atomic(path, manifest)
    finally:
        if keep_lock:
            print(f"Output lock retained: {lock}. Confirm the previous torchrun job and all workers "
                  "have stopped before removing this lock and resuming.", file=sys.stderr, flush=True)
        else:
            lock.unlink()
