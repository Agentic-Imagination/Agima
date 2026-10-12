"""Two-rank CPU checks for loss/forward participation; no BAGEL model imports."""

import ast
import datetime
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import time
import traceback
import unittest


ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ("cot", None),
    ("noncot", None),
    ("mse_only", None),
    ("mixed_mse", "MSE targets"),
    ("mixed_ce", "CE targets"),
    ("mixed_vit", "ViT inputs"),
    ("empty_ce_one", "Empty CE"),
    ("empty_ce_all", "Empty CE"),
    ("no_targets", "no active CE or MSE"),
)


def distributed_worker(rank, rendezvous, result_dir):
    import torch
    import torch.distributed as dist
    from training.distributed_checks import check_batch_participation

    results = {}
    try:
        dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                                timeout=datetime.timedelta(seconds=15))
        for case, expected_error in CASES:
            data = {"ce_loss_indexes": torch.arange(2 + rank),
                    "mse_loss_indexes": torch.arange(1 + rank),
                    "vit_token_seqlens": torch.tensor([2]),
                    "packed_vit_tokens": torch.ones(2, 3)}
            if case == "noncot":
                data["mse_loss_indexes"] = torch.empty(0) if rank == 0 else None
            elif case == "mse_only":
                data["ce_loss_indexes"] = None
            elif case == "mixed_mse" and rank == 1:
                data["mse_loss_indexes"] = torch.empty(0)
            elif case == "mixed_ce" and rank == 1:
                data["ce_loss_indexes"] = None
            elif case == "mixed_vit" and rank == 1:
                # Keep seqlens present: both fields must satisfy the model's AND condition.
                data["packed_vit_tokens"] = None
            elif case == "empty_ce_all" or (case == "empty_ce_one" and rank == 0):
                data["ce_loss_indexes"] = torch.empty(0)
            elif case == "no_targets":
                data["ce_loss_indexes"] = None
                data["mse_loss_indexes"] = torch.empty(0)
            before = {key: value.clone() if isinstance(value, torch.Tensor) else value
                      for key, value in data.items()}
            error = None
            try:
                check_batch_participation(data, visual_gen=True, visual_und=True, device="cpu")
            except ValueError as exc:
                error = str(exc)
            if expected_error is None:
                if error is not None:
                    raise AssertionError(f"{case}: unexpected rejection: {error}")
            elif error is None or expected_error not in error:
                raise AssertionError(f"{case}: expected {expected_error!r}, got {error!r}")
            for key, value in before.items():
                if value is None:
                    assert data[key] is None
                else:
                    assert torch.equal(data[key], value), f"guard changed {key}"
            # Even rejected batches leave every rank at the same collective boundary.
            dist.barrier()
            marker = torch.tensor(rank + 1)
            dist.all_reduce(marker)
            assert marker.item() == 3
            results[case] = error
        (Path(result_dir) / f"rank-{rank}.json").write_text(json.dumps(results))
    except BaseException:
        (Path(result_dir) / f"rank-{rank}.error").write_text(traceback.format_exc())
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class DistributedTrainingTests(unittest.TestCase):
    def require_gloo(self):
        error = None
        if importlib.util.find_spec("torch") is None:
            error = "PyTorch is not installed"
        else:
            try:
                import torch.distributed as dist
                if not dist.is_available() or not dist.is_gloo_available():
                    error = "PyTorch's Gloo backend is unavailable"
            except (ImportError, OSError) as exc:
                error = str(exc)
        if error:
            if os.environ.get("AGIMA_REQUIRE_DISTRIBUTED_SMOKE") == "1":
                self.fail(error)
            self.skipTest(error)

    def test_two_ranks_agree_before_forward_and_loss_collectives(self):
        self.require_gloo()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rendezvous = (root / "rendezvous").as_uri()
            context = multiprocessing.get_context("spawn")
            workers = [context.Process(target=distributed_worker, args=(rank, rendezvous, directory))
                       for rank in range(2)]
            try:
                for worker in workers:
                    worker.start()
                deadline = time.monotonic() + 45
                for worker in workers:
                    worker.join(timeout=max(0, deadline - time.monotonic()))
                errors = "\n".join(path.read_text() for path in root.glob("*.error"))
                self.assertFalse(any(worker.is_alive() for worker in workers), "Gloo workers hung\n" + errors)
                self.assertTrue(all(worker.exitcode == 0 for worker in workers), errors)
                results = [json.loads((root / f"rank-{rank}.json").read_text()) for rank in range(2)]
                self.assertEqual(results[0], results[1])
                self.assertEqual(set(results[0]), {case for case, _ in CASES})
            finally:
                for worker in workers:
                    if worker.is_alive():
                        worker.kill()
                    if worker.pid is not None:
                        worker.join(timeout=3)

    def test_trainer_checks_batch_before_fsdp_forward(self):
        tree = ast.parse((ROOT / "training/pretrain_unified_navit.py").read_text())
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)]
        guards = [node for node in calls if isinstance(node.func, ast.Name)
                  and node.func.id == "check_batch_participation"]
        forwards = [node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "fsdp_model"]
        self.assertEqual(len(guards), 1)
        self.assertTrue(forwards)
        self.assertLess(guards[0].lineno, min(node.lineno for node in forwards))
        guard_statement = next(node for node in ast.walk(main) if isinstance(node, ast.Expr)
                               and node.value is guards[0])
        train_loop = next(node for node in ast.walk(main) if isinstance(node, ast.For)
                          and guard_statement in node.body)
        self.assertIn(guard_statement, train_loop.body)  # Unconditional for every rank/batch.


if __name__ == "__main__":
    unittest.main()
