"""Exercise the actual JSONL indexing, partitioning and resume methods without Torch."""

import ast
import gzip
import json
import os
from pathlib import Path
import random
import re
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def source_loaders():
    worker = SimpleNamespace(info=None)
    repeats = []

    def bounded_print(*args, **kwargs):
        # A broken infinite iterator must fail this test rather than hang CI.
        if "repeat in rank-" in " ".join(map(str, args)):
            repeats.append(True)
            if len(repeats) > 8:
                raise RuntimeError("Loader repeated without making expected progress")

    namespace = {
        "os": os, "gzip": gzip, "json": json, "random": random, "_re": re,
        "print": bounded_print,
        "torch": SimpleNamespace(utils=SimpleNamespace(data=SimpleNamespace(
            get_worker_info=lambda: worker.info))),
    }
    path = ROOT / "data/distributed_iterable_dataset.py"
    tree = ast.parse(path.read_text())
    base = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                and node.name == "DistributedIterableDataset")
    base.bases = []  # Replace only Torch's base; execute the real partition methods.
    exec(compile(ast.fix_missing_locations(ast.Module(body=[base], type_ignores=[])),
                 str(path), "exec"), namespace)
    path = ROOT / "data/interleave_datasets/interleave_t2i_dataset.py"
    tree = ast.parse(path.read_text())
    names = {"_open_jsonl", "_extract_order_key", "_worker_resume_cursor", "_jsonl_resume_start",
             "JsonLIterableDataset", "JsonLIterableDatasetEvalOnce", "JsonLIterableDatasetEvalOnceShardPad"}
    nodes = [node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    nodes.insert(0, next(node for node in tree.body if isinstance(node, ast.Assign)
                         and any(isinstance(target, ast.Name) and target.id == "_ORDER_KEY_RE"
                                 for target in node.targets)))
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 str(path), "exec"), namespace)
    return namespace, worker


def row(index, indexed=False):
    value = {"__bagel_order_key": f"{index:012d}"} if indexed else {}
    value.update(index=index, messages=[
        {"role": "user", "content": [{"type": "text", "text": "observation"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "action"}]},
    ])
    return json.dumps(value) + "\n"


class DataLoadingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.ns, self.worker = source_loaders()

    def write(self, name, text):
        path = self.root / name
        if name.endswith(".gz"):
            with gzip.open(path, "wt") as stream:
                stream.write(text)
        else:
            path.write_text(text)
        return path

    def loader(self, paths=None, *, rank=0, world=1, limit=100000, status=None,
               streaming=False, cls="JsonLIterableDataset"):
        with patch.dict(os.environ, {"BAGEL_JSONL_STREAMING": str(int(streaming))}):
            loader = self.ns[cls](
                "fixture", SimpleNamespace(stride=1), None, None, [], [limit],
                [str(path) for path in (paths or [self.root])], local_rank=rank,
                world_size=world, num_workers=1, data_status=status)
        loader.parse_row = lambda text: json.loads(text) if text.strip() else []
        return loader

    def test_blank_lines_never_occupy_a_rank_partition(self):
        from scripts._common import validate_rows

        for indexed in (False, True):
            with self.subTest(indexed=indexed):
                path = self.write("part.jsonl", "\n" + row(0, indexed) + row(1, indexed))
                self.assertEqual(validate_rows([path], minimum=2), 2)
                samples = [next(iter(self.loader([path], rank=rank, world=2))) for rank in range(2)]
                self.assertEqual({sample["index"] for sample in samples}, {0, 1})

    def test_empty_and_all_blank_inputs_fail_before_iteration(self):
        for contents in ("", "\n \n\t\n"):
            with self.subTest(contents=contents):
                path = self.write("blank.jsonl.gz", contents)
                with self.assertRaisesRegex(ValueError, "no data entries"):
                    self.loader([path])

    def test_rank_and_worker_partitions_cannot_be_empty(self):
        path = self.write("small.jsonl", row(0))
        for rank in range(2):
            with self.subTest(rank=rank):
                with self.assertRaisesRegex(ValueError, "has no entries"):
                    next(iter(self.loader([path], rank=rank, world=2)))
        loader = self.loader([path])
        self.worker.info = SimpleNamespace(id=0, num_workers=2)
        with self.assertRaisesRegex(ValueError, "reduce num_workers"):
            next(iter(loader))

    def test_all_filtered_rows_raise_instead_of_repeating_forever(self):
        path = self.write("filtered.jsonl", row(0) + row(1))
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                loader = self.loader([path], streaming=streaming)
                loader.parse_row = lambda text: []
                with self.assertRaisesRegex(ValueError, "produced no valid rows"):
                    next(iter(loader))

    def test_row_limit_is_global_across_shards_in_both_index_modes(self):
        for indexed in (False, True):
            with self.subTest(indexed=indexed):
                paths = [self.write(f"part-{i}.jsonl", "\n" + "".join(row(i * 3 + j, indexed)
                                                                        for j in range(3)))
                         for i in range(3)]
                loader = self.loader(paths, limit=2)
                iterator = iter(loader)
                self.assertEqual(len(loader.data_paths), 2)
                self.assertEqual({next(iterator)["index"] for _ in range(2)}, {0, 1})

    def test_valid_rows_keep_the_original_shuffle_and_worker_order(self):
        for indexed in (False, True):
            contents = [row(i, indexed) for i in range(8)]
            path = self.write("ordered.jsonl", "".join(contents))
            expected = sorted(contents)
            random.Random(42).shuffle(expected)
            for rank in range(2):
                for worker_id in range(2):
                    with self.subTest(indexed=indexed, rank=rank, worker=worker_id):
                        self.worker.info = SimpleNamespace(id=worker_id, num_workers=2)
                        loader = self.loader([path], rank=rank, world=2)
                        start = rank * 4 + worker_id * 2
                        expected_rows = [json.loads(text)["index"] for text in expected[start:start + 2][::-1]]
                        iterator = iter(loader)
                        self.assertEqual([next(iterator)["index"] for _ in range(2)], expected_rows)

    def test_saved_and_legacy_cursors_resume_after_the_last_yielded_row(self):
        path = self.write("resume.jsonl", "".join(row(i, True) for i in range(4)))
        original = iter(self.loader([path]))
        first = next(original)
        expected = next(original)["index"]
        saved = first["data_indexes"]["data_indexes"]
        for status in ({0: saved}, {0: saved[0]}, {"0": saved}, {0: tuple(saved)}):
            with self.subTest(status=status):
                resumed = self.loader([path], status=status)
                self.assertEqual(next(iter(resumed))["index"], expected)

    def test_last_row_cursor_wraps_to_the_original_epoch_start(self):
        path = self.write("resume.jsonl", row(0) + row(1))
        loader = self.loader([path])
        iterator = iter(loader)
        first, last = next(iterator), next(iterator)
        resumed = self.loader([path], status={0: last["data_indexes"]["data_indexes"]})
        self.assertEqual(next(iter(resumed))["index"], first["index"])

    def test_invalid_cursors_are_explicit_errors(self):
        path = self.write("resume.jsonl", row(0) + row(1))
        for cursor in (None, -1, 2, [], [0, 1], True, "0"):
            with self.subTest(cursor=cursor):
                with self.assertRaisesRegex(ValueError, "Invalid JSONL resume cursor"):
                    next(iter(self.loader([path], status={0: cursor})))
        with self.assertRaisesRegex(ValueError, "no cursor for worker 0"):
            next(iter(self.loader([path], status={1: [0]})))
        with self.assertRaisesRegex(ValueError, "must map worker IDs"):
            next(iter(self.loader([path], status=[[0]])))

    def test_eval_once_accepts_saved_cursor_and_skips_blank_lines(self):
        path = self.write("eval.jsonl", "\n" + row(0) + row(1))
        original = list(self.loader([path], cls="JsonLIterableDatasetEvalOnce"))
        resumed = self.loader([path], status={0: original[0]["data_indexes"]["data_indexes"]},
                              cls="JsonLIterableDatasetEvalOnce")
        self.assertEqual([sample["index"] for sample in resumed], [original[1]["index"]])

    def test_padded_eval_can_still_have_fewer_rows_than_ranks(self):
        path = self.write("eval.jsonl", row(0))
        self.ns["dist"] = SimpleNamespace(get_rank=lambda: 1, get_world_size=lambda: 2)
        loader = self.loader([path], rank=1, world=2, cls="JsonLIterableDatasetEvalOnceShardPad")
        samples = list(loader)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["eval_weight"], 0.0)

    def test_streaming_empty_shards_and_partitions_fail_fast(self):
        path = self.write("blank.jsonl", "\n")
        with self.assertRaisesRegex(ValueError, "produced no valid rows"):
            next(iter(self.loader([path], streaming=True)))
        with self.assertRaisesRegex(ValueError, "no JSONL shards"):
            next(iter(self.loader([path], streaming=True, rank=1, world=2)))
        loader = self.loader([path], streaming=True)
        self.worker.info = SimpleNamespace(id=1, num_workers=2)
        with self.assertRaisesRegex(ValueError, "no JSONL shards"):
            next(iter(loader))

    def test_streaming_saved_file_line_cursor_resumes_across_shards(self):
        paths = [self.write(f"part-{i}.jsonl.gz", "\n" + row(i * 2) + "\n" + row(i * 2 + 1))
                 for i in range(2)]
        original = iter(self.loader(paths, streaming=True))
        first = next(original)
        last_in_shard = next(original)
        expected = next(original)
        self.assertEqual(first["data_indexes"]["data_indexes"][1], 1)
        self.assertEqual(last_in_shard["data_indexes"]["data_indexes"][1], 3)
        resumed = self.loader(paths, streaming=True,
                              status={0: last_in_shard["data_indexes"]["data_indexes"]})
        self.assertEqual(next(iter(resumed))["index"], expected["index"])
        legacy = self.loader(paths, streaming=True, status={0: 1})
        self.assertEqual(next(iter(legacy))["index"], expected["index"])

    def test_streaming_cursors_must_identify_existing_nonblank_rows(self):
        path = self.write("resume.jsonl", "\n" + row(0) + row(1))
        for cursor in ([str(path), 0], [str(path), 8], [str(path) + "-other", 1], 99):
            with self.subTest(cursor=cursor):
                with self.assertRaisesRegex(ValueError, "resume cursor"):
                    next(iter(self.loader([path], streaming=True, status={0: cursor})))

    def test_streaming_last_cursor_restarts_at_the_worker_row_limit(self):
        path = self.write("limited.jsonl", "\n" + "".join(row(i) for i in range(4)))
        original = iter(self.loader([path], streaming=True, limit=2))
        first, last = next(original), next(original)
        self.assertEqual(next(original)["index"], first["index"])
        for cursor in (1, last["data_indexes"]["data_indexes"]):
            with self.subTest(cursor=cursor):
                resumed = self.loader([path], streaming=True, limit=2, status={0: cursor})
                self.assertEqual(next(iter(resumed))["index"], first["index"])
        with self.assertRaisesRegex(ValueError, "exceeds the worker's row limit"):
            next(iter(self.loader([path], streaming=True, limit=2, status={0: [str(path), 3]})))

    def test_streaming_limit_keeps_rank_and_worker_partitioning(self):
        paths = [self.write(f"part-{i}.jsonl", "".join(row(i * 4 + j) for j in range(4)))
                 for i in range(4)]
        for rank in range(2):
            for worker_id in range(2):
                with self.subTest(rank=rank, worker=worker_id):
                    self.worker.info = SimpleNamespace(id=worker_id, num_workers=2)
                    loader = self.loader(paths, streaming=True, rank=rank, world=2, limit=8)
                    iterator = iter(loader)
                    first, last = next(iterator), next(iterator)
                    self.assertEqual(next(iterator)["index"], first["index"])
                    resumed = self.loader(paths, streaming=True, rank=rank, world=2, limit=8,
                                          status={worker_id: last["data_indexes"]["data_indexes"]})
                    self.assertEqual(next(iter(resumed))["index"], first["index"])

    def test_streaming_cap_below_partition_count_fails_before_parsing(self):
        paths = [self.write(f"part-{i}.jsonl", row(i)) for i in range(8)]
        for world, workers in ((8, 1), (2, 4)):
            with self.subTest(world=world, workers=workers):
                self.worker.info = SimpleNamespace(id=0, num_workers=workers)
                loader = self.loader(paths, streaming=True, world=world, limit=2)
                loader.parse_row = lambda text: self.fail("Invalid row limit must fail before parsing")
                with self.assertRaisesRegex(ValueError, "row limit 2 is smaller than 8"):
                    next(iter(loader))


if __name__ == "__main__":
    unittest.main()
