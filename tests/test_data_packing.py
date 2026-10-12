"""Run the real packing iterator with small records and no Torch dependency."""

import ast
from pathlib import Path
import random
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CountingRows:
    def __init__(self, rows):
        self.rows = rows
        self.read = 0

    def __iter__(self):
        for row in self.rows:
            self.read += 1
            yield row


def sample(index, tokens, *, plan_items=0, packed_tokens=None):
    return {
        "num_tokens": tokens,
        "sequence_plan": [{} for _ in range(plan_items)],
        "data_indexes": {"dataset_name": "fixture", "data_indexes": [index]},
        "fixture_packed_tokens": tokens if packed_tokens is None else packed_tokens,
    }


def packing_class(source, mandatory):
    path = ROOT / "data/dataset_base.py"
    tree = ast.parse(path.read_text())
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                    and node.name == "PackedDataset")
    names = {"__init__", "__iter__", "_validated_num_tokens"}
    methods = [node for node in original.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in methods} != names:
        raise AssertionError("Missing production packing methods")
    # Only replace Torch's base class and tensor-building operations. Selection,
    # length validation, buffering, and yield decisions execute the source methods.
    fixture = ast.ClassDef(name="PackedDataset", bases=[], keywords=[], body=methods, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[fixture], type_ignores=[]))
    namespace = {
        "random": random.Random(0),
        "get_flattened_position_ids_interpolate": None,
        "get_flattened_position_ids_extrapolate": None,
    }
    exec(compile(module, str(path), "exec"), namespace)
    cls = namespace["PackedDataset"]
    builds = []

    def build_datasets(self, grouped, status):
        builds.append(True)
        return [source], [mandatory], [1.0]

    def set_sequence_status(self):
        return {"curr": 0, "sample_lens": [], "packed": []}

    def pack_sequence(self, row, state):
        state["curr"] += row["fixture_packed_tokens"]
        state["sample_lens"].append(row["fixture_packed_tokens"])
        state["packed"].append(row["data_indexes"]["data_indexes"][0])
        return state

    cls.build_datasets = build_datasets
    cls.set_sequence_status = set_sequence_status
    cls.pack_sequence = pack_sequence
    cls.to_tensor = lambda self, state: {"packed": list(state["packed"]), "token_count": state["curr"]}
    return cls, builds


def dataset(rows, *, mandatory, **overrides):
    source = CountingRows(rows)
    cls, builds = packing_class(source, mandatory)
    settings = {
        "data_config": SimpleNamespace(grouped_datasets={}), "tokenizer": None,
        "special_tokens": {}, "local_rank": 0, "world_size": 1, "num_workers": 1,
        "expected_num_tokens": 100, "max_num_tokens_per_sample": 100, "max_num_tokens": 100,
        "prefer_buffer_before": 40, "max_buffer_size": 2, "max_samples_per_batch": 1,
        **overrides,
    }
    return cls(**settings), source, builds


class DataPackingTests(unittest.TestCase):
    def test_exact_budget_is_accepted_in_both_paths_including_plan_overhead(self):
        for mandatory in (False, True):
            with self.subTest(mandatory=mandatory):
                packed, source, _ = dataset(
                    [sample("boundary", 96, plan_items=2, packed_tokens=100)], mandatory=mandatory)
                batch = next(iter(packed))
                self.assertEqual(batch["token_count"], 100)
                self.assertEqual(batch["packed"], ["boundary"])
                self.assertEqual(source.read, 1)

    def test_oversized_sample_fails_before_reading_or_packing_another_row(self):
        for mandatory in (False, True):
            with self.subTest(mandatory=mandatory):
                packed, source, _ = dataset(
                    [sample("too-long", 97, plan_items=2, packed_tokens=101), sample("next", 10)],
                    mandatory=mandatory)
                with self.assertRaises(ValueError) as error:
                    next(iter(packed))
                message = str(error.exception)
                self.assertIn("sample length 101", message)
                self.assertIn("max_num_tokens_per_sample=100", message)
                self.assertIn("data_indexes=", message)
                self.assertIn("too-long", message)
                self.assertEqual(source.read, 1)

    def test_normal_samples_keep_their_order_and_batch_boundaries(self):
        for mandatory in (False, True):
            with self.subTest(mandatory=mandatory):
                packed, source, _ = dataset(
                    [sample("a", 30), sample("b", 40), sample("c", 20), sample("d", 50)],
                    mandatory=mandatory, expected_num_tokens=70, max_samples_per_batch=10)
                iterator = iter(packed)
                batches = [next(iterator), next(iterator)]
                self.assertEqual([batch["packed"] for batch in batches], [["a", "b"], ["c", "d"]])
                self.assertEqual([batch["token_count"] for batch in batches], [70, 70])
                self.assertEqual(source.read, 4)

    def test_buffered_sample_retains_the_existing_pack_order(self):
        packed, source, _ = dataset(
            [sample("a", 70), sample("b", 50), sample("c", 30), sample("d", 50)],
            mandatory=False, max_samples_per_batch=10)
        iterator = iter(packed)
        batches = [next(iterator), next(iterator)]
        self.assertEqual([batch["packed"] for batch in batches], [["a", "c"], ["b", "d"]])
        self.assertEqual([batch["token_count"] for batch in batches], [100, 100])
        self.assertEqual(source.read, 4)

    def test_single_sample_limit_cannot_exceed_batch_limit(self):
        source = CountingRows([sample("oversized-pack", 101)])
        cls, builds = packing_class(source, mandatory=True)
        with self.assertRaisesRegex(ValueError, "max_num_tokens_per_sample=101 exceeds max_num_tokens=100"):
            cls(data_config=SimpleNamespace(grouped_datasets={}), tokenizer=None, special_tokens={},
                local_rank=0, world_size=1, num_workers=1,
                max_num_tokens_per_sample=101, max_num_tokens=100)
        self.assertEqual(builds, [])
        self.assertEqual(source.read, 0)


if __name__ == "__main__":
    unittest.main()
