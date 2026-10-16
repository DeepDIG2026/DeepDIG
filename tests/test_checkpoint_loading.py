"""Checkpoint compatibility and singleton Top-K regressions (CPU, no downloads)."""

from collections import OrderedDict
import unittest

import torch
from torch import nn

from model.checkpoint import load_deepdig_checkpoint
from model.RADC_module import ReliabilityAwareTemporalAggregation


class CheckpointFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.static_path = nn.Linear(2, 2, bias=False)
        self.dynamic_path = nn.ModuleDict({
            "radc": nn.ModuleDict({
                "rata": nn.ModuleDict({
                    "region_reliability_mlp": nn.Linear(2, 1, bias=False),
                }),
                "cadc": nn.ModuleDict({
                    "kernel_generator": nn.Conv2d(1, 1, 1, bias=False),
                }),
            }),
        })


class CheckpointLoadingTests(unittest.TestCase):
    def setUp(self):
        self.model = CheckpointFixture()

    def legacy_state(self):
        return OrderedDict([
            ("module.static_path.weight", torch.full((2, 2), 1.0)),
            ("module.sp.weight", torch.full((2, 2), 1.0)),
            ("module.dynamic_path.tadc.accumulator.content_mlp.weight", torch.full((1, 2), 2.0)),
            ("module.dynamic_path.tadc.kernel_generator.weight", torch.full((1, 1, 1, 1), 3.0)),
        ])

    def test_current_raw_and_wrapped_states(self):
        state = self.model.state_dict()
        load_deepdig_checkpoint(self.model, state)
        load_deepdig_checkpoint(self.model, {"state_dict": state, "pipeline_ablation": "full"})

    def test_legacy_dataparallel_and_duplicate_alias(self):
        load_deepdig_checkpoint(self.model, {"state_dict": self.legacy_state()})
        self.assertTrue(torch.equal(self.model.static_path.weight, torch.full((2, 2), 1.0)))
        self.assertTrue(torch.equal(self.model.dynamic_path.radc.rata.region_reliability_mlp.weight, torch.full((1, 2), 2.0)))
        self.assertTrue(torch.equal(self.model.dynamic_path.radc.cadc.kernel_generator.weight, torch.full((1, 1, 1, 1), 3.0)))

    def test_conflicting_aliases_fail_before_loading(self):
        initial = self.model.static_path.weight.detach().clone()
        state = self.legacy_state()
        state["module.sp.weight"] = torch.zeros(2, 2)
        with self.assertRaisesRegex(ValueError, "Conflicting checkpoint aliases"):
            load_deepdig_checkpoint(self.model, state)
        self.assertTrue(torch.equal(self.model.static_path.weight, initial))

    def test_missing_unexpected_and_wrong_shape_are_rejected(self):
        for corruption in ("missing", "unexpected", "shape"):
            with self.subTest(corruption=corruption):
                state = self.model.state_dict()
                if corruption == "missing":
                    del state["static_path.weight"]
                elif corruption == "unexpected":
                    state["unknown.weight"] = torch.zeros(1)
                else:
                    state["static_path.weight"] = torch.zeros(3, 3)
                with self.assertRaises(RuntimeError):
                    load_deepdig_checkpoint(self.model, state)

    def test_ablation_checkpoint_rejected(self):
        with self.assertRaisesRegex(ValueError, "complete"):
            load_deepdig_checkpoint(self.model, {"state_dict": self.model.state_dict(), "pipeline_ablation": "plain"})

    def test_non_tensor_state_rejected(self):
        with self.assertRaises(TypeError):
            load_deepdig_checkpoint(self.model, {"static_path.weight": "invalid"})


class TemporalAggregationTests(unittest.TestCase):
    def test_singleton_topk_is_finite_and_initially_uniform(self):
        module = ReliabilityAwareTemporalAggregation(3).eval()
        difference = torch.randn(2, 3, 16, 16)
        with torch.inference_mode():
            accumulated = module(difference)
        self.assertTrue(bool(torch.isfinite(accumulated).all()))
        torch.testing.assert_close(accumulated, difference.abs().sum(dim=1, keepdim=True), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
