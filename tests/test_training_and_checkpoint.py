from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from deadlock_monsoon.checkpoint import export_grast_bundle, load_grast_bundle
from deadlock_monsoon.model import GRAST, GRASTConfig
from deadlock_monsoon.training import (
    RainLossConfig,
    curriculum_horizon,
    normalized_lead_weights,
    rollout_huber_loss,
)


def test_lead_weights_and_curriculum_are_explicit():
    assert torch.isclose(normalized_lead_weights(10).sum(), torch.tensor(1.0))
    increasing = normalized_lead_weights(10, "mild_increase")
    assert increasing[-1] > increasing[0]
    assert [curriculum_horizon(epoch) for epoch in (0, 3, 6, 10)] == [2, 4, 8, 10]


def test_rollout_loss_reports_every_lead_and_channel_and_backpropagates():
    prediction = torch.nn.Parameter(torch.zeros(2, 4, 6, 32, 36))
    target = torch.ones_like(prediction)
    result = rollout_huber_loss(prediction, target, torch.ones(6))
    assert result.per_lead.shape == (4,)
    assert result.per_channel.shape == (6,)
    assert torch.isfinite(result.rain_occurrence)
    assert torch.isfinite(result.rain_tail)
    result.total.backward()
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()


def test_tiny_optimization_reduces_multi_step_loss():
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.SGD([parameter], lr=0.2)
    target = torch.ones(1, 4, 6, 32, 36)
    initial = None
    for _ in range(12):
        prediction = parameter.expand_as(target)
        loss = rollout_huber_loss(prediction, target, torch.ones(6)).total
        initial = float(loss.detach()) if initial is None else initial
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert float(loss.detach()) < initial * 0.25


def test_bundle_round_trip_and_hash_guard(tmp_path):
    model = GRAST(GRASTConfig(width=16, blocks=2))
    export_grast_bundle(
        tmp_path,
        model,
        role="fallback",
        normalization_mean=np.arange(6, dtype=np.float32),
        normalization_scale=np.arange(1, 7, dtype=np.float32),
        split={"train_end": "2017-12-31", "validation_end": "2019-12-31"},
        training={"horizon": 10, "lead_weights": "uniform", "seed": 7},
        code_commit="test",
        static_features=np.zeros((2, 32, 36), dtype=np.float32),
    )
    restored, manifest = load_grast_bundle(tmp_path, expected_role="fallback")
    assert manifest.training["horizon"] == 10
    for before, after in zip(model.parameters(), restored.parameters()):
        assert torch.equal(before, after)

    manifest_path = tmp_path / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["weights_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        load_grast_bundle(tmp_path)


def test_rain_objective_penalizes_missing_heavy_rain_more_than_base_loss():
    prediction = torch.zeros(1, 2, 6, 32, 36)
    target = torch.zeros_like(prediction)
    target[:, :, 1, 4:8, 4:8] = 25.0
    base = rollout_huber_loss(prediction, target, torch.ones(6)).total
    aware = rollout_huber_loss(
        prediction,
        target,
        torch.ones(6),
        rain=RainLossConfig(occurrence_weight=0.2, tail_weight=0.5),
    ).total
    assert aware > base


def test_rain_objective_remains_finite_without_events():
    prediction = torch.zeros(1, 2, 6, 32, 36)
    target = torch.zeros_like(prediction)
    result = rollout_huber_loss(
        prediction,
        target,
        torch.ones(6),
        rain=RainLossConfig(occurrence_weight=0.2, tail_weight=0.5),
    )
    assert torch.isfinite(result.total)
    assert result.rain_tail == 0


def test_direct_occurrence_logits_receive_gradients():
    prediction = torch.zeros(1, 2, 6, 32, 36, requires_grad=True)
    target = torch.zeros_like(prediction)
    target[:, :, 1, 2:5, 2:5] = 20.0
    logits = torch.zeros(1, 2, 32, 36, requires_grad=True)
    result = rollout_huber_loss(
        prediction,
        target,
        torch.ones(6),
        rain=RainLossConfig(occurrence_weight=0.2, tail_weight=0.2),
        rain_occurrence_logits=logits,
    )
    result.total.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_occurrence_weight_is_fixed_not_batch_prevalence_dependent():
    prediction = torch.zeros(1, 1, 6, 32, 36)
    logits = torch.zeros(1, 1, 32, 36)
    sparse = torch.zeros_like(prediction)
    dense = torch.zeros_like(prediction)
    sparse[:, :, 1, 0, 0] = 20.0
    dense[:, :, 1, :16] = 20.0
    config = RainLossConfig(occurrence_weight=1.0, positive_class_weight=1.0)
    sparse_loss = rollout_huber_loss(
        prediction, sparse, torch.ones(6), rain=config, rain_occurrence_logits=logits
    ).rain_occurrence
    dense_loss = rollout_huber_loss(
        prediction, dense, torch.ones(6), rain=config, rain_occurrence_logits=logits
    ).rain_occurrence
    assert torch.allclose(sparse_loss, dense_loss)

