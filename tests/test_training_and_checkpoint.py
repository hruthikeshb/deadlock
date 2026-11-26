from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from deadlock_monsoon.checkpoint import export_grast_bundle, load_grast_bundle
from deadlock_monsoon.model import GRAST, GRASTConfig
from deadlock_monsoon.training import curriculum_horizon, normalized_lead_weights, rollout_huber_loss


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

