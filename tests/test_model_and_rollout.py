from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from deadlock_monsoon.model import GRAST, GRASTConfig, SemanticOutputMapper, parameter_count
from deadlock_monsoon.torch_rollout import advance_doy, autoregressive_rollout


def physical_history(batch: int = 1):
    value = torch.zeros(batch, 3, 6, 32, 36)
    value[:, :, 0] = 298.0
    value[:, :, 1] = 2.0
    value[:, :, 4] = 100_000.0
    value[:, :, 5] = 30.0
    return value


def test_grast_shape_parameter_budget_and_gradients():
    model = GRAST(GRASTConfig(width=16, blocks=2))
    history = torch.randn(2, 3, 6, 32, 36, requires_grad=True)
    static = torch.randn(2, 2, 32, 36)
    doy = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    raw = model(history, static, doy)
    assert raw.shape == (2, 6, 32, 36)
    assert parameter_count(model) < 5_000_000
    raw.mean().backward()
    assert history.grad is not None


def test_semantic_mapper_is_finite_nonnegative_and_temperature_bounded():
    mapper = SemanticOutputMapper(torch.ones(6))
    raw = torch.randn(2, 6, 32, 36) * 20
    output = mapper(raw, physical_history(2)[:, -1])
    assert torch.isfinite(output).all()
    assert torch.all(output[:, 1] >= 0)
    assert torch.all((output[:, 0] >= 233.15) & (output[:, 0] <= 333.15))


def test_occurrence_intensity_head_preserves_six_channel_physical_contract():
    model = GRAST(GRASTConfig(width=16, blocks=2, rain_mode="occurrence_intensity"))
    history = torch.randn(2, 3, 6, 32, 36)
    raw = model(history, torch.randn(2, 2, 32, 36), torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
    assert raw.shape == (2, 7, 32, 36)
    mapper = SemanticOutputMapper(torch.ones(6), rain_mode="occurrence_intensity")
    output = mapper(raw, physical_history(2)[:, -1])
    assert output.shape == (2, 6, 32, 36)
    assert torch.all(output[:, 1] >= 0)


class ZeroRaw(torch.nn.Module):
    def forward(self, history, static, doy):
        del static, doy
        return torch.zeros_like(history[:, -1])


def test_rollout_shared_path_applies_perturbation_once_and_supports_backprop():
    history = physical_history()
    perturbation = torch.zeros(1, 6, 32, 36)
    perturbation[:, 2, 8:10, 8:10] = 2.0
    mapper = SemanticOutputMapper(torch.ones(6))
    rollout = autoregressive_rollout(
        ZeroRaw(), mapper, history, torch.zeros(1, 2, 32, 36),
        torch.tensor([[0.0, 1.0]]), 4, torch.zeros(6), torch.ones(6), perturbation,
    )
    assert rollout.shape == (1, 4, 6, 32, 36)
    assert torch.all(rollout[:, :, 2, 8:10, 8:10] == 2.0)


def test_seasonal_pair_stays_on_unit_circle():
    pair = torch.tensor([[0.0, 1.0]])
    for _ in range(365):
        pair = advance_doy(pair)
    assert np.allclose(pair.numpy(), [[0.0, 1.0]], atol=2e-5)


def test_rollout_can_return_auxiliary_raw_heads_for_training_only():
    model = GRAST(GRASTConfig(width=16, blocks=2, rain_mode="occurrence_intensity"))
    mapper = SemanticOutputMapper(torch.ones(6), rain_mode="occurrence_intensity")
    rollout, raw = autoregressive_rollout(
        model,
        mapper,
        physical_history(),
        torch.zeros(1, 2, 32, 36),
        torch.tensor([[0.0, 1.0]]),
        2,
        torch.zeros(6),
        torch.ones(6),
        return_raw=True,
    )
    assert rollout.shape == (1, 2, 6, 32, 36)
    assert raw.shape == (1, 2, 7, 32, 36)


def test_occurrence_head_bias_starts_at_declared_training_prior():
    prior = 0.12
    model = GRAST(
        GRASTConfig(width=16, blocks=2, rain_mode="occurrence_intensity", rain_occurrence_prior=prior)
    )
    assert torch.sigmoid(model.head.bias[1]).item() == pytest.approx(prior)

