from __future__ import annotations

import pytest
import torch
from torch import nn

pytest.importorskip("vllm", reason="The adapter integration test requires a vLLM runtime.")

from pagestorm.vllm_ministral3_adapter import (  # noqa: E402
    PagestormResidualAdapter,
    set_pagestorm_adapters_enabled,
)


class IdentityParallelProjection(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
        return hidden_states, None


def _adapter_for_test() -> PagestormResidualAdapter:
    adapter = PagestormResidualAdapter.__new__(PagestormResidualAdapter)
    nn.Module.__init__(adapter)
    adapter.in_proj = IdentityParallelProjection()
    adapter.out_proj = IdentityParallelProjection()
    adapter.register_buffer("output_scale", torch.ones(()), persistent=False)
    return adapter


def test_residual_adapter_can_be_disabled_without_removing_weights() -> None:
    adapter = _adapter_for_test()
    model = nn.Sequential(adapter)
    hidden_states = torch.tensor([[[-1.0, 2.0]]])

    enabled_output = model(hidden_states)
    adapter_count = set_pagestorm_adapters_enabled(model, enabled=False)
    disabled_output = model(hidden_states)

    assert adapter_count == 1
    assert not torch.equal(enabled_output, hidden_states)
    assert torch.equal(disabled_output, hidden_states)
    assert list(adapter.state_dict()) == []
