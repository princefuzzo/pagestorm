from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PretrainedConfig
from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.llama import LlamaDecoderLayer, LlamaForCausalLM, LlamaModel

from .vllm_registration import FIXED_ADAPTER_IFFM, FIXED_ADAPTER_NAME


class PagestormResidualAdapter(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        adapter_hidden_size: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.in_proj = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=adapter_hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj",
        )
        self.out_proj = RowParallelLinear(
            input_size=adapter_hidden_size,
            output_size=hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        projected, _ = self.in_proj(hidden_states)
        activated = F.gelu(projected, approximate="none")
        updated, _ = self.out_proj(activated)
        return updated + hidden_states


def _validate_model_adapter_config(config: PretrainedConfig) -> None:
    adapters = getattr(config, "adapters", None)
    expected_adapters = [{"name": FIXED_ADAPTER_NAME, "iffm": FIXED_ADAPTER_IFFM}]
    if adapters != expected_adapters:
        raise ValueError(f"Pagestorm fixed-adapter model requires exactly adapters={expected_adapters!r}.")
    hidden_size = getattr(config, "hidden_size", None)
    if not isinstance(hidden_size, int):
        raise TypeError("Pagestorm fixed-adapter config must define integer hidden_size.")
    _fixed_adapter_hidden_size(hidden_size)


def _fixed_adapter_hidden_size(hidden_size: int) -> int:
    adapter_hidden_size = int(hidden_size * FIXED_ADAPTER_IFFM)
    if adapter_hidden_size <= 0:
        raise ValueError("Pagestorm fixed-adapter hidden size must be positive.")
    return adapter_hidden_size


class PagestormMinistral3DecoderLayer(LlamaDecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        config: PretrainedConfig | None = None,
        attn_layer_type: type[nn.Module] | None = None,
    ) -> None:
        if attn_layer_type is None:
            super().__init__(vllm_config=vllm_config, prefix=prefix, config=config)
        else:
            super().__init__(vllm_config=vllm_config, prefix=prefix, config=config, attn_layer_type=attn_layer_type)
        adapter_hidden_size = _fixed_adapter_hidden_size(self.hidden_size)
        quant_config = self.get_quant_config(vllm_config)
        self.attn_adapter = PagestormResidualAdapter(
            hidden_size=self.hidden_size,
            adapter_hidden_size=adapter_hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.attn_adapter",
        )
        self.mlp_adapter = PagestormResidualAdapter(
            hidden_size=self.hidden_size,
            adapter_hidden_size=adapter_hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp_adapter",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states = self.attn_adapter(hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.mlp_adapter(hidden_states)
        return hidden_states, residual


class PagestormMinistral3Model(LlamaModel):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix, layer_type=PagestormMinistral3DecoderLayer)


def _rewrite_adapter_weight_name(name: str) -> str:
    replacements = {
        ".attn_adapter.in_proj_weight": ".attn_adapter.in_proj.weight",
        ".attn_adapter.out_proj_weight": ".attn_adapter.out_proj.weight",
        ".mlp_adapter.in_proj_weight": ".mlp_adapter.in_proj.weight",
        ".mlp_adapter.out_proj_weight": ".mlp_adapter.out_proj.weight",
    }
    rewritten_name = name
    for source_suffix, target_suffix in replacements.items():
        if rewritten_name.endswith(source_suffix):
            return rewritten_name[: -len(source_suffix)] + target_suffix
    return rewritten_name


def _rewrite_adapter_weights(weights: Iterable[tuple[str, torch.Tensor]]) -> Iterable[tuple[str, torch.Tensor]]:
    for name, tensor in weights:
        yield _rewrite_adapter_weight_name(name), tensor


class PagestormMinistral3ForCausalLM(LlamaForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        _validate_model_adapter_config(cast(PretrainedConfig, vllm_config.model_config.hf_config))
        super().__init__(vllm_config=vllm_config, prefix=prefix, layer_type=PagestormMinistral3DecoderLayer)

    def _init_model(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[nn.Module] = PagestormMinistral3DecoderLayer,
    ) -> LlamaModel:
        return PagestormMinistral3Model(vllm_config=vllm_config, prefix=prefix)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return super().load_weights(_rewrite_adapter_weights(weights))
