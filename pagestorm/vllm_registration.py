from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from transformers import PretrainedConfig


FIXED_ADAPTER_NAME = "book_preview"
FIXED_ADAPTER_IFFM = 0.125
FIXED_ADAPTER_HF_ARCHITECTURE = "Ministral3ForCausalLM"
FIXED_ADAPTER_VLLM_ARCHITECTURE = "PagestormMinistral3AdapterForCausalLM"
FIXED_ADAPTER_MODEL_IMPORT = "pagestorm.vllm_ministral3_adapter:PagestormMinistral3ForCausalLM"
JsonObject = dict[str, object]


class PagestormMinistral3Config(PretrainedConfig):
    model_type = "ministral3"


def _read_config(bundle_path: Path) -> JsonObject:
    config_path = bundle_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json at {config_path}")
    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config_payload, dict):
        raise ValueError(f"config.json at {config_path} must decode to an object.")
    return cast(JsonObject, config_payload)


def _validate_fixed_adapter_config(config_payload: JsonObject) -> None:
    if config_payload.get("architectures") != [FIXED_ADAPTER_HF_ARCHITECTURE]:
        raise ValueError(f"Fixed adapter exports must use architectures=['{FIXED_ADAPTER_HF_ARCHITECTURE}'].")
    adapters = config_payload.get("adapters")
    expected_adapters = [{"name": FIXED_ADAPTER_NAME, "iffm": FIXED_ADAPTER_IFFM}]
    if adapters != expected_adapters:
        raise ValueError(f"Fixed adapter exports must include exactly adapters={expected_adapters!r}.")


def _register_transformers_ministral3_config() -> None:
    from transformers import AutoConfig

    AutoConfig.register("ministral3", PagestormMinistral3Config, exist_ok=True)


def _register_vllm_model() -> None:
    from vllm import ModelRegistry

    ModelRegistry.register_model(FIXED_ADAPTER_VLLM_ARCHITECTURE, FIXED_ADAPTER_MODEL_IMPORT)


def register_pagestorm_vllm_plugin() -> None:
    _register_transformers_ministral3_config()
    _register_vllm_model()


def configure_pagestorm_vllm_model(bundle_path: Path) -> JsonObject | None:
    config_payload = _read_config(bundle_path)
    if "adapters" not in config_payload:
        return None
    _validate_fixed_adapter_config(config_payload)
    _register_transformers_ministral3_config()
    _register_vllm_model()
    return {"architectures": [FIXED_ADAPTER_VLLM_ARCHITECTURE]}
