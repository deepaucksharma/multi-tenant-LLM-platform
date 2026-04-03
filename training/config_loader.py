"""
Load and merge YAML training configs.
Supports inheritance via 'inherits' key.
"""
import yaml
from pathlib import Path
from typing import Dict, Any

from loguru import logger

CONFIGS_DIR = Path(__file__).parent / "configs"


def deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge two dicts. Override values take precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_name: str) -> Dict[str, Any]:
    """
    Load a YAML config, resolving inheritance.

    Args:
        config_name: Name of the config file (e.g., 'sft_sis.yaml')

    Returns:
        Merged configuration dict
    """
    config_path = CONFIGS_DIR / config_name
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if "inherits" in config:
        parent_name = config.pop("inherits")
        parent_config = load_config(parent_name)
        config = deep_merge(parent_config, config)

    return config


def get_sft_config(tenant_id: str) -> Dict[str, Any]:
    """Get SFT config for a specific tenant."""
    return load_config(f"sft_{tenant_id}.yaml")


def get_dpo_config() -> Dict[str, Any]:
    """Get DPO config."""
    return load_config("dpo_config.yaml")


if __name__ == "__main__":
    import json

    for name in ["sft_sis.yaml", "sft_mfg.yaml", "dpo_config.yaml"]:
        config = load_config(name)
        print(f"\n{'='*50}")
        print(f"Config: {name}")
        print(json.dumps(config, indent=2, default=str))
