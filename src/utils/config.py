import os
import copy
import yaml


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(config_path: str) -> dict:
    """Load a YAML config with `inherits:` key support."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if "inherits" in config:
        parent_name = config.pop("inherits")
        parent_path = os.path.join(os.path.dirname(config_path), parent_name)
        parent_config = load_config(parent_path)
        config = deep_merge(parent_config, config)

    return config
