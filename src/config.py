from typing import Any

import yaml


def load_config(path: str = "config/config.yaml") -> dict[str, Any]:
    with open(path) as f:
        result: dict[str, Any] = yaml.safe_load(f)
        return result
