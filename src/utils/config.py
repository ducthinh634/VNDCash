import yaml
from pathlib import Path
from typing import Any, Dict, Union

def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Loads a YAML configuration file safely."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def save_config(config: Dict[str, Any], path: Union[str, Path]) -> None:
    """Saves a configuration dictionary to a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False, allow_unicode=True)
