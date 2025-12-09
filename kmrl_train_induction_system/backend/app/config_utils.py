import yaml
from pathlib import Path


def load_config() -> dict:
    cfg_path = Path(__file__).parent / "config.yaml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)



