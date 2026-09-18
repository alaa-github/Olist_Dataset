import yaml
from pathlib import Path
from loguru import logger

def load_config(config_path: str = "config/config.yaml"):
    """Load the YAML configuration file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise e

def get_project_root() -> Path:
    """Get the root directory of the project."""
    return Path(__file__).parent.parent
