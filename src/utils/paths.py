from pathlib import Path

# Project root: root is 2 levels up from src/utils/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

METADATA_CSV = METADATA_DIR / "metadata.csv"
SPLITS_CSV = METADATA_DIR / "splits.csv"

# Configs directory
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_CONFIG_PATH = CONFIGS_DIR / "data.yaml"

# Artifacts directories
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
LOGS_DIR = ARTIFACTS_DIR / "logs"
MODELS_DIR = ARTIFACTS_DIR / "models"
PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

# Docs & Notebooks
DOCS_DIR = PROJECT_ROOT / "docs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
