"""Project-wide settings. Change values here, not scattered across files."""
import random
from pathlib import Path

import numpy as np

# Folders (built from this file's location, so it works on any machine)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Dataset
DATASET_NAME = "google-research-datasets/go_emotions"
DATASET_CONFIG = "simplified"

# Reproducibility
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Make random choices repeatable."""
    random.seed(seed)
    np.random.seed(seed)