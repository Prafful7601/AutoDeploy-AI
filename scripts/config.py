"""
Shared configuration for AutoDeploy AI.

Every script imports SEED from here and seeds every source of randomness it
touches (python's random, numpy, and each model's own random_state) with it.
This is what makes data splits, model training, and reported metrics
reproducible run to run.

Also centralizes filesystem paths so scripts don't hardcode relative paths
that break depending on where they're invoked from.
"""

from pathlib import Path

# --- Reproducibility ---
SEED = 42

# --- Paths (all relative to the project root, resolved absolutely) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"


def set_global_seed(seed: int = SEED) -> None:
    """Seed python's random and numpy so any script can call this once at
    startup and get deterministic behavior everywhere that doesn't take an
    explicit random_state (e.g. numpy-based shuffling, sampling)."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
