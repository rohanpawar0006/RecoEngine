"""Seed utilities for reproducible experiments."""

import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across Python and NumPy (and TF if imported)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

