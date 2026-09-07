"""
Random seed and reproducibility utilities.
"""

import os
import random

import numpy as np
import torch


# ============================================================
# SET GLOBAL SEED
# ============================================================

def set_seed(
    seed: int = 42,
    deterministic: bool = False,
) -> None:
    """
    Set random seeds for reproducible experiments.

    Args:
        seed:
            Random seed value.

        deterministic:
            If True, request deterministic PyTorch/CUDA behavior.
            This can reduce performance but improves reproducibility.
    """

    # --------------------------------------------------------
    # Python
    # --------------------------------------------------------

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)

    # --------------------------------------------------------
    # NumPy
    # --------------------------------------------------------

    np.random.seed(seed)

    # --------------------------------------------------------
    # PyTorch CPU
    # --------------------------------------------------------

    torch.manual_seed(seed)

    # --------------------------------------------------------
    # PyTorch CUDA
    # --------------------------------------------------------

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    # --------------------------------------------------------
    # Deterministic behavior
    # --------------------------------------------------------

    if deterministic:

        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = False

        torch.use_deterministic_algorithms(True)

    else:

        torch.backends.cudnn.deterministic = False

        torch.backends.cudnn.benchmark = True


# ============================================================
# DATALOADER WORKER SEED
# ============================================================

def seed_worker(
    worker_id: int,
) -> None:
    """
    Seed an individual PyTorch DataLoader worker.

    This is useful when using multiple workers for loading
    medical datasets.
    """

    del worker_id

    worker_seed = torch.initial_seed() % (2**32)

    np.random.seed(worker_seed)

    random.seed(worker_seed)


# ============================================================
# PYTORCH GENERATOR
# ============================================================

def create_generator(
    seed: int = 42,
) -> torch.Generator:
    """
    Create a seeded PyTorch random generator.

    Args:
        seed:
            Random seed.

    Returns:
        Seeded torch.Generator.
    """

    generator = torch.Generator()

    generator.manual_seed(seed)

    return generator