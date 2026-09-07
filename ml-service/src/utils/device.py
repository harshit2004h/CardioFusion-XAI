"""
PyTorch device management utilities.

Handles CPU/GPU selection and provides information about
the available CUDA device.
"""

from typing import Any

import torch


# ============================================================
# DEVICE SELECTION
# ============================================================

def get_device(
    device_type: str = "auto",
) -> torch.device:
    """
    Select the appropriate PyTorch device.

    Args:
        device_type:
            "auto"  -> Use CUDA if available, otherwise CPU.
            "cuda"  -> Require CUDA.
            "cpu"   -> Force CPU.

    Returns:
        torch.device

    Raises:
        RuntimeError:
            If CUDA is explicitly requested but unavailable.

        ValueError:
            If an unsupported device type is provided.
    """

    device_type = device_type.lower()

    # --------------------------------------------------------
    # Automatic selection
    # --------------------------------------------------------

    if device_type == "auto":

        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")

    # --------------------------------------------------------
    # CUDA
    # --------------------------------------------------------

    if device_type == "cuda":

        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA was requested, but PyTorch cannot "
                "access an NVIDIA GPU."
            )

        return torch.device("cuda")

    # --------------------------------------------------------
    # CPU
    # --------------------------------------------------------

    if device_type == "cpu":

        return torch.device("cpu")

    # --------------------------------------------------------
    # Invalid value
    # --------------------------------------------------------

    raise ValueError(
        f"Unsupported device type: '{device_type}'. "
        "Expected 'auto', 'cuda', or 'cpu'."
    )


# ============================================================
# DEVICE INFORMATION
# ============================================================

def get_device_info() -> dict[str, Any]:
    """
    Return information about the current PyTorch environment.

    Returns:
        Dictionary containing PyTorch, CUDA and GPU information.
    """

    cuda_available = torch.cuda.is_available()

    info: dict[str, Any] = {
        "pytorch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
    }

    if cuda_available:

        info["device_name"] = torch.cuda.get_device_name(0)

        info["device_index"] = 0

        info["device_capability"] = (
            torch.cuda.get_device_capability(0)
        )

        total_memory = torch.cuda.get_device_properties(
            0
        ).total_memory

        info["total_memory_gb"] = (
            total_memory / (1024**3)
        )

    else:

        info["device_name"] = "CPU"

        info["device_index"] = None

        info["device_capability"] = None

        info["total_memory_gb"] = None

    return info


# ============================================================
# PRINT DEVICE INFORMATION
# ============================================================

def print_device_info() -> None:
    """
    Print a readable summary of the PyTorch compute environment.
    """

    info = get_device_info()

    print("=" * 60)
    print("CardioFusion-XAI Compute Environment")
    print("=" * 60)

    print(
        f"PyTorch version : "
        f"{info['pytorch_version']}"
    )

    print(
        f"CUDA available  : "
        f"{info['cuda_available']}"
    )

    print(
        f"CUDA version    : "
        f"{info['cuda_version']}"
    )

    print(
        f"Device count    : "
        f"{info['device_count']}"
    )

    print(
        f"Device name     : "
        f"{info['device_name']}"
    )

    if info["total_memory_gb"] is not None:

        print(
            f"GPU memory      : "
            f"{info['total_memory_gb']:.2f} GB"
        )

    print("=" * 60)