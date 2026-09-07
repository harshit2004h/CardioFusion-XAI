"""
Logging utilities for CardioFusion-XAI.
"""

import logging
from pathlib import Path


# ============================================================
# DEFAULT FORMAT
# ============================================================

DEFAULT_FORMAT = (
    "%(asctime)s | "
    "%(levelname)s | "
    "%(name)s | "
    "%(message)s"
)


# ============================================================
# LOGGER CREATION
# ============================================================

def get_logger(
    name: str = "cardiofusion",
    level: str = "INFO",
    log_file: str | Path | None = None,
) -> logging.Logger:
    """
    Create and configure a project logger.

    Args:
        name:
            Logger name.

        level:
            Logging level.

        log_file:
            Optional file path where logs should be stored.

    Returns:
        Configured logging.Logger.
    """

    logger = logging.getLogger(name)

    logger.setLevel(level.upper())

    logger.propagate = False

    # --------------------------------------------------------
    # Prevent duplicate handlers
    # --------------------------------------------------------

    if logger.handlers:

        return logger

    formatter = logging.Formatter(
        DEFAULT_FORMAT
    )

    # --------------------------------------------------------
    # Console handler
    # --------------------------------------------------------

    console_handler = logging.StreamHandler()

    console_handler.setFormatter(
        formatter
    )

    logger.addHandler(
        console_handler
    )

    # --------------------------------------------------------
    # File handler
    # --------------------------------------------------------

    if log_file is not None:

        log_path = Path(log_file)

        log_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        file_handler = logging.FileHandler(
            log_path,
            encoding="utf-8",
        )

        file_handler.setFormatter(
            formatter
        )

        logger.addHandler(
            file_handler
        )

    return logger


# ============================================================
# ROOT PROJECT LOGGER
# ============================================================

def configure_root_logger(
    level: str = "INFO",
    log_file: str | Path | None = None,
) -> logging.Logger:
    """
    Configure the main CardioFusion-XAI logger.
    """

    return get_logger(
        name="cardiofusion",
        level=level,
        log_file=log_file,
    )