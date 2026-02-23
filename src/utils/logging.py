"""Rich-powered logging helpers for CLI output."""

from __future__ import annotations

import logging
import sys

from rich.console import Console
from rich.logging import RichHandler

console = Console()

_CONFIGURED = False


def get_logger(name: str = "trading_bot", level: int = logging.INFO) -> logging.Logger:
    """Return a logger configured with Rich handler (configured once)."""
    global _CONFIGURED
    logger = logging.getLogger(name)
    if not _CONFIGURED:
        logger.setLevel(level)
        handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            show_path=False,
            markup=True,
        )
        handler.setLevel(level)
        fmt = logging.Formatter("%(message)s", datefmt="[%X]")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        _CONFIGURED = True
    return logger
