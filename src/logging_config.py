from __future__ import annotations

import logging
import sys
from typing import TextIO

RESET = "\033[0m"
LEVEL_COLORS = {
    logging.DEBUG: "\033[90m",
    logging.INFO: "\033[32m",
    logging.WARNING: "\033[33m",
    logging.ERROR: "\033[31m",
    logging.CRITICAL: "\033[1;31m",
}


class ColorFormatter(logging.Formatter):
    def __init__(self, *args: object, use_colors: bool = True, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        original_levelname = record.levelname
        if self.use_colors:
            color = LEVEL_COLORS.get(record.levelno)
            if color is not None:
                record.levelname = f"{color}{record.levelname}{RESET}"
        try:
            return super().format(record)
        finally:
            record.levelname = original_levelname


def configure_logging(
    level: str | int = logging.INFO,
    *,
    use_colors: bool | None = None,
    stream: TextIO | None = None,
) -> None:
    log_stream = stream or sys.stderr
    color_enabled = log_stream.isatty() if use_colors is None else use_colors

    handler = logging.StreamHandler(log_stream)
    handler.setFormatter(
        ColorFormatter(
            fmt="%(asctime)s %(levelname)-18s [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
            use_colors=color_enabled,
        )
    )

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)
