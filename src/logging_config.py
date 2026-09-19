from pathlib import Path

from loguru import logger


def configure_logging(log_directory: str = "logs") -> None:
    """Configure console and file logging."""
    Path(log_directory).mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(
        sink=lambda message: print(message, end=""),
        level="INFO",
        enqueue=True,
    )
    logger.add(
        f"{log_directory}/application.log",
        level="INFO",
        rotation="10 MB",
        retention="14 days",
        enqueue=True,
    )
from pathlib import Path

from loguru import logger


def configure_logging(log_directory: str = "logs") -> None:
    """Configure console and file logging."""
    Path(log_directory).mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(
        sink=lambda message: print(message, end=""),
        level="INFO",
        enqueue=True,
    )
    logger.add(
        f"{log_directory}/application.log",
        level="INFO",
        rotation="10 MB",
        retention="14 days",
        enqueue=True,
    )
