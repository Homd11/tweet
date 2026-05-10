import sys
from pathlib import Path
from loguru import logger
from src.config.settings import LOGS_DIR, LOG_CONFIG


def setup_logger(name: str = __name__):
    logger.remove()
    logger.add(
        sys.stdout,
        level=LOG_CONFIG["level"],
        format=LOG_CONFIG["format"],
        colorize=True,
    )
    log_file = LOGS_DIR / f"{name}.log"
    logger.add(
        str(log_file),
        rotation=LOG_CONFIG["rotation"],
        retention=LOG_CONFIG["retention"],
        level=LOG_CONFIG["level"],
        format=LOG_CONFIG["format"],
        compression="zip",
    )
    return logger


app_logger = setup_logger("tweet_sentiment")