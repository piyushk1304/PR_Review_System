# logger_config.py
import sys
from loguru import logger

logger.remove()
logger.add(
    sys.stderr,          # stderr — uvicorn never swallows stderr
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
    colorize=False,
    enqueue=True,        # thread-safe queue for background threads
    backtrace=True,
    diagnose=True,
)