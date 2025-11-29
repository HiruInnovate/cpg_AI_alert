import logging
import os
from logging.handlers import RotatingFileHandler
import sys

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "app.log")


class SafeRotatingFileHandler(RotatingFileHandler):
    """Windows-safe rotating file handler that gracefully handles file-in-use errors."""

    def doRollover(self):
        try:
            super().doRollover()
        except PermissionError:
            import shutil
            backup = f"{self.baseFilename}.1"
            try:
                # Fallback: copy & truncate if rename fails (Windows file lock)
                shutil.copy(self.baseFilename, backup)
                open(self.baseFilename, 'w').close()
                print(f"[LOGGER] Fallback rollover: {backup}")
            except Exception as e:
                print(f"[LOGGER] Rollover failed: {e}")


def get_logger(name: str) -> logging.Logger:
    """
    Create or retrieve a logger instance with:
      - Windows-safe log rotation
      - Console + file handlers
      - Prevents duplicate handlers on Streamlit reload
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers (important for Streamlit hot-reload)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    # ---- File Handler ----
    file_handler = SafeRotatingFileHandler(
        LOG_PATH,
        maxBytes=1_000_000,  # 1 MB
        backupCount=3,
        delay=True,          # Delay open until first log write (reduces lock risk)
        encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    # ---- Console Handler ----
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # ---- Propagation ----
    logger.propagate = False

    logger.info(f"✅ Logger initialized for [{name}] at {LOG_PATH}")
    return logger
