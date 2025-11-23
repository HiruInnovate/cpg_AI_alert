import logging, os
from logging.handlers import RotatingFileHandler

os.makedirs("logs", exist_ok=True)
LOG_PATH = "logs/app.log"

def get_logger(name: str):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        fh = RotatingFileHandler(LOG_PATH, maxBytes=1_000_000, backupCount=3,delay=True)
        fh.setFormatter(fmt)
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger
