# services/storage.py
import json
import os
import threading
from services.logger_config import get_logger

# Initialize logger
logger = get_logger(__name__)

# Ensure base directory exists
os.makedirs("data/json_store", exist_ok=True)
_lock = threading.Lock()


def _path(fn):
    """Resolve relative file path for JSON storage."""
    return os.path.join("data/json_store", fn)


def load_json(fn):
    """
    Safely loads JSON file from disk.
    Returns an empty list if the file doesn't exist.
    """
    p = _path(fn)
    try:
        if not os.path.exists(p):
            logger.warning(f"[STORAGE] File not found: {p}. Returning empty list.")
            return []

        with open(p, "r", encoding="utf8") as f:
            data = json.load(f)
            logger.debug(f"[STORAGE] Loaded {len(data) if isinstance(data, list) else 'object'} items from {fn}.")
            return data
    except json.JSONDecodeError as e:
        logger.error(f"[STORAGE] JSON decode error in file {fn}: {e}", exc_info=True)
        return []
    except Exception as e:
        logger.error(f"[STORAGE] Error loading file {fn}: {e}", exc_info=True)
        return []


def save_json(fn, data):
    """
    Overwrites an entire JSON file safely with thread locking.
    """
    p = _path(fn)
    try:
        with _lock:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf8") as f:
                json.dump(data, f, indent=2, default=str)
            logger.info(f"[STORAGE] Saved {len(data) if isinstance(data, list) else 'object'} items to {fn}.")
    except Exception as e:
        logger.error(f"[STORAGE] Failed to save JSON file {fn}: {e}", exc_info=True)


def append_json(fn, obj):
    """
    Appends a single object to a JSON list file.
    """
    try:
        logger.debug(f"[STORAGE] Appending new record to {fn}.")
        items = load_json(fn)

        if not isinstance(items, list):
            logger.warning(f"[STORAGE] File {fn} does not contain a list. Overwriting with new list.")
            items = []

        items.append(obj)
        save_json(fn, items)
        logger.info(f"[STORAGE] Appended new record to {fn} (total: {len(items)} records).")

    except Exception as e:
        logger.error(f"[STORAGE] Error appending to file {fn}: {e}", exc_info=True)


def overwrite_json(fn, obj):
    """
    Replaces file content completely with provided object/list.
    """
    try:
        logger.info(f"[STORAGE] Overwriting file {fn} with new data.")
        save_json(fn, obj)
    except Exception as e:
        logger.error(f"[STORAGE] Error overwriting file {fn}: {e}", exc_info=True)
