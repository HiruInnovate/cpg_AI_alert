import os
import uuid
from .normalizer import smart_normalize
from .storage import load_json, overwrite_json
from .embeddings import index_text
from services.logger_config import get_logger

# Initialize logger
logger = get_logger(__name__)

FEED_DIR = "data/feeds"
os.makedirs(FEED_DIR, exist_ok=True)


def save_upload(uploaded_file):
    """
    Saves uploaded file to FEED_DIR and logs the event.
    """
    try:
        fname = f"{uuid.uuid4().hex}_{uploaded_file.name}"
        path = os.path.join(FEED_DIR, fname)
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        logger.info(f"[INGESTION] Uploaded file saved: {fname} ({uploaded_file.size / 1024:.1f} KB)")
        return path
    except Exception as e:
        logger.error(f"[INGESTION] Failed to save uploaded file '{uploaded_file.name}': {e}", exc_info=True)
        raise


def process_file(path):
    """
    Normalizes and stores uploaded data based on its detected type.
    Logs each decision branch and output count for observability.
    Supports:
      - shipment_events
      - news_articles
      - social_posts
      - logistics_performance
      - generic text docs
    """
    logger.info(f"[INGESTION] Starting processing for file: {path}")

    try:
        data = smart_normalize(path)
        dtype, items = data.get("type"), data.get("items", [])
        logger.info(f"[INGESTION] Detected file type: {dtype} | Total items: {len(items)}")

        # --- Shipment Events ---
        if dtype == "shipment_events":
            logger.info(f"[INGESTION] Processing shipment events from file '{path}'")
            existing = load_json("shipments.json")
            logger.debug(f"[INGESTION] Loaded existing shipments: {len(existing)} entries")
            existing.extend(items)
            overwrite_json("shipments.json", existing)
            logger.info(f"[INGESTION] Stored {len(items)} new shipment events (Total now: {len(existing)})")
            return {"stored": len(items), "type": dtype, "namespace": "shipments.json"}

        # --- News Articles ---
        elif dtype == "news_articles":
            logger.info(f"[INGESTION] Indexing news articles into vector DB (namespace='news_data')")
            count = index_text(items, namespace="news_data")
            logger.info(f"[INGESTION] Indexed {count} news text chunks")
            return {"indexed": count, "type": dtype, "namespace": "news_data"}

        # --- Social Posts ---
        elif dtype == "social_posts":
            logger.info(f"[INGESTION] Indexing social posts into vector DB (namespace='social_data')")
            count = index_text(items, namespace="social_data")
            logger.info(f"[INGESTION] Indexed {count} social media text chunks")
            return {"indexed": count, "type": dtype, "namespace": "social_data"}

        # --- Logistics Performance Data ---
        elif dtype == "logistics_performance":
            logger.info(f"[INGESTION] Processing logistics performance data from file '{path}'")
            existing = load_json("logistics_performance.json")
            logger.debug(f"[INGESTION] Loaded existing logistics records: {len(existing)} entries")

            # Merge and overwrite
            existing.extend(items)
            overwrite_json("logistics_performance.json", existing)
            logger.info(f"[INGESTION] Stored {len(items)} new logistics performance records (Total now: {len(existing)})")

            # Index textual version for vector search
            logger.info(f"[INGESTION] Indexing logistics performance data into vector DB (namespace='logistics_data')")
            count = index_text(items, namespace="logistics_data")
            logger.info(f"[INGESTION] Indexed {count} logistics text chunks")
            return {"stored": len(items), "indexed": count, "type": dtype, "namespace": "logistics_data"}

        # --- Generic Text Docs ---
        elif dtype == "text_doc":
            logger.info(f"[INGESTION] Indexing generic text documents (namespace='misc_data')")
            count = index_text(items, namespace="misc_data")
            logger.info(f"[INGESTION] Indexed {count} misc text chunks")
            return {"indexed": count, "type": dtype, "namespace": "misc_data"}

        # --- Unknown Type ---
        else:
            logger.warning(f"[INGESTION] Unknown file type for '{path}'. Skipping storage/indexing.")
            return {"type": "unknown", "message": "File not recognized"}

    except Exception as e:
        logger.error(f"[INGESTION] Error while processing file '{path}': {e}", exc_info=True)
        return {"error": str(e), "type": "error"}
