# services/normalizer.py
import pandas as pd
import json
import os
from services.logger_config import get_logger

# Initialize logger
logger = get_logger(__name__)


def smart_normalize(path):
    """
    Detects uploaded file type: shipment, news, or social.
    Returns standardized dict with {type, items}.
    Logs detection decisions, errors, and summary of extracted records.
    """
    name = os.path.basename(path).lower()
    logger.info(f"[NORMALIZER] Starting normalization for file: {name}")

    try:
        # --- CSV files ---
        if name.endswith(".csv"):
            logger.info(f"[NORMALIZER] Detected CSV format for {name}")
            try:
                df = pd.read_csv(path)
            except Exception as e:
                logger.error(f"[NORMALIZER] Failed to read CSV file {name}: {e}", exc_info=True)
                return {"type": "error", "items": [], "message": str(e)}

            cols = [c.lower() for c in df.columns]
            logger.debug(f"[NORMALIZER] CSV Columns: {cols}")

            # Shipment data
            if "shipment_id" in cols:
                logger.info(f"[NORMALIZER] Classified as 'shipment_events' file.")
                return {"type": "shipment_events", "items": df.to_dict(orient="records")}

            # News data
            if "headline" in cols or "title" in cols:
                logger.info(f"[NORMALIZER] Classified as 'news_articles' file.")
                items = []
                for i, row in df.iterrows():
                    text = str(row.get('headline', '') or row.get('title', '')) + "\n" + str(row.get('body', ''))
                    items.append({"id": f"news_{i}", "text": text, "meta": row.to_dict()})
                logger.info(f"[NORMALIZER] Extracted {len(items)} news articles.")
                return {"type": "news_articles", "items": items}

            # Social posts
            if "post" in cols or "tweet" in cols or "content" in cols:
                logger.info(f"[NORMALIZER] Classified as 'social_posts' file.")
                items = []
                for i, row in df.iterrows():
                    text = str(row.get('post', '') or row.get('tweet', '') or row.get('content', ''))
                    items.append({"id": f"social_{i}", "text": text, "meta": row.to_dict()})
                logger.info(f"[NORMALIZER] Extracted {len(items)} social posts.")
                return {"type": "social_posts", "items": items}

            logger.warning(f"[NORMALIZER] CSV file '{name}' could not be classified by column names.")
            return {"type": "unknown", "items": []}

        # --- JSON files ---
        elif name.endswith(".json"):
            logger.info(f"[NORMALIZER] Detected JSON format for {name}")
            try:
                data = json.load(open(path, "r", encoding="utf-8"))
            except Exception as e:
                logger.error(f"[NORMALIZER] Failed to load JSON file: {e}", exc_info=True)
                return {"type": "error", "items": [], "message": str(e)}

            if not isinstance(data, list) or not data:
                logger.warning(f"[NORMALIZER] JSON file '{name}' is empty or invalid list structure.")
                return {"type": "unknown", "items": []}

            first = data[0]
            keys = list(first.keys())
            logger.debug(f"[NORMALIZER] JSON first record keys: {keys}")

            # Shipment
            if "shipment_id" in first:
                logger.info(f"[NORMALIZER] Classified JSON as 'shipment_events' file.")
                return {"type": "shipment_events", "items": data}

            # News
            if "headline" in first or "title" in first:
                logger.info(f"[NORMALIZER] Classified JSON as 'news_articles' file.")
                items = [{"id": f"news_{i}", "text": d.get("headline", "") + "\n" + d.get("body", "")} for i, d in enumerate(data)]
                logger.info(f"[NORMALIZER] Extracted {len(items)} news articles.")
                return {"type": "news_articles", "items": items}

            # Social
            if "tweet" in first or "post" in first:
                logger.info(f"[NORMALIZER] Classified JSON as 'social_posts' file.")
                items = [{"id": f"social_{i}", "text": d.get("tweet", "") or d.get("post", "")} for i, d in enumerate(data)]
                logger.info(f"[NORMALIZER] Extracted {len(items)} social posts.")
                return {"type": "social_posts", "items": items}

            logger.warning(f"[NORMALIZER] JSON file '{name}' could not be classified by keys.")
            return {"type": "unknown", "items": []}

        # --- TXT files ---
        elif name.endswith(".txt"):
            logger.info(f"[NORMALIZER] Detected TXT format for {name}")
            try:
                text = open(path, "r", encoding="utf-8", errors="ignore").read()
                logger.info(f"[NORMALIZER] Classified as 'text_doc' | Length: {len(text)} characters.")
                return {"type": "text_doc", "items": [{"id": path, "text": text}]}
            except Exception as e:
                logger.error(f"[NORMALIZER] Failed to read TXT file: {e}", exc_info=True)
                return {"type": "error", "items": [], "message": str(e)}

        # --- Unknown / unsupported file type ---
        else:
            logger.warning(f"[NORMALIZER] Unsupported file type for {name}. Skipping normalization.")
            return {"type": "unknown", "items": []}

    except Exception as e:
        logger.error(f"[NORMALIZER] Unexpected error during normalization of '{name}': {e}", exc_info=True)
        return {"type": "error", "items": [], "message": str(e)}
