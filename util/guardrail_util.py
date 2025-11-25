# Load Guardrails Dynamically
# ---------------------------
import json
import os
from typing import Dict, Any

from services.logger_config import get_logger

logger = get_logger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "guardrails.json")  # 👈 Guardrail config


def load_guardrails() -> Dict[str, Any]:
    """Load configurable guardrails."""
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf8") as f:
                guard = json.load(f)
            return guard
        else:
            logger.warning("[GUARDRAIL] Missing config, loading defaults")
            return {
                "banned_phrases": ["blame vendor", "ignore delay"],
                "min_confidence_for_auto_action": 70,
                "business_rules": [
                    "Ensure recommendations are grounded in shipment or logistics data.",
                    "Never hallucinate unsupported causes."
                ]
            }
    except Exception as e:
        logger.error(f"[GUARDRAIL] Load failed: {e}")
        return {}
