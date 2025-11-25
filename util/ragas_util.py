import json
import os
from datetime import datetime

from dotenv import load_dotenv

from services.logger_config import get_logger

# Load environment & logger
load_dotenv()
logger = get_logger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RAGAS_STORE = os.path.join(BASE_DIR, "data/json_store/agent_ragas_records.json")


# Utility to record agent runs
def record_agent_run_for_ragas(agent_name, alert, question, contexts, answer):
    try:
        os.makedirs(os.path.dirname(RAGAS_STORE), exist_ok=True)
        data = json.load(open(RAGAS_STORE, "r", encoding="utf8")) if os.path.exists(RAGAS_STORE) else []
        data.append({
            "timestamp": datetime.now().isoformat(),
            "agent_name": agent_name,
            "alert_id": alert.get("alert_id"),
            "question": question,
            "contexts": contexts,
            "generated_answer": answer
        })
        json.dump(data, open(RAGAS_STORE, "w"), indent=2)
        logger.info(f"[RAGAS_RECORD] Logged {agent_name} for alert {alert.get('alert_id')}")
    except Exception as e:
        logger.error(f"[RAGAS_RECORD] Error saving: {e}", exc_info=True)
