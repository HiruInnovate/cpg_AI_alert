# agents/monitor_agent.py
import random
import datetime
from services.storage import load_json, append_json
from agents.langchain_agents import orchestrate_alert
from services.logger_config import get_logger

# Initialize logger
logger = get_logger(__name__)


def run_monitor_and_orchestrate(max_events=20, verbose=False):
    """
    Scans shipment events for potential disruptions and orchestrates multi-agent analysis.
    Logs every major decision and orchestration step for observability.
    """
    logger.info(f"[MONITOR] Starting supply chain monitor. Checking up to {max_events} shipment events.")

    try:
        shipments = load_json("shipments.json")
        logger.info(f"[MONITOR] Loaded {len(shipments)} total shipments from store.")
    except Exception as e:
        logger.error(f"[MONITOR] Failed to load shipments.json: {e}", exc_info=True)
        return []

    created_alerts = []

    for idx, e in enumerate(shipments[:max_events]):
        shipment_id = e.get("shipment_id")
        event_type = e.get("event_type")
        location = e.get("location")

        logger.debug(f"[MONITOR] Processing shipment {shipment_id} | Event: {event_type} | Location: {location}")

        try:
            # Demo rule: flag delayed or random events as alerts
            if event_type == "Delayed" or random.random() > 0.92:
                severity = "High" if event_type == "Delayed" else random.choice(["Low", "Medium"])
                alert = {
                    "alert_id": f"ALRT_{random.randint(10000, 99999)}",
                    "shipment_id": shipment_id,
                    "summary": f"Detected delay for shipment {shipment_id} at {location}",
                    "created_at": datetime.datetime.utcnow().isoformat(),
                    "severity": severity,
                    "status": "Active"
                }

                append_json("alerts.json", alert)
                created_alerts.append(alert)

                logger.info(f"[MONITOR] Created new alert {alert['alert_id']} | Severity: {severity} | Shipment: {shipment_id}")

                # Run orchestration for this alert
                logger.info(f"[MONITOR] Starting orchestration for alert {alert['alert_id']}.")
                try:
                    orchestrate_alert(alert, verbose=verbose)
                    logger.info(f"[MONITOR] Completed orchestration for alert {alert['alert_id']}.")
                except Exception as e:
                    logger.error(f"[MONITOR] Orchestration failed for alert {alert['alert_id']}: {e}", exc_info=True)
            else:
                logger.debug(f"[MONITOR] No disruption detected for shipment {shipment_id}.")
        except Exception as e:
            logger.error(f"[MONITOR] Error processing shipment {shipment_id}: {e}", exc_info=True)
            continue

    logger.info(f"[MONITOR] Monitoring complete. Total alerts created: {len(created_alerts)}")
    return created_alerts
