# services/ragas_integration.py
"""
RAGAS Evaluation Integration Module.

Provides an interface to evaluate agent runs using the RAGAS library if available.
Falls back to heuristic grounding and hallucination scoring when RAGAS is not installed.
"""

from services.logger_config import get_logger

# Initialize logger
logger = get_logger(__name__)

try:
    from ragas import evaluator
    RAGAS_AVAILABLE = True
    logger.info("[RAGAS] RAGAS library successfully imported.")
except Exception as e:
    RAGAS_AVAILABLE = False
    logger.warning(f"[RAGAS] RAGAS library not available. Using heuristic evaluation. ({e})")


def run_ragas_evaluation(run_obj: dict):
    """
    Evaluate an agent run using RAGAS metrics or heuristic fallback.

    Args:
        run_obj (dict): agent run record containing input, retrieved_docs/context_docs, and output.

    Returns:
        dict: evaluation metrics including grounding_score, hallucination_rate, etc.
    """
    logger.info("[RAGAS] Starting evaluation of agent run.")
    run_id = run_obj.get("run_id", "unknown")
    agent_name = run_obj.get("agent", "unknown")
    logger.debug(f"[RAGAS] Evaluating run_id={run_id} | agent={agent_name}")

    # ---- CASE 1: Official RAGAS Evaluation ----
    if RAGAS_AVAILABLE:
        try:
            logger.info(f"[RAGAS] Running official RAGAS evaluation for agent '{agent_name}'")
            ev = evaluator.Evaluator()
            res = ev.evaluate(run_obj)
            logger.info(f"[RAGAS] Evaluation complete for run {run_id}: {res}")
            return res
        except Exception as e:
            logger.error(f"[RAGAS] Error during RAGAS evaluation: {e}", exc_info=True)
            return {
                "error": str(e),
                "notes": "RAGAS evaluation failed; fallback to heuristic scoring."
            }

    # ---- CASE 2: Heuristic Fallback ----
    try:
        output_text = run_obj.get("raw_output", "") or ""
        retrieved = run_obj.get("context_docs", []) or run_obj.get("retrieved_docs", [])
        grounded = 0
        logger.debug(f"[RAGAS] Performing heuristic grounding check for {len(retrieved)} retrieved docs.")

        for rid in retrieved:
            if str(rid).lower() in output_text.lower():
                grounded += 1

        grounding_score = (grounded / max(1, len(retrieved))) * 100
        hallucination_rate = 100 - grounding_score

        result = {
            "grounding_score": round(grounding_score, 2),
            "hallucination_rate": round(hallucination_rate, 2),
            "notes": "approximate heuristic (RAGAS not available)"
        }

        logger.info(f"[RAGAS] Heuristic evaluation complete for run {run_id}: {result}")
        return result

    except Exception as e:
        logger.error(f"[RAGAS] Error in heuristic evaluation: {e}", exc_info=True)
        return {"error": str(e), "notes": "Evaluation failed due to internal error."}
