import json
import os
from typing import List
from langchain.tools import tool
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from services.llm_factory import create_embedding_model

from services.logger_config import get_logger

# Load environment & logger
load_dotenv()
logger = get_logger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "guardrails.json")  # 👈 Guardrail config
SHIPMENTS_PATH = os.path.join(BASE_DIR, "data/json_store/shipments.json")
LOGISTICS_PATH = os.path.join(BASE_DIR, "data/json_store/logistics_performance.json")
CHROMA_DIR = os.path.join(BASE_DIR, "data/vector_db")
RAGAS_STORE = os.path.join(BASE_DIR, "data/json_store/agent_ragas_records.json")


# -----------------------
# Tools (langchain @tool decorated callables)
# -----------------------

@tool
def get_shipment_details(shipment_id: str) -> str:
    """
    Tool: get_shipment_details(shipment_id)

    Purpose:
        Return the shipment event record(s) for the given shipment_id from shipments.json.

    Behavior:
      - Normalizes the input and data values (strip + lower) for robust matching.
      - Tries exact match on common keys first (shipment_id, shipmentId, shipment).
      - If no exact match, tries partial substring matches (contains / endswith / startswith).
      - Logs sample records and match diagnostics for debugging.

    Input:
        shipment_id (str) e.g. "SHP1023"

    Output:
        JSON string of matching shipment records or "NOT_FOUND" / error message.
    """
    try:
        q = str(shipment_id).strip()
        logger.info(f"[TOOL][get_shipment_details] Lookup shipment_id='{q}'")

        if not os.path.exists(SHIPMENTS_PATH):
            msg = "ERROR: shipments.json not found"
            logger.error(msg)
            return msg

        with open(SHIPMENTS_PATH, "r", encoding="utf8") as f:
            data = json.load(f)

        logger.info(f"[TOOL][get_shipment_details] Loaded {len(data)} shipment records")
        # show a couple of sample records for debugging
        for i, rec in enumerate(data[:5]):
            logger.info(
                f"[TOOL][get_shipment_details] sample[{i}] keys={list(rec.keys())} values_snippet={str(list(rec.values())[:3])}")

        # helper to extract candidate id strings from a record
        def _extract_ids(rec: dict) -> List[str]:
            candidates = []
            for k in ("shipment_id", "shipmentId", "shipment", "id"):
                if k in rec and rec[k] is not None:
                    candidates.append(str(rec[k]))
            # also consider nested fields or fields present in metadata
            # fall back to any string field that looks like an SHP id (contains 'SHP' or startswith letter+digits)
            for k, v in rec.items():
                if isinstance(v, str) and v and k not in ("remarks", "location", "timestamp"):
                    candidates.append(v)

            logger.info(f"Candidates :: {candidates}")
            # normalize candidates
            return list({c.strip() for c in candidates if c is not None})

        # normalized query
        q_norm = q.lower()

        # 1) exact normalized match on common keys
        matches = []
        for rec in data:
            ids = _extract_ids(rec)
            for cid in ids:
                if cid and cid.strip().lower() in q_norm:
                    matches.append(rec)
                    break

        # 2) If none found, try partial matches (contains / endswith / startswith)
        if not matches:
            for rec in data:
                ids = _extract_ids(rec)
                joined = " ".join(ids).lower()
                if q_norm in joined or joined.endswith(q_norm) or joined.startswith(q_norm):
                    matches.append(rec)

        # 3) If still none, try numeric match if query is numeric and data may store ints
        if not matches and q.isdigit():
            for rec in data:
                ids = _extract_ids(rec)
                for cid in ids:
                    if cid.isdigit() and cid == q:
                        matches.append(rec)
                        break

        logger.info(f"[TOOL][get_shipment_details] Matches found: {len(matches)}")
        logger.debug(f"[TOOL][get_shipment_details] Matches snippet keys: {[list(m.keys()) for m in matches[:3]]}")

        if not matches:
            logger.info(f"[TOOL][get_shipment_details] No record found for '{shipment_id}'")
            return "NOT_FOUND"

        # Return JSON: single object if 1 match else list
        return json.dumps(matches if len(matches) > 1 else matches[0], default=str, indent=2)

    except Exception as e:
        logger.error(f"[TOOL][get_shipment_details] Error: {e}", exc_info=True)
        return f"ERROR: {e}"


@tool
def get_logistics_performance(carrier_name: str) -> str:
    """
    Tool: get_logistics_performance(carrier_name)

    Purpose:
        Return the carrier-level performance summary from logistics_performance.json.
    Input:
        carrier_name (str) e.g. "DTDC"
    Output:
        JSON string summary (KPIs and common delay reasons) or "NOT_FOUND" / error message.
    """
    try:
        carrier_name = str(carrier_name).strip().replace('"', '')
        logger.info(f"[TOOL][get_logistics_performance] Lookup carrier={carrier_name}")
        if not os.path.exists(LOGISTICS_PATH):
            msg = "ERROR: logistics_performance.json not found"
            logger.error(msg)
            return msg
        with open(LOGISTICS_PATH, "r", encoding="utf8") as f:
            data = json.load(f)
        for item in data:
            if str(item.get("carrier_name", "")).strip().lower() == str(carrier_name).strip().lower():
                logger.debug(f"[TOOL][get_logistics_performance] Found carrier entry for {carrier_name}")
                return json.dumps(item, default=str, indent=2)
        logger.info(f"[TOOL][get_logistics_performance] Carrier {carrier_name} NOT_FOUND")
        return "NOT_FOUND"
    except Exception as e:
        logger.error(f"[TOOL][get_logistics_performance] Error: {e}", exc_info=True)
        return f"ERROR: {e}"

# @tool
# def search_external_context(query: str) -> str:
#     """
#     Tool: search_external_context(query)
#
#     Purpose:
#         Search vector DB namespaces 'news_data' and 'social_data' for relevant snippets.
#     Input:
#         query (str) - natural language query
#     Output:
#         newline-separated list of "[namespace:source] snippet..." strings, or "NO_RELEVANT_CONTEXT_FOUND" / error
#     """
#     try:
#         logger.info(f"[TOOL][search_external_context] Running vector search for query: {query[:120]}")
#         # Import Chroma at runtime to avoid import-time dependency issues
#         try:
#             from langchain_chroma import Chroma
#         except Exception:
#             # fallback: try langchain.vectorstores.Chroma if older langchain installed
#             try:
#                 from langchain_chroma.vectorstores import Chroma
#             except Exception as e:
#                 logger.error("[TOOL][search_external_context] Chroma import failed", exc_info=True)
#                 return f"ERROR: Chroma not available ({e})"
#
#         emb = OpenAIEmbeddings(model="text-embedding-3-small")
#         namespaces = ["news_data", "social_data", "logistics_data", "misc_data"]
#         results = []
#         for ns in namespaces:
#             try:
#                 vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=emb, collection_name=ns)
#                 retriever = vectordb.as_retriever(search_kwargs={"k": 3})
#                 docs = retriever.get_relevant_documents(query)
#                 for d in docs:
#                     snippet = (d.page_content or "")[:600].replace("\n", " ")
#                     src = d.metadata.get("source", d.metadata.get("source_id", "unknown")) if getattr(d, "metadata",
#                                                                                                       None) else "unknown"
#                     results.append(f"[{ns}:{src}] {snippet}")
#             except Exception as e_ns:
#                 logger.warning(f"[TOOL][search_external_context] Namespace '{ns}' search error: {e_ns}")
#                 # continue to next namespace
#                 continue
#
#         if not results:
#             logger.info("[TOOL][search_external_context] No relevant context found")
#             return "NO_RELEVANT_CONTEXT_FOUND"
#         logger.debug(f"[TOOL][search_external_context] Retrieved {len(results)} snippets")
#         return "\n".join(results)
#     except Exception as e:
#         logger.error(f"[TOOL][search_external_context] Error: {e}", exc_info=True)
#         return f"ERROR: {e}"

@tool
def search_external_context(query: str) -> str:
    """
       Tool: search_external_context(query)

    Purpose:
        Search vector DB namespaces 'news_data' and 'social_data' for relevant snippets.
    Input:
        query (str) - natural language query
    Output:
        newline-separated list of "[namespace:source] snippet..." strings, or "NO_RELEVANT_CONTEXT_FOUND" / error

    """
    try:
        logger.info(f"[TOOL][search_external_context] Running vector search for query: {query[:100]}")

        try:
            from langchain_chroma import Chroma
        except Exception:
            try:
                from langchain_chroma.vectorstores import Chroma
            except Exception as e:
                logger.error("[TOOL][search_external_context] ❌ Chroma import failed", exc_info=True)
                return f"ERROR: Chroma not available ({e})"

        # ✅ Use LLM factory for environment-aware embeddings
        emb = create_embedding_model()

        namespaces = ["news_data", "social_data", "logistics_data", "misc_data"]
        results = []

        for ns in namespaces:
            try:
                vectordb = Chroma(
                    persist_directory=CHROMA_DIR,
                    embedding_function=emb,
                    collection_name=ns
                )
                retriever = vectordb.as_retriever(search_kwargs={"k": 3})
                docs = retriever.get_relevant_documents(query)

                for d in docs:
                    snippet = (d.page_content or "")[:600].replace("\n", " ").strip()
                    src = (
                            d.metadata.get("source")
                            or d.metadata.get("source_id")
                            or "unknown"
                    ) if getattr(d, "metadata", None) else "unknown"
                    results.append(f"[{ns}:{src}] {snippet}")
            except Exception as e_ns:
                logger.warning(f"[TOOL][search_external_context] Namespace '{ns}' search error: {e_ns}")
                continue

        if not results:
            logger.info("[TOOL][search_external_context] No relevant context found")
            return "NO_RELEVANT_CONTEXT_FOUND"

        logger.info(f"[TOOL][search_external_context] ✅ Retrieved {len(results)} snippets")
        return "\n".join(results)

    except Exception as e:
        logger.error(f"[TOOL][search_external_context] Error: {e}", exc_info=True)
        return f"ERROR: {e}"
