import os, json, traceback
from datetime import datetime
from typing import Any, Dict, List, Union, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool, render_text_description
from langchain_core.agents import AgentFinish, AgentAction
from langchain_classic.agents.format_scratchpad import format_log_to_str
from langchain_classic.agents.output_parsers import ReActSingleInputOutputParser
from services.logger_config import get_logger
from langchain.tools import tool
import httpx, re

# Load environment & logger
load_dotenv()
logger = get_logger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "guardrails.json")   # 👈 Guardrail config
SHIPMENTS_PATH = os.path.join(BASE_DIR, "data/json_store/shipments.json")
LOGISTICS_PATH = os.path.join(BASE_DIR, "data/json_store/logistics_performance.json")
CHROMA_DIR = os.path.join(BASE_DIR, "data/vector_db")
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


# ---------------------------
# Load Guardrails Dynamically
# ---------------------------
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
        logger.info(f"[TOOL][search_external_context] Running vector search for query: {query[:120]}")
        # Import Chroma at runtime to avoid import-time dependency issues
        try:
            from langchain_chroma import Chroma
        except Exception:
            # fallback: try langchain.vectorstores.Chroma if older langchain installed
            try:
                from langchain_chroma.vectorstores import Chroma
            except Exception as e:
                logger.error("[TOOL][search_external_context] Chroma import failed", exc_info=True)
                return f"ERROR: Chroma not available ({e})"

        emb = OpenAIEmbeddings(model="text-embedding-3-small")
        namespaces = ["news_data", "social_data", "logistics_data", "misc_data"]
        results = []
        for ns in namespaces:
            try:
                vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=emb, collection_name=ns)
                retriever = vectordb.as_retriever(search_kwargs={"k": 3})
                docs = retriever.get_relevant_documents(query)
                for d in docs:
                    snippet = (d.page_content or "")[:600].replace("\n", " ")
                    src = d.metadata.get("source", d.metadata.get("source_id", "unknown")) if getattr(d, "metadata",
                                                                                                      None) else "unknown"
                    results.append(f"[{ns}:{src}] {snippet}")
            except Exception as e_ns:
                logger.warning(f"[TOOL][search_external_context] Namespace '{ns}' search error: {e_ns}")
                # continue to next namespace
                continue

        if not results:
            logger.info("[TOOL][search_external_context] No relevant context found")
            return "NO_RELEVANT_CONTEXT_FOUND"
        logger.debug(f"[TOOL][search_external_context] Retrieved {len(results)} snippets")
        return "\n".join(results)
    except Exception as e:
        logger.error(f"[TOOL][search_external_context] Error: {e}", exc_info=True)
        return f"ERROR: {e}"


# -----------------------
# Utility: find tool by name (used by the ReAct loop)
# -----------------------
def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name '{tool_name}' not found")







# ---------------------------
# Build RCA Agent
# ---------------------------
def build_rca_agent_instance(tools: List[Tool], llm_model="gpt-4o-mini", temperature=0.2):
    guard = load_guardrails()  # 👈 Inject latest configuration

    guardrail_prompt = f"""
⚙️ **Supply Chain Guardrails & Compliance:**
- Minimum confidence for recommendations: {guard.get('min_confidence_for_auto_action', 70)}%
- Banned phrases: {', '.join(guard.get('banned_phrases', []))}
- Business rules:
  {"; ".join(guard.get('business_rules', []))}

Ensure all reasoning, causes, and recommendations strictly comply with these guardrails.
If an action violates a rule, explicitly mention "⚠️ REJECTED_BY_GUARDRAIL" and propose an alternative.
"""

    # Updated ReAct prompt
    template = """
    You are a Supply Chain Root Cause Analysis (RCA) expert.
    You are investigating an alert about shipment disruption.

    You have access to the following tools:
    {tools}

    {guardrail_prompt}
    When you use a tool, please follow this format:

    Question: the input alert details or question
    Thought: think about what to check next
    Action: one of [{tool_names}]
    Action Input: the input to the tool
    Observation: result of the tool
    ... (repeat Thought/Action/Action Input/Observation as required)
    Thought: I now know the root cause
    Final Answer: Provide an elaborate answer in a JSON object with keys:
      - causes: list of {{label, explanation}}
      - confidence: float (0-1)
      - summary: brief 1-2 line summary
      - recommendations: list of suggested actions {{id, action, eta_hours, confidence_est}}
      - evidence: list of references (snippet identifiers or short quotes)

    Be thorough and ground reasoning to the tool observations.

    Begin!

    Question: {input}
    Thought: {agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        guardrail_prompt= guardrail_prompt,
        tool_names=", ".join([t.name for t in tools])
    )

    llm = ChatOpenAI(
        model=llm_model,
        temperature=temperature,
        stop=["\nObservation", "Observation"],
        http_client=httpx.Client(verify=False)
    )

    return (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
        }
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )


# ---------------------------
# RCA Runner
# ---------------------------
def run_rca_agent(alert: Dict[str, Any], max_steps=8, llm_model="gpt-4o-mini", temperature=0.2, tools=None):
    """Main RCA run loop with guardrail injection."""

    run_trace, agent_scratchpad, retrieved_contexts = [], [], []
    final_text, final_json, raw_agent_return = "", None, None
    tools = tools or [get_shipment_details, get_logistics_performance, search_external_context]

    try:
        agent = build_rca_agent_instance(tools, llm_model=llm_model, temperature=temperature)
        question = f"What is the cause of this alert and recommendations to mitigate? {json.dumps(alert)}"

        steps = 0
        while steps < max_steps:
            agent_step = agent.invoke({"input": question, "agent_scratchpad": agent_scratchpad})
            if isinstance(agent_step, AgentAction):
                tool_name, tool_input = agent_step.tool, agent_step.tool_input
                try:
                    tool_obj = next(t for t in tools if t.name == tool_name)
                    observation = tool_obj.func(str(tool_input))
                except Exception as e:
                    observation = f"ERROR: {e}"

                run_trace.append({
                    "step": steps + 1,
                    "tool": tool_name,
                    "tool_input": tool_input,
                    "observation": observation
                })
                agent_scratchpad.append((agent_step, observation))
                retrieved_contexts.append(str(observation))
                steps += 1
                continue

            if isinstance(agent_step, AgentFinish):
                final_text = agent_step.return_values.get("output", "")
                try:
                    match = re.search(r'(\{.*\}|\[.*\])', final_text, re.DOTALL)
                    if match:
                        final_json = json.loads(match.group(0))
                except Exception:
                    pass

                record_agent_run_for_ragas(
                    agent_name="RCA_Agent",
                    alert=alert,
                    question=question,
                    contexts=retrieved_contexts,
                    answer=final_text
                )
                return {
                    "steps": run_trace,
                    "final_answer": final_text,
                    "final_json": final_json,
                    "logs": format_log_to_str(agent_scratchpad)
                }

        return {
            "steps": run_trace,
            "final_answer": final_text,
            "final_json": final_json,
            "logs": format_log_to_str(agent_scratchpad),
            "note": "max_steps_reached"
        }

    except Exception as e:
        logger.error(f"[RCA_AGENT] Error: {e}")
        return {"error": str(e), "steps": run_trace, "logs": format_log_to_str(agent_scratchpad)}
