"""
LangChain Agent executors and Tools for the CPG Supply Chain Disruption app.
"""

import os
import json
import re
import datetime
from typing import Dict, Any, Optional

from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool

# Local services
from services.logger_config import get_logger
from services.storage import load_json, append_json
from services.mailer import send_mail
from services.ragas_integration import run_ragas_evaluation

# Embeddings helpers
try:
    from services.embeddings import get_retriever, index_text, get_retrievers
except Exception:
    try:
        from services.embeddings import get_retriever, index_text

        get_retrievers = None
    except Exception:
        get_retriever = None
        index_text = None
        get_retrievers = None

# Initialize logger
logger = get_logger(__name__)


# ----------------------- Helper JSON and LLM utilities ----------------------- #

def _safe_json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, default=str)
    except Exception as e:
        logger.warning(f"[UTILS] JSON serialization failed: {e}")
        return str(obj)


# ----------------------- LangChain Tool Wrappers ----------------------- #

def tool_retrieve_multi(query: str) -> str:
    """
    Multi-source retrieval tool for contextual grounding.

    Purpose:
        Retrieve relevant snippets from all available Chroma vector namespaces:
        `news_data`, `social_data`, and `misc_data`. This tool allows the agent
        to gather real-world evidence (e.g., news or social signals) to support
        root-cause analysis or recommendation generation.

    Parameters:
        query (str): The natural language question or search phrase.
                     Example: "Why are shipments delayed in North region?"

    Returns:
        str: A newline-separated string of contextual snippets formatted as:
             "[namespace:source_id] snippet text"
             or "NO_CONTEXT_FOUND" / "RETRIEVE_ERROR:<msg>" if retrieval fails.

    Example:
        Input  -> "supplier disruption"
        Output -> "[news_data:news_23] Supplier strike in Chennai..."
    """
    logger.info(f"[TOOL:RETRIEVE] Starting retrieval for query: {query[:100]}...")
    parts = []
    try:
        if callable(get_retrievers):
            retrievers = get_retrievers()
            logger.debug(f"[TOOL:RETRIEVE] Retrieved namespaces: {list(retrievers.keys())}")
            for ns, retr in retrievers.items():
                docs = retr.get_relevant_documents(query)
                for d in docs:
                    src = d.metadata.get("source") if d.metadata else "doc_unknown"
                    snippet = d.page_content.replace("\n", " ").strip()[:800]
                    parts.append(f"[{ns}:{src}] {snippet}")
        else:
            for ns in ["news_data", "social_data", "misc_data"]:
                if callable(get_retriever):
                    retr = get_retriever(namespace=ns, k=5)
                    docs = retr.get_relevant_documents(query)
                    for d in docs:
                        src = d.metadata.get("source") if d.metadata else "doc_unknown"
                        snippet = d.page_content.replace("\n", " ").strip()[:800]
                        parts.append(f"[{ns}:{src}] {snippet}")
        if not parts:
            logger.warning("[TOOL:RETRIEVE] No documents found for query.")
            return "NO_CONTEXT_FOUND"
        logger.info(f"[TOOL:RETRIEVE] Retrieved {len(parts)} relevant snippets.")
        return "\n\n".join(parts)
    except Exception as e:
        logger.error(f"[TOOL:RETRIEVE] Error retrieving context: {e}", exc_info=True)
        return f"RETRIEVE_ERROR: {e}"


def tool_storage(query: str) -> str:
    """
    Local JSON storage query and writer tool.

    Purpose:
        Interact with internal JSON-based data stores used by this app for
        shipments, alerts, and agent run logs.

    Supported Commands:
        - "get_shipment <shipment_id>"
        - "list_alerts"
        - "list_agent_runs"
        - "get_agent_run <run_id>"
        - "write_alert {json_payload}"

    Parameters:
        query (str): Command string, optionally followed by an argument or JSON payload.

    Returns:
        str: A JSON string response containing the requested data or a status message.

    Example:
        Input  -> "get_shipment SHP1023"
        Output -> {"shipment_id": "SHP1023", "status": "Delayed", ...}

        Input  -> "write_alert {'alert_id':'A102','summary':'Supplier delay detected'}"
        Output -> {"status": "ok", "written": "A102"}
    """
    logger.debug(f"[TOOL:STORAGE] Received query: {query}")
    try:
        tokens = query.strip().split(maxsplit=1)
        cmd = tokens[0].lower() if tokens else None
        arg = tokens[1] if len(tokens) > 1 else None

        logger.info(f"[TOOL:STORAGE] Executing command: {cmd}")
        if not tokens:
            res = {"shipments": len(load_json("shipments.json")), "alerts": len(load_json("alerts.json"))}
            return _safe_json_dumps(res)

        if cmd == "get_shipment" and arg:
            sid = arg.strip()
            shipments = load_json("shipments.json")
            result = next((s for s in shipments if str(s.get("shipment_id")) == sid), {})
            return _safe_json_dumps(result)

        if cmd == "list_alerts":
            return _safe_json_dumps(load_json("alerts.json"))

        if cmd == "list_agent_runs":
            return _safe_json_dumps(load_json("agent_runs.json"))

        if cmd == "get_agent_run" and arg:
            rid = arg.strip()
            runs = load_json("agent_runs.json")
            result = next((r for r in runs if r.get("run_id") == rid), {})
            return _safe_json_dumps(result)

        if cmd == "write_alert" and arg:
            try:
                payload = json.loads(arg)
                append_json("alerts.json", payload)
                logger.info(f"[TOOL:STORAGE] New alert written: {payload.get('alert_id')}")
                return _safe_json_dumps({"status": "ok", "written": payload.get("alert_id")})
            except Exception as e:
                logger.error(f"[TOOL:STORAGE] Invalid JSON payload: {e}", exc_info=True)
                return f"WRITE_ERROR: invalid json payload - {e}"

        logger.warning(f"[TOOL:STORAGE] Unknown command: {cmd}")
        return _safe_json_dumps({"error": "unknown_command", "cmd": cmd})
    except Exception as e:
        logger.error(f"[TOOL:STORAGE] Storage tool error: {e}", exc_info=True)
        return f"STORAGE_ERROR: {e}"


def tool_index_text(payload: str) -> str:
    """
    Document indexing tool for Chroma vector DB.

    Purpose:
        Index text documents into a specific Chroma namespace for semantic retrieval.

    Parameters:
        payload (str): JSON string containing:
            {
                "namespace": "news_data" | "social_data" | "misc_data",
                "docs": [{"id": "doc123", "text": "Document content"}, ...]
            }

    Returns:
        str: Human-readable message with count of text chunks indexed or error message.

    Example:
        Input  -> '{"namespace": "news_data", "docs": [{"id": "n1", "text": "Factory delay in Gujarat"}]}'
        Output -> "Indexed 2 chunks into namespace 'news_data'"
    """
    logger.info("[TOOL:INDEX] Starting indexing process.")
    try:
        data = json.loads(payload)
        ns = data.get("namespace", "news_data")
        docs = data.get("docs", [])
        if not isinstance(docs, list):
            return "INDEX_ERROR: 'docs' must be a list"
        count = index_text(docs, namespace=ns)
        logger.info(f"[TOOL:INDEX] Indexed {count} chunks into namespace '{ns}'.")
        return f"Indexed {count} chunks into namespace '{ns}'"
    except Exception as e:
        logger.error(f"[TOOL:INDEX] Indexing error: {e}", exc_info=True)
        return f"INDEX_ERROR: {e}"


def tool_send_mail(payload: str) -> str:
    """
    Notification / email dispatch tool.

    Purpose:
        Send alert or recommendation emails to configured stakeholders via SMTP.

    Parameters:
        payload (str): JSON string containing:
            {
                "subject": "Email Subject",
                "body": "Message content"
            }

    Returns:
        str: JSON string with send status ("sent", "mock_sent", or "error").

    Example:
        Input  -> '{"subject": "Shipment Delay Alert", "body": "Shipment SHP102 is delayed by 2 days."}'
        Output -> '{"status": "sent", "to": ["ops@company.com"]}'
    """
    logger.info("[TOOL:MAIL] Sending email via mailer service.")
    try:
        obj = json.loads(payload)
        subj = obj.get("subject", "CPG Alert")
        body = obj.get("body", "")
        res = send_mail(subj, body)
        logger.info(f"[TOOL:MAIL] Mail tool completed with status: {res.get('status')}")
        return _safe_json_dumps(res)
    except Exception as e:
        logger.error(f"[TOOL:MAIL] Error in mail tool: {e}", exc_info=True)
        return f"MAIL_ERROR: {e}"


def tool_ragas_eval(payload: str) -> str:
    """
    Evaluation tool for reasoning quality (RAGAS / heuristic fallback).

    Purpose:
        Evaluate an agent run’s factual grounding and hallucination rate based on
        retrieved vs generated content.

    Parameters:
        payload (str): JSON string representing a single agent run object, e.g.:
            {
                "input": "query text",
                "retrieved_docs": ["snippet1", "snippet2"],
                "raw_output": "agent response"
            }

    Returns:
        str: JSON string with evaluation metrics such as grounding_score and hallucination_rate.

    Example:
        Input  -> '{"input":"Why delay?","retrieved_docs":["factory fire"],"raw_output":"Fire delayed shipments."}'
        Output -> '{"grounding_score": 90.0, "hallucination_rate": 10.0, "notes": "..."}'
    """
    logger.info("[TOOL:RAGAS] Running RAGAS evaluation on agent output.")
    try:
        run_obj = json.loads(payload)
        res = run_ragas_evaluation(run_obj)
        logger.info(f"[TOOL:RAGAS] Evaluation completed: {res}")
        return _safe_json_dumps(res)
    except Exception as e:
        logger.error(f"[TOOL:RAGAS] RAGAS evaluation error: {e}", exc_info=True)
        return f"RAGAS_ERROR: {e}"


# ----------------------- Agent Builders ----------------------- #
LLM_FACT = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
LLM_DIALOG = ChatOpenAI(model="gpt-4o-mini", temperature=0.25)


def build_rca_agent(verbose: bool = False):
    logger.info("[AGENT:BUILD] Building RCA Agent.")
    tools = [
        Tool.from_function(
            tool_retrieve_multi,
            name="multi_retriever",
            description="Retrieve relevant context snippets from vector DBs (news_data, social_data, misc_data)."
        ),
        Tool.from_function(
            tool_storage,
            name="storage_query",
            description="Query or update local JSON stores for shipments, alerts, or agent runs."
        ),
        Tool.from_function(
            tool_index_text,
            name="index_texts",
            description="Index JSON-formatted documents into vector DB namespaces."
        ),
    ]
    return initialize_agent(
        tools=tools,
        llm=LLM_FACT,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=verbose,
        max_iterations=10,
    )


def build_rec_agent(verbose: bool = False):
    logger.info("[AGENT:BUILD] Building Recommendation Agent.")
    tools = [
        Tool.from_function(
            tool_storage,
            name="storage_query",
            description="Access or query stored shipment/alert/agent data from local JSON store."
        ),
        Tool.from_function(
            tool_retrieve_multi,
            name="multi_retriever",
            description="Retrieve multi-source context from news and social datasets for deeper reasoning."
        ),
    ]
    return initialize_agent(
        tools=tools,
        llm=LLM_DIALOG,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=verbose,
        max_iterations=10,
    )


def build_comm_agent(verbose: bool = False):
    logger.info("[AGENT:BUILD] Building Communicator Agent.")
    tools = [
        Tool.from_function(
            tool_send_mail,
            name="send_mail",
            description="Send alert or report email to configured contacts via SMTP."
        ),
        Tool.from_function(
            tool_storage,
            name="storage_query",
            description="Read/write alerts and runs from the local JSON store."
        ),
    ]
    return initialize_agent(
        tools=tools,
        llm=LLM_DIALOG,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=verbose,
        max_iterations=10,
    )


def build_prompt_tuner(verbose: bool = False):
    logger.info("[AGENT:BUILD] Building Prompt Tuner Agent.")
    tools = [
        Tool.from_function(
            tool_ragas_eval,
            name="ragas_eval",
            description="Run RAGAS or heuristic evaluation on an agent's reasoning trace."
        ),
        Tool.from_function(
            tool_storage,
            name="storage_query",
            description="Access previous agent runs and evaluation results."
        ),
    ]
    return initialize_agent(
        tools=tools,
        llm=LLM_DIALOG,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=verbose,
        max_iterations=10,
    )


# ----------------------- Orchestrator ----------------------- #

from agents.rca_agent import run_rca_agent  # ✅ new import


def orchestrate_alert(alert: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
    """
    Executes end-to-end orchestration for a given alert:
      1. Root Cause Analysis (via new ReAct RCA Agent)
      2. Impact Estimation
      3. Recommendation Generation
      4. Communication and Mail Dispatch

    Returns:
        Dict[str, Any]: results of each stage including logs and structured RCA output.
    """
    logger.info(
        f"[ORCHESTRATOR] Starting orchestration for alert: {alert.get('alert_id')} | Summary: {alert.get('summary')}")
    try:
        # -------------------- RCA STAGE --------------------
        logger.info("[ORCHESTRATOR] Running RCA Agent (ReAct Mode)")
        rca_result = run_rca_agent(alert, max_steps=8, llm_model="gpt-4o-mini", temperature=0.2)

        rca_parsed = rca_result.get("final_json", {})
        rca_logs = rca_result.get("logs", "")
        rca_text = rca_result.get("final_answer", "")

        # Save to agent_runs.json for traceability
        append_json("agent_runs.json", _build_run_record(
            agent="RCA_ReAct",
            alert_id=alert.get("alert_id"),
            input_text=json.dumps(alert),
            raw_output=rca_text,
            parsed_output={
                "rca_json": rca_parsed,
                "steps": rca_result.get("steps"),
                "logs": rca_logs
            }
        ))
        logger.info("[ORCHESTRATOR] RCA Agent completed successfully.")

        # -------------------- IMPACT STAGE --------------------
        logger.info("[ORCHESTRATOR] Computing impact heuristic.")
        impact = _compute_impact(alert, rca_parsed)
        append_json("agent_runs.json", _build_run_record(
            agent="Impact",
            alert_id=alert.get("alert_id"),
            input_text="impact_calc",
            raw_output=_safe_json_dumps(impact),
            parsed_output=impact
        ))

        # -------------------- RECOMMENDATION STAGE --------------------
        logger.info("[ORCHESTRATOR] Running Recommendation Agent.")
        rec_agent = build_rec_agent(verbose)
        rec_query = (
            f"You are a supply chain advisor. "
            f"Provide mitigation recommendations based on the RCA and impact data. "
            f"Use your tools to learn about related news or social media trends that may affect mitigation actions.\n"
            f"RCA: {json.dumps(rca_parsed)}\n"
            f"IMPACT: {json.dumps(impact)}"
        )
        rec_raw = rec_agent.run(rec_query)
        rec_parsed = _try_parse_json(rec_raw)
        append_json("agent_runs.json", _build_run_record(
            agent="Recommendation",
            alert_id=alert.get("alert_id"),
            input_text=rec_query,
            raw_output=rec_raw,
            parsed_output=rec_parsed
        ))
        logger.info("[ORCHESTRATOR] Recommendation Agent completed successfully.")

        # -------------------- COMMUNICATION STAGE --------------------
        logger.info("[ORCHESTRATOR] Running Communicator Agent.")
        comm_agent = build_comm_agent(verbose)
        comm_query = (
            f"Prepare a clear and professional email summary for stakeholders.\n"
            f"Include the RCA causes, impact, and recommendations.\n"
            f"ALERT: {json.dumps(alert)}\n"
            f"RCA: {json.dumps(rca_parsed)}\n"
            f"RECOMMENDATIONS: {json.dumps(rec_parsed)}"
        )
        comm_raw = comm_agent.run(comm_query)
        comm_parsed = _try_parse_json(comm_raw)
        append_json("agent_runs.json", _build_run_record(
            agent="Communicator",
            alert_id=alert.get("alert_id"),
            input_text=comm_query,
            raw_output=comm_raw,
            parsed_output=comm_parsed
        ))
        logger.info("[ORCHESTRATOR] Communicator Agent completed.")

        # -------------------- MAIL STAGE --------------------
        logger.info("[ORCHESTRATOR] Sending alert email to stakeholders.")
        mail_result = None
        if isinstance(comm_parsed, dict) and comm_parsed.get("subject") and comm_parsed.get("body"):
            mail_result = json.loads(tool_send_mail(_safe_json_dumps({
                "subject": comm_parsed["subject"],
                "body": comm_parsed["body"]
            })))
        else:
            mail_result = {"status": "not_sent", "reason": "invalid communicator output"}

        append_json("agent_runs.json", _build_run_record(
            agent="MailResult",
            alert_id=alert.get("alert_id"),
            input_text="send_mail",
            raw_output=_safe_json_dumps(mail_result),
            parsed_output=mail_result
        ))
        logger.info("[ORCHESTRATOR] Email dispatch completed successfully.")

        # -------------------- RETURN FULL RESULTS --------------------
        return {
            "rca": {
                "structured": rca_parsed,
                "logs": rca_logs,
                "text": rca_text
            },
            "impact": impact,
            "recommendations": rec_parsed,
            "mail_result": mail_result
        }

    except Exception as e:
        logger.error(f"[ORCHESTRATOR] Error during orchestration: {e}", exc_info=True)
        append_json("agent_runs.json", _build_run_record(
            agent="OrchestratorError",
            alert_id=alert.get("alert_id"),
            input_text="orchestrate_alert",
            raw_output=str(e),
            parsed_output={"error": str(e)}
        ))
        return {"error": str(e)}


# ----------------------- Helpers ----------------------- #

def _try_parse_json(text: str) -> Any:
    if not text:
        return {"raw": ""}
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return {"raw": text}
        return {"raw": text}


def _build_run_record(agent: str, alert_id: Optional[str], input_text: str, raw_output: str, parsed_output: Any) -> \
Dict[str, Any]:
    return {
        "run_id": f"{agent}_{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
        "agent": agent,
        "alert_id": alert_id,
        "input": input_text,
        "raw_output": raw_output,
        "parsed_output": parsed_output,
        "timestamp": datetime.datetime.utcnow().isoformat()
    }


def _compute_impact(alert: Dict[str, Any], rca: Dict[str, Any]) -> Dict[str, Any]:
    severity = str(alert.get("severity", "Medium"))
    mapping = {"Low": 0.2, "Medium": 0.5, "High": 0.85}
    base = mapping.get(severity, 0.5)
    try:
        causes_text = json.dumps(rca.get("causes", [])).lower()
        if any(k in causes_text for k in ["short", "stockout", "supplier"]):
            base = min(0.98, base + 0.12)
    except Exception as e:
        logger.warning(f"[IMPACT] Could not compute based on causes: {e}")
    impact = {
        "impact_score": round(base, 2),
        "expected_stockout_days": int(round(base * 10)),
        "affected_regions": alert.get("affected_regions", ["North", "West"])[:3]
    }
    logger.info(f"[IMPACT] Computed impact: {impact}")
    return impact
