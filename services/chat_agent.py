"""
services/chat_agent.py
Strict ID-aware RCA Chat Agent (Ultra-Strict, merged with enhanced features)

This file implements:
- Robust session_state initialization
- Ultra-strict ID extraction that preserves full numeric token (so ALRT_235543 != ALRT_235543432)
- Auto-detection of IDs in free text (but strict validation when scope_id explicitly provided)
- Follow-up handling (uses last RCA; never re-runs RCA)
- Multi-ID merge helper
- Defensive handling around missing final_json/logs
- Consistent return shapes suitable for the Streamlit UI
"""

import json
import traceback
import re
from time import time
from typing import List, Tuple, Dict, Any, Optional

import streamlit as st
from langchain_openai import ChatOpenAI

from services.storage import load_json
from agents.rca_agent import run_rca_agent
from agents.langchain_agents import tool_retrieve_multi

# -------------------------
# Session-state safe init
# -------------------------
if "last_rca" not in st.session_state:
    st.session_state.last_rca = None
if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = []
if "pending_action" not in st.session_state:
    st.session_state.pending_action = None

# -------------------------
# ID patterns & validators (ULTRA-STRICT)
# - For strict validation we require the full numeric token following ALRT_ or SHP
# - Validator uses anchored regex; extractor captures full digit group
# -------------------------
ALERT_ID_VALIDATOR = re.compile(r"^ALRT[_-]?\d{3,12}$", re.IGNORECASE)   # explicit validation (ALRT_12345)
SHIP_ID_VALIDATOR  = re.compile(r"^SHP[_-]?\d{2,12}$", re.IGNORECASE)    # explicit validation (SHP1234)

def is_valid_alert_id(raw: Optional[str]) -> bool:
    if not raw:
        return False
    return bool(ALERT_ID_VALIDATOR.match(raw.strip()))

def is_valid_shipment_id(raw: Optional[str]) -> bool:
    if not raw:
        return False
    return bool(SHIP_ID_VALIDATOR.match(raw.strip()))

# -------------------------
# Flexible extractor (auto-detect) - ULTRA-STRICT CAPTURE
# - Capture the entire digit sequence after the ALRT/SHP token so we don't mistakenly
#   match a substring of a longer token.
# Examples:
#   "ALRT_235543" -> ALRT_235543
#   "ALRT_235543432" -> ALRT_235543432 (distinct)
# -------------------------
EXTRACT_ALERT_RE = re.compile(r"\bALRT[_-]?(\d{3,12})\b", re.IGNORECASE)
EXTRACT_SHIP_RE  = re.compile(r"\bSHP[_-]?(\d{2,12})\b", re.IGNORECASE)

def extract_ids(query: str) -> Tuple[List[str], List[str]]:
    """Return (alerts, shipments) lists normalized (ALRT_12345, SHP123)."""
    q = (query or "").strip()
    alerts: List[str] = []
    ships: List[str] = []

    # capture full digit group to avoid partial matches
    for m in EXTRACT_ALERT_RE.finditer(q):
        digits = m.group(1)
        if digits:
            alerts.append(f"ALRT_{digits}")

    for m in EXTRACT_SHIP_RE.finditer(q):
        digits = m.group(1)
        if digits:
            ships.append(f"SHP{digits}")

    # dedupe but preserve order
    def _uniq(seq: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for s in seq:
            if s not in seen:
                out.append(s)
                seen.add(s)
        return out

    return _uniq(alerts), _uniq(ships)

# -------------------------
# Loaders
# -------------------------
def _load_alert(alert_id: str):
    alerts = load_json("alerts.json")
    return next((a for a in alerts if str(a.get("alert_id")) == alert_id), None)

def _load_shipment(shipment_id: str):
    shipments = load_json("shipments.json")
    return next((s for s in shipments if str(s.get("shipment_id")) == shipment_id), None)

# -------------------------
# Format JSON → Natural language
# -------------------------
def format_rca_json(rca_json: Optional[Dict[str, Any]]) -> str:
    if not rca_json:
        return "No structured RCA data available."

    parts: List[str] = []

    summary = rca_json.get("summary")
    if summary:
        parts.append(f"Summary: {summary}")

    causes = rca_json.get("causes") or []
    if causes:
        parts.append("Root causes:")
        for c in causes:
            lbl = c.get("label", "Cause")
            expl = c.get("explanation", "")
            parts.append(f"• {lbl}: {expl}")

    recs = rca_json.get("recommendations") or []
    if recs:
        parts.append("Recommended actions:")
        for r in recs:
            action = r.get("action") or r.get("description") or str(r)
            eta = r.get("eta_hours", "N/A")
            parts.append(f"• {action} (ETA: {eta} hrs)")

    evidence = rca_json.get("evidence") or []
    if evidence:
        parts.append("Evidence:")
        for e in evidence:
            parts.append(f"• {e}")

    confidence = rca_json.get("confidence")
    if confidence is not None:
        try:
            parts.append(f"Confidence: {round(float(confidence), 2)}")
        except Exception:
            parts.append(f"Confidence: {confidence}")

    return "\n\n".join(parts)

# -------------------------
# Memory helpers
# -------------------------
def _append_memory(query: str, rca_json: Dict[str, Any]):
    if "conversation_memory" not in st.session_state:
        st.session_state.conversation_memory = []
    st.session_state.conversation_memory.append({
        "query": query,
        "rca": rca_json
    })
    st.session_state.conversation_memory = st.session_state.conversation_memory[-10:]

# -------------------------
# Follow-up handling
# -------------------------
FOLLOW_UP_KEYWORDS = [
    "improve", "fix", "avoid", "prevent", "next time",
    "reduce", "optimize", "summarize", "simple",
    "explain", "what else", "give only", "provide only", "how to"
]

def is_follow_up(query: str) -> bool:
    if not query:
        return False
    q = query.lower()
    return any(k in q for k in FOLLOW_UP_KEYWORDS)

def run_follow_up(query: str) -> str:
    """Generate a short follow-up using last RCA + recent memory (no re-run)."""
    last = st.session_state.last_rca or {}
    recent_snippet = ""
    if st.session_state.conversation_memory:
        items = st.session_state.conversation_memory[-3:]
        recent_snippet = "\n".join(
            f"Q: {m['query']}\nA: {m['rca'].get('summary','')}" for m in items if m.get("rca")
        )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.35)
    prompt = f"""
You are an operations advisor. Use the last RCA JSON and recent conversation.

Recent memory:
{recent_snippet}

Latest RCA JSON:
{json.dumps(last.get('final_json', {}), indent=2)}

User follow-up: "{query}"

Respond concisely in bullet points. No JSON, no fluff.
"""
    return llm.invoke(prompt).content

# -------------------------
# Multi-RCA merger (simple)
# -------------------------
def merge_multiple_rca(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge multiple RCA run outputs into one consolidated structure.
    - Picks the most frequent causes (by label)
    - Concatenates evidence, recommendations (deduped)
    - Uses average confidence
    """
    merged = {"summary": "", "causes": [], "recommendations": [], "evidence": [], "confidence": 0.0}
    if not results:
        return merged

    jsons = [r.get("final_json") for r in results if r and r.get("final_json")]
    jsons = [j for j in jsons if isinstance(j, dict)]
    if not jsons:
        return merged

    # summaries: join brief summaries
    summaries = [j.get("summary", "") for j in jsons if j.get("summary")]
    merged["summary"] = " | ".join(summaries[:3])

    # causes: collect unique by label (preserve first occurrence)
    seen_causes: Dict[str, Dict[str, Any]] = {}
    for j in jsons:
        for c in (j.get("causes") or []):
            lbl = (c.get("label") or "").strip()
            if not lbl:
                continue
            if lbl not in seen_causes:
                seen_causes[lbl] = c
    merged["causes"] = list(seen_causes.values())

    # recommendations: dedupe by action text
    seen_recs: Dict[str, Dict[str, Any]] = {}
    for j in jsons:
        for r in (j.get("recommendations") or []):
            key = (r.get("action") or str(r)).strip()
            if key and key not in seen_recs:
                seen_recs[key] = r
    merged["recommendations"] = list(seen_recs.values())

    # evidence: dedupe preserve order
    evidence_set: List[str] = []
    for j in jsons:
        for e in (j.get("evidence") or []):
            if e not in evidence_set:
                evidence_set.append(e)
    merged["evidence"] = evidence_set

    # confidence: mean of confidences present
    confidences: List[float] = []
    for j in jsons:
        try:
            if j.get("confidence") is not None:
                confidences.append(float(j.get("confidence")))
        except Exception:
            continue
    merged["confidence"] = sum(confidences) / len(confidences) if confidences else 0.0

    return merged

# -------------------------
# MAIN: chat_agent
# -------------------------
def chat_agent(query: str, scope: str, scope_id: Optional[str] = None,
               explain: bool = False, debate: bool = False) -> Dict[str, Any]:
    """
    Returns a dict:
      {
        "answer": str,
        "trace": str | None,
        "sources": list,
        "confidence": float,
        "meta": dict
      }
    """
    try:
        q = (query or "").strip()

        # 1) Follow-up detection (use last_rca, do NOT re-run)
        if st.session_state.last_rca and is_follow_up(q):
            result_text = run_follow_up(q)
            return {"answer": result_text, "trace": None, "sources": ["follow-up"], "confidence": 0.95, "meta": {}}

        # 2) Auto-detect IDs inside query (override scope if helpful)
        detected_alerts, detected_shipments = extract_ids(q)
        # If UI scope is global but we found an ID in the query, switch to that scope
        if scope == "global":
            if detected_alerts:
                scope = "alert"
                scope_id = detected_alerts[0]
            elif detected_shipments:
                scope = "shipment"
                scope_id = detected_shipments[0]

        # 3) ALERT scope handling
        if scope == "alert":
            # if no explicit scope_id but detected, use detection
            if not scope_id and detected_alerts:
                scope_id = detected_alerts[0]

            if not scope_id:
                return {"answer": "No alert ID provided or detected in the query.", "trace": None, "sources": [], "confidence": 0.0, "meta": {}}

            # normalize scope_id
            scope_id_norm = str(scope_id).strip().upper()
            # Ultra-strict validation: ensure the full token matches pattern
            if not is_valid_alert_id(scope_id_norm):
                return {"answer": f"Invalid alert ID format: '{scope_id}'. Expected like ALRT_25981", "trace": None, "sources": [], "confidence": 0.0, "meta": {}}

            alert = _load_alert(scope_id_norm)
            if not alert:
                return {"answer": f"Alert {scope_id_norm} not found.", "trace": None, "sources": [], "confidence": 0.0, "meta": {}}

            rca = run_rca_agent(alert, max_steps=10)
            st.session_state.last_rca = rca
            if rca.get("final_json"):
                _append_memory(q, rca["final_json"])

            formatted = format_rca_json(rca.get("final_json"))
            return {
                "answer": formatted,
                "trace": rca.get("logs") if rca.get("logs") and explain else None,
                "sources": ["alerts.json", "rca_agent"],
                "confidence": (rca.get("final_json") or {}).get("confidence", 0.85),
                "meta": {"rca": rca.get("final_json"), "raw": rca}
            }

        # 4) SHIPMENT scope handling
        if scope == "shipment":
            if not scope_id and detected_shipments:
                scope_id = detected_shipments[0]

            if not scope_id:
                return {"answer": "No shipment ID provided or detected in the query.", "trace": None, "sources": [], "confidence": 0.0, "meta": {}}

            scope_id_norm = str(scope_id).strip().upper().replace("SHP_", "SHP")
            if not is_valid_shipment_id(scope_id_norm):
                return {"answer": f"Invalid shipment ID format: '{scope_id}'. Expected like SHP1013", "trace": None, "sources": [], "confidence": 0.0, "meta": {}}

            shipment = _load_shipment(scope_id_norm)
            if not shipment:
                return {"answer": f"Shipment {scope_id_norm} not found.", "trace": None, "sources": [], "confidence": 0.0, "meta": {}}

            fake_alert = {
                "alert_id": f"SHP_{scope_id_norm}",
                "shipment_id": scope_id_norm,
                "summary": q,
                "severity": "Medium",
                "shipment_data": shipment
            }
            rca = run_rca_agent(fake_alert, max_steps=10)
            st.session_state.last_rca = rca
            if rca.get("final_json"):
                _append_memory(q, rca["final_json"])

            formatted = format_rca_json(rca.get("final_json"))
            return {
                "answer": formatted,
                "trace": rca.get("logs") if rca.get("logs") and explain else None,
                "sources": ["shipments.json", "rca_agent"],
                "confidence": (rca.get("final_json") or {}).get("confidence", 0.85),
                "meta": {"rca": rca.get("final_json"), "raw": rca}
            }

        # 5) Multi-ID handling (if multiple IDs present in query)
        if detected_alerts or detected_shipments:
            runs: List[Dict[str, Any]] = []
            for aid in detected_alerts:
                a = _load_alert(aid)
                if a:
                    runs.append(run_rca_agent(a, max_steps=6))
            for sid in detected_shipments:
                s = _load_shipment(sid)
                if s:
                    fake_alert = {
                        "alert_id": f"SHP_{sid}",
                        "shipment_id": sid,
                        "summary": q,
                        "severity": "Medium",
                        "shipment_data": s
                    }
                    runs.append(run_rca_agent(fake_alert, max_steps=6))

            merged = merge_multiple_rca(runs)
            st.session_state.last_rca = {"final_json": merged, "final_answer": None, "steps": [], "logs": ""}
            _append_memory(q, merged)
            return {
                "answer": format_rca_json(merged),
                "trace": None,
                "sources": ["multi-id"],
                "confidence": merged.get("confidence", 0.85),
                "meta": {"rca": merged, "raw_runs": runs}
            }

        # 6) GLOBAL fallback
        retrieved = tool_retrieve_multi(q)
        fake_alert = {
            "alert_id": f"GLB_{int(time()*1000)}",
            "summary": q,
            "severity": "Low",
            "retrieved_context": retrieved
        }
        rca = run_rca_agent(fake_alert, max_steps=10)
        st.session_state.last_rca = rca
        if rca.get("final_json"):
            _append_memory(q, rca["final_json"])
        formatted = format_rca_json(rca.get("final_json"))
        return {
            "answer": f"{formatted}\n\n---\nRelevant context:\n{(retrieved or '')[:800]}",
            "trace": rca.get("logs") if rca.get("logs") and explain else None,
            "sources": ["vector_db", "rca_agent"],
            "confidence": (rca.get("final_json") or {}).get("confidence", 0.80),
            "meta": {"rca": rca.get("final_json"), "raw": rca}
        }

    except Exception as e:
        return {"answer": f"⚠️ Error: {e}", "trace": traceback.format_exc() if hasattr(traceback, "format_exc") else None,
                "sources": ["internal-error"], "confidence": 0.0, "meta": {}}
