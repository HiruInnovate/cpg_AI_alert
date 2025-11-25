# orchestrators/graph.py
from typing import TypedDict, Optional, List, Any
from langgraph.constants import END
from langgraph.graph import StateGraph

from agents.impact_analysis_agent import run_impact_agent
from agents.rca_agent import run_rca_agent


class FlowState(TypedDict):
    alert: dict
    rca: Optional[dict]
    impact: Optional[dict]
    recommendations: Optional[dict]
    comm_result: Optional[dict]
    traces: List[dict]
    logs: List[str]


def node_rca(state: FlowState) -> FlowState:
    """Run the RCA agent and store its trace + logs in the state."""
    res = run_rca_agent(state["alert"], max_steps=6)

    # Capture reasoning trace and logs
    rca_trace = res.get("steps", [])
    rca_logs = res.get("logs", "")

    traces = state.get("traces", [])
    logs = state.get("logs", [])

    traces.append({
        "agent": "RCA_Agent",
        "steps": rca_trace
    })
    logs.append(f"--- RCA Agent Logs ---\n{rca_logs}")

    return {
        **state,
        "rca": res.get("final_json") or {},
        "traces": traces,
        "logs": logs
    }


def node_impact(state: FlowState) -> FlowState:
    """Run the Impact Assessment agent and propagate trace and logs."""
    res = run_impact_agent(state["alert"], state.get("rca") or {}, max_steps=8)

    impact_trace = res.get("steps", [])
    impact_logs = res.get("logs", "")

    traces = state.get("traces", [])
    logs = state.get("logs", [])

    traces.append({
        "agent": "Impact_Agent",
        "steps": impact_trace
    })
    logs.append(f"--- Impact Agent Logs ---\n{impact_logs}")

    return {
        **state,
        "impact": res.get("final_json") or {},
        "traces": traces,
        "logs": logs
    }


def build_flow():
    """Build the multi-agent orchestration pipeline with trace chaining."""
    g = StateGraph(FlowState)

    g.add_node("rca", node_rca)
    g.add_node("impact", node_impact)

    g.set_entry_point("rca")
    g.add_edge("rca", "impact")
    g.add_edge("impact", END)

    return g.compile()
