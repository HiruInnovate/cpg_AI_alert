import streamlit as st
import time
import random
from services.storage import load_json, overwrite_json
from agents.monitor_agent import run_monitor_and_orchestrate
from agents.rca_agent import run_rca_agent
from services.logger_config import get_logger

logger = get_logger(__name__)

# Page title
st.set_page_config(page_title="Supply Chain Alerts", layout="wide", page_icon="🔔")
st.header("🔔 Supply Chain Alerts Dashboard")

# -----------------------------
# Run Monitor & Orchestrate Section
# -----------------------------
st.markdown("### 🧭 Monitor & Generate Alerts")

if st.button("📡 Scan for Alerts", use_container_width=True):
    with st.spinner("Scanning supply chain data sources and orchestrating alerts..."):
        new_alerts = run_monitor_and_orchestrate(verbose=True)
        time.sleep(1)
        st.success(f"✅ {len(new_alerts)} new alerts generated!")

st.divider()

# -----------------------------
# Display Alerts Section
# -----------------------------
st.markdown("### 🚨 Active Alerts")

alerts = load_json("alerts.json")

if not alerts:
    st.info("No alerts available. Please run 'Scan for Alerts' first.")
else:
    # Group alerts by severity for colored presentation
    severity_colors = {
        "High": "#ff4b4b",
        "Medium": "#ffa94d",
        "Low": "#4dabf7"
    }

    for a in sorted(alerts, key=lambda x: x.get("created_at", ""), reverse=True)[:30]:
        sev = a.get("severity", "Medium")
        color = severity_colors.get(sev, "#cccccc")

        with st.container():
            st.markdown(
                f"""
                <div style="background-color:{color}22;padding:1rem;border-radius:10px;margin-bottom:10px;
                border-left:6px solid {color}">
                    <h5 style="margin:0;">🆔 {a['alert_id']} &nbsp; | &nbsp; <b>{sev}</b></h5>
                    <p style="margin:0.3rem 0;">📦 <b>Shipment:</b> {a.get('shipment_id','N/A')}</p>
                    <p style="margin:0.3rem 0;">📝 {a['summary']}</p>
                    <p style="font-size:0.8rem;margin:0;color:gray;">Created: {a.get('created_at')}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

st.divider()

# -----------------------------
# RCA Section
# -----------------------------
st.markdown("### 🧠 AI Agent: Alert Analysis and Recommended Actions")

alerts = load_json("alerts.json")
if not alerts:
    st.warning("⚠️ No alerts to analyze. Please create alerts first.")
else:
    alert_options = {f"{a['alert_id']} - {a['summary'][:50]}": a for a in alerts}
    selected_label = st.selectbox("Select an Alert for RCA", list(alert_options.keys()))
    selected_alert = alert_options[selected_label]

    if st.button("🚀 Run RCA Agent", use_container_width=True):
        with st.spinner(f"Analyzing alert {selected_alert['alert_id']}..."):
            # Run the RCA Agent
            res = run_rca_agent(selected_alert, max_steps=6)
            time.sleep(1.5)

        # -------------------
        # Display RCA Logs (color-coded)
        # -------------------
        st.markdown("#### 🧩 Agent Reasoning Trace")

        if res.get("steps"):
            for s in res["steps"]:
                color_map = {
                    "get_shipment_details": "#74c0fc",
                    "get_logistics_performance": "#ffd43b",
                    "search_external_context": "#a9e34b"
                }
                step_color = color_map.get(s["tool"], "#dee2e6")
                st.markdown(
                    f"""
                    <div style="background-color:{step_color}22;padding:0.8rem;border-radius:8px;
                    border-left:5px solid {step_color};margin-bottom:10px;">
                        <b>Step {s['step']} — {s['tool']}</b><br>
                        <span style="color:#555;">🔹 Input:</span> {s['tool_input']}<br>
                        <span style="color:#555;">🔹 Observation:</span> <pre style="background-color:#f8f9fa;
                        padding:0.5rem;border-radius:6px;">{s['observation'][:500]}</pre>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        else:
            st.info("No reasoning steps were captured.")

        # -------------------
        # Final RCA Output (Dark Theme Optimized)
        # -------------------
        st.divider()
        st.markdown("### ✅ Final Root Cause Analysis")

        final_json = res.get("final_json", {})
        final_text = res.get("final_answer", "")

        if final_json:
            # ---- Summary Section ----
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(145deg, #1e1e1e, #2a2a2a);
                    padding:1.2rem;border-radius:12px;
                    border-left:5px solid #339af0;margin-bottom:15px;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.4);">
                    <h5 style="margin:0;color:#91d5ff;">🧾 <b>Summary</b></h5>
                    <p style="margin-top:0.5rem;color:#dee2e6;">{final_json.get('summary', 'No summary available.')}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            # ---- Confidence Meter ----
            conf = final_json.get("confidence", 0)
            st.markdown("#### 🎯 Confidence Level")
            st.progress(min(max(float(conf), 0), 1))
            st.caption(f"{conf * 100:.1f}% confidence in this RCA conclusion.")

            # ---- Causes ----
            causes = final_json.get("causes", [])
            if causes:
                st.markdown("#### 🧩 Root Causes Identified")
                for c in causes:
                    label = c.get("label", "Unknown Cause")
                    explanation = c.get("explanation", "")
                    st.markdown(
                        f"""
                        <div style="
                            background-color:#2f2f2f;
                            padding:1rem;border-radius:10px;
                            border-left:6px solid #fab005;
                            margin-bottom:10px;
                            box-shadow:inset 0 0 6px rgba(255,255,255,0.05);">
                            <h6 style="margin:0;color:#ffd43b;">⚠️ <b>{label}</b></h6>
                            <p style="margin-top:0.5rem;margin-bottom:0;color:#f1f3f5;">{explanation}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.info("No causes were explicitly identified by the RCA agent.")

            # ---- Recommendations ----
            recs = final_json.get("recommendations", [])
            if recs:
                st.markdown("#### 🧭 Recommended Actions")
                for r in recs:
                    st.markdown(
                        f"""
                        <div style="
                            background-color:#1f3b1f;
                            padding:1rem;border-radius:10px;
                            border-left:6px solid #37b24d;
                            margin-bottom:10px;
                            box-shadow:inset 0 0 8px rgba(0,255,100,0.1);">
                            <h6 style="margin:0;color:#69db7c;">✅ <b>{r.get('action', '(Action)')}</b></h6>
                            <p style="margin:0.3rem 0;color:#b2f2bb;">
                            ETA: <b>{r.get('eta_hours', '?')}</b> hours &nbsp;|&nbsp; 
                            Confidence: <b>{r.get('confidence_est', 0) * 100:.0f}%</b></p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.info("No recommendations were generated.")

            # ---- Evidence ----
            evid = final_json.get("evidence", [])
            if evid:
                st.markdown("#### 📚 Supporting Evidence")
                for e in evid:
                    st.markdown(
                        f"""
                        <div style="
                            background-color:#262626;
                            padding:0.8rem;border-radius:8px;
                            border-left:5px solid #868e96;
                            margin-bottom:6px;
                            color:#e9ecef;">
                            <p style="margin:0;">🧠 {e}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.caption("No explicit evidence snippets were returned.")

        elif final_text:
            st.markdown(
                f"""
                <div style="background-color:#1e1e1e;padding:1rem;border-radius:8px;
                border-left:5px solid #74c0fc;">
                <p style="color:#e9ecef;">{final_text}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.warning("No structured RCA output generated.")

        # -------------------
        # Logs section (for debugging / transparency)
        # -------------------
        st.divider()
        st.markdown("### 🧾 Agent Execution Logs")
        st.code(res.get("logs", "No logs available."), language="markdown")
