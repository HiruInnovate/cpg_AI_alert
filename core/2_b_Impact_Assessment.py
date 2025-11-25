
import streamlit as st
import time
import json

from orchestrators.alert_rca_impact_graph import build_flow
from services.storage import load_json

from fpdf import FPDF
from services.logger_config import get_logger

logger = get_logger(__name__)

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="📊 Impact Assessment", layout="wide", page_icon="💰")
st.header("💰 Supply Chain Impact Assessment Dashboard")

# -----------------------------
# LOAD ALERTS
# -----------------------------
alerts = load_json("alerts.json")

if not alerts:
    st.warning("⚠️ No alerts available. Please create alerts first using the Alerts page.")
    st.stop()

# -----------------------------
# SELECT ALERT
# -----------------------------
st.markdown("### 🧭 Select an Alert for Impact Analysis")
alert_options = {f"{a['alert_id']} - {a['summary'][:60]}": a for a in alerts}
selected_label = st.selectbox("Choose an Alert", list(alert_options.keys()))
selected_alert = alert_options[selected_label]

# -----------------------------
# RUN IMPACT FLOW
# -----------------------------
if st.button("🚀 Run Impact Assessment", use_container_width=True):
    with st.spinner(f"Running RCA ➜ Impact flow for {selected_alert['alert_id']}..."):
        flow = build_flow()
        result_state = flow.invoke({"alert": selected_alert, "traces": [], "logs": []})
        time.sleep(1.2)

    st.success("✅ Impact assessment completed successfully!")

    # -----------------------------
    # DISPLAY IMPACT SUMMARY
    # -----------------------------
    st.divider()
    st.markdown("## 📈 Business Impact Assessment Report")

    impact = result_state.get("impact", {})
    rca = result_state.get("rca", {})

    if not impact:
        st.warning("No structured impact output found.")
        st.stop()

    # Impact Summary
    st.markdown(
        f"""
        <div style="background:linear-gradient(145deg,#1e1e1e,#2a2a2a);
            padding:1.2rem;border-radius:12px;border-left:5px solid #f59f00;
            margin-bottom:15px;box-shadow:0 2px 6px rgba(0,0,0,0.4);">
            <h5 style="color:#ffd43b;">📊 <b>Impact Summary</b></h5>
            <p style="color:#dee2e6;">{impact.get('impact_summary','No summary available.')}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Delay & Confidence
    delay = impact.get("delay_hours", 0)
    confidence = impact.get("confidence", 0)
    st.markdown(f"#### ⏱️ Estimated Delay: **{delay} hours**")
    st.progress(min(max(float(confidence), 0), 1))
    st.caption(f"{confidence}% confidence in impact analysis.")

    # Itemized Impact Table
    items = impact.get("items", [])
    if items:
        st.markdown("#### 💼 Itemized Impact Breakdown")
        for i in items:
            st.markdown(
                f"""
                <div style="background-color:#2f2f2f;padding:1rem;border-radius:10px;
                border-left:6px solid #fab005;margin-bottom:10px;">
                    <h6 style="color:#ffd43b;">📦 SKU: {i.get('sku','')}</h6>
                    <p style="color:#f1f3f5;">
                    Region: <b>{i.get('region','')}</b> |
                    At-Risk Units: <b>{i.get('at_risk_units',0)}</b> |
                    Unfulfilled Units: <b>{i.get('unfulfilled_units',0)}</b><br>
                    Lost Sales: ₹{i.get('lost_sales',0):,.2f} |
                    SLA Penalty: ₹{i.get('sla_penalty',0):,.2f} |
                    Expedite Cost: ₹{i.get('expedite_cost',0):,.2f}<br>
                    <i>{i.get('notes','')}</i>
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Totals
    total = impact.get("total", {})
    if total:
        st.markdown("#### 💰 Total Estimated Impact")
        st.metric("Total Lost Sales (₹)", f"{total.get('lost_sales',0):,.2f}")
        st.metric("Total SLA Penalty (₹)", f"{total.get('sla_penalty',0):,.2f}")
        st.metric("Total Expedite Cost (₹)", f"{total.get('expedite_cost',0):,.2f}")
        st.metric("Overall Cost Impact (₹)", f"{total.get('overall',0):,.2f}")

    # Assumptions
    assumptions = impact.get("assumptions", [])
    if assumptions:
        st.markdown("#### 🧩 Key Assumptions")
        for a in assumptions:
            st.markdown(f"- {a}")


    # -----------------------------
    # ADMIN LOG SECTION
    # -----------------------------
    st.divider()
    st.markdown("### ⚙️ Agent Trace & Reasoning (Admin Only)")

    admin_mode = True if "admin" in st.session_state.get("user", "") else False

    if admin_mode:
        with st.expander("🧾 View Full Agent Logs", expanded=False):
            traces = result_state.get("traces", [])
            for t in traces:
                st.markdown(f"#### 🤖 {t['agent']} Trace")
                for s in t["steps"]:
                    st.markdown(
                        f"""
                        <div style="background-color:#333;padding:0.8rem;border-radius:8px;
                        border-left:5px solid #888;margin-bottom:10px;">
                            <b>Step {s.get('step','?')} — {s.get('tool','')}</b><br>
                            <span style="color:#999;">🔹 Input:</span> {s.get('tool_input','')}<br>
                            <span style="color:#999;">🔹 Observation:</span>
                            <pre style="background-color:#111;
                            padding:0.5rem;border-radius:6px;color:#ccc;">{s.get('observation','')[:400]}</pre>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            st.markdown("#### 🧾 Combined Execution Logs")
            all_logs = "\n\n".join(result_state.get("logs", []))
            st.code(all_logs or "No logs available.", language="markdown")
    else:
        st.caption("🔒 Agent traces and logs are restricted to admin users.")
