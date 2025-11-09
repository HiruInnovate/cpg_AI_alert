#core/1_dashboard.py
import streamlit as st
from services.storage import load_json
import pandas as pd, plotly.express as px

st.header("📊 Dashboard")
st.write("Live KPIs and top impacted SKUs/stores.")
alerts = load_json("alerts.json")
agent_runs = load_json("agent_runs.json")

col1, col2, col3 = st.columns(3)
col1.metric("Active Alerts", sum(1 for a in alerts if a.get("status")=="Active"))
col2.metric("Total Alerts", len(alerts))
col3.metric("Agent Runs", len(agent_runs))

st.markdown("---")
st.subheader("Recent Alerts")
if alerts:
    df = pd.DataFrame(alerts).sort_values("created_at", ascending=False)
    st.dataframe(df[["alert_id","shipment_id","summary","severity","status","created_at"]].head(20))
else:
    st.info("No alerts yet. Upload data and run monitor.")
