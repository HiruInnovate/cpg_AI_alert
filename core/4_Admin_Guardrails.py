#core/8_admin_guardrails.py
import streamlit as st, json
GR = "config/guardrails.json"
st.header("🛡️ Admin — Guardrails")

guard = json.load(open(GR))
banned = st.text_area("Banned phrases (comma separated)", ", ".join(guard.get("banned_phrases",[])))
min_conf = st.number_input("Min confidence for auto-action", min_value=0, max_value=100, value=int(guard.get("min_confidence_for_auto_action",70)))
if st.button("Save Guardrails"):
    guard["banned_phrases"] = [p.strip() for p in banned.split(",") if p.strip()]
    guard["min_confidence_for_auto_action"] = min_conf
    json.dump(guard, open(GR,"w"), indent=2)
    st.success("Saved")
