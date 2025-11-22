import streamlit as st
import json
import os

# --- File Path ---
GR = "config/guardrails.json"

# --- Page Setup ---
st.set_page_config(page_title="🛡️ Admin — Guardrails Config Editor", layout="centered")
st.title("🛡️ Supply Chain AI Guardrails Configuration")

# --- Load or Initialize Config ---
if os.path.exists(GR):
    guard = json.load(open(GR))
else:
    guard = {"banned_phrases": [], "min_confidence_for_auto_action": 70, "business_rules": []}

# --- Banned Phrases Section ---
st.markdown("### 🚫 Banned Phrases")
st.caption("These phrases are restricted from appearing in the AI's recommendations or reasoning output.")
banned = st.text_area(
    "Enter banned phrases (comma-separated)",
    ", ".join(guard.get("banned_phrases", [])),
    placeholder="e.g., blame vendor, cancel all shipments, ignore delay"
)

st.divider()

# --- Confidence Threshold Section ---
st.markdown("### 🎯 Minimum Confidence Threshold")
st.caption("Minimum confidence (%) required for the AI to auto-recommend or execute actions without human approval.")
min_conf = st.slider(
    "Set confidence threshold",
    0,
    100,
    int(guard.get("min_confidence_for_auto_action", 70)),
    help="Below this threshold, human review is required."
)

st.divider()

# --- Business Rules Section ---
st.markdown("### 🧩 Business Rules")
st.caption("Define fairness, safety, and operational constraints for the RCA AI agent. Each rule ensures responsible decision-making.")

rules = guard.get("business_rules", [])
if "new_rules" not in st.session_state:
    st.session_state.new_rules = list(rules)

# --- Rule Editor UI Blocks ---
for idx, rule in enumerate(st.session_state.new_rules):
    with st.container():
        st.markdown(
            f"""
            <div style="background-color:#1e1e1e;padding:1rem;border-radius:10px;margin-bottom:10px;
            border-left:5px solid #4dabf7;box-shadow:0 2px 5px rgba(0,0,0,0.4);">
            <b>Rule {idx + 1}</b>
            </div>
            """,
            unsafe_allow_html=True
        )

        new_value = st.text_area(
            f"✏️ Edit Rule {idx + 1}",
            rule,
            key=f"rule_{idx}",
            height=80
        )

        col1, col2 = st.columns([0.15, 0.85])
        with col1:
            if st.button("🗑️ Remove", key=f"remove_{idx}"):
                st.session_state.new_rules.pop(idx)
                st.rerun()
        st.session_state.new_rules[idx] = new_value.strip()

# --- Add New Rule ---
st.divider()
st.markdown("#### ➕ Add New Business Rule")
new_rule_input = st.text_area("Enter a new rule", placeholder="Type your new business rule here...", height=70)
if st.button("Add Rule"):
    if new_rule_input.strip():
        st.session_state.new_rules.append(new_rule_input.strip())
        st.success("✅ Rule added successfully!")
        st.rerun()
    else:
        st.warning("⚠️ Please enter a valid rule before adding.")

st.divider()

# --- Save Configuration ---
if st.button("💾 Save Guardrails Configuration", use_container_width=True):
    guard["banned_phrases"] = [p.strip() for p in banned.split(",") if p.strip()]
    guard["min_confidence_for_auto_action"] = min_conf
    guard["business_rules"] = [r for r in st.session_state.new_rules if r.strip()]
    json.dump(guard, open(GR, "w"), indent=2)
    st.success("✅ Guardrails configuration updated successfully!")

# --- Footer ---
st.caption("""
💡 *These guardrails are dynamically enforced by the RCA agent at runtime to ensure compliance, fairness, and explainability.*
""")
