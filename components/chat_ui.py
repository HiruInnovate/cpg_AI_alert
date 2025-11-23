import streamlit as st
import time

from components.timeline_component import trace_timeline
from components.map_widget import shipment_map
from components.reaction_buttons import recommended_actions
from services.chat_agent import chat_agent
from services.chat_agent import format_rca_json



def typing_effect(text: str, speed=0.008):
    final = ""
    placeholder = st.empty()
    for char in text:
        final += char
        placeholder.markdown(
            f"""
            <div style='background:white; color:#1e293b; padding:10px; 
                        border-radius:8px; width:70%; border:1px solid #334155;'>
                <b>RCA Agent:</b><br>{final}
            </div>
            """,
            unsafe_allow_html=True
        )
        time.sleep(speed)
    return placeholder


def chat_ui():

    # -------------------------
    # INITIAL STATE
    # -------------------------
    st.sidebar.title("Settings")
    st.sidebar.write("---")

    if "debate_mode" not in st.session_state:
        st.session_state.debate_mode = False

    st.session_state.debate_mode = st.sidebar.checkbox(
        "Enable Multi-Agent Debate Mode",
        value=st.session_state.debate_mode
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "chat_scope" not in st.session_state:
        st.session_state.chat_scope = "global"

    if "chat_scope_id" not in st.session_state:
        st.session_state.chat_scope_id = ""

    if "chat_show_trace" not in st.session_state:
        st.session_state.chat_show_trace = False

    if "chat_latency" not in st.session_state:
        st.session_state.chat_latency = None

    if "last_rca" not in st.session_state:
        st.session_state.last_rca = None

    if "conversation_memory" not in st.session_state:
        st.session_state.conversation_memory = []

    # -------------------------
    st.markdown("## 💬 RCA Chat Assistant")
    st.caption("Ask about alerts, shipments or delays.")

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        st.session_state.chat_scope = st.selectbox(
            "Scope", 
            ["global", "alert", "shipment"]
        )
    with col2:
        st.session_state.chat_scope_id = st.text_input("ID (optional)")
    with col3:
        st.session_state.chat_show_trace = st.checkbox("Show reasoning trace")

    st.markdown("---")

    # -------------------------
    # CHAT HISTORY
    # -------------------------
    for msg in st.session_state.chat_history:

        # USER BUBBLE
        if msg["sender"] == "user":
            st.markdown(
                f"""
                <div style='background:#1e293b; color:white; padding:10px;
                            margin:6px; border-radius:8px; width:70%; float:right;'>
                    <b>You:</b><br>{msg['text']}
                </div>
                <div style='clear:both;'></div>
                """,
                unsafe_allow_html=True
            )

        # AGENT BUBBLE
        else:
            st.markdown(
                f"""
                <div style='margin:6px;'>
                    <div style='background:white; color:#1e293b; padding:10px;
                                border-radius:8px; width:70%; border:1px solid #334155;'>
                        <b>RCA Agent:</b><br>{msg['text']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # SOURCES
            if msg.get("sources"):
                st.caption(f"🔍 Sources: {msg['sources']}")

            # TRACE
            if st.session_state.chat_show_trace and msg.get("trace"):
                trace_timeline(msg["trace"])

            # MAP (optional)
            if st.session_state.chat_scope == "shipment" and st.session_state.chat_scope_id:
                shipment_map(st.session_state.chat_scope_id)

            # ACTION BUTTONS (UNIQUE KEY)
            recommended_actions(
                alert_id=st.session_state.chat_scope_id if st.session_state.chat_scope == "alert" else None,
                shipment_id=st.session_state.chat_scope_id if st.session_state.chat_scope == "shipment" else None,
                unique_key=str(id(msg))
            )

        st.write("")

    if st.session_state.chat_latency:
        st.caption(f"⏱ Response time: {st.session_state.chat_latency:.0f} ms")

    st.markdown("---")

    # -------------------------
    # INPUT
    # -------------------------
    query = st.text_area("Ask something...", height=80)

    colA, colB, _ = st.columns([1, 1, 2])

    with colA:
        send = st.button("Send", type="primary")

    with colB:
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.session_state.last_rca = None
            st.session_state.conversation_memory = []
            st.rerun()

    # -------------------------
    # SEND HANDLER
    # -------------------------
    if send:
        last_msg = st.session_state.chat_history[-1] if st.session_state.chat_history else None

        # --- HANDLE SPECIAL COMMANDS (Re-run RCA / Send Email) ---
        if last_msg and last_msg["sender"] == "user" and last_msg["text"] in ["__RE_RUN_RCA__", "__SEND_EMAIL_REPORT__"]:
            special = last_msg["text"]
            st.session_state.chat_history.pop()

            # Re-run RCA
            if special == "__RE_RUN_RCA__":
                user_text = "Please re-run the RCA analysis"

            # ---- EMAIL TRIGGER HERE ----
            elif special == "__SEND_EMAIL_REPORT__":
                from services.mailer import send_mail

                # Require previous RCA
                last_rca = st.session_state.last_rca
                if not last_rca or not last_rca.get("final_json"):
                    st.session_state.chat_history.append({
                        "sender": "agent",
                        "text": "No previous RCA available to send.",
                        "sources": [],
                    })
                    st.rerun()

                # Build subject + email body
                rca_json = last_rca["final_json"]
                subject = f"RCA Report: {rca_json.get('summary', 'Shipment/Alert Report')}"
                body = format_rca_json(rca_json)

                result = send_mail(subject, body)

                st.session_state.chat_history.append({
                    "sender": "agent",
                    "text": f"Email Status: {result.get('status')}\nRecipients: {result.get('to')}",
                    "sources": ["mailer"],
                })
                st.rerun()
    
        else:
            user_text = query.strip()


        if not user_text:
            st.stop()

        # ADD USER MESSAGE
        st.session_state.chat_history.append({"sender": "user", "text": user_text})

        # RUN AGENT
        start = time.time()
        data = chat_agent(
            user_text,
            st.session_state.chat_scope,
            st.session_state.chat_scope_id,
            explain=st.session_state.chat_show_trace,
            debate=st.session_state.debate_mode
        )
        st.session_state.chat_latency = (time.time() - start) * 1000

        # TYPING EFFECT FOR RESPONSE
        typing_effect(data.get("answer", ""))

        st.session_state.chat_history.append({
            "sender": "agent",
            "text": data.get("answer"),
            "trace": data.get("trace"),
            "sources": data.get("sources"),
        })

        st.rerun()
