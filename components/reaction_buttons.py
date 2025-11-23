import streamlit as st

def recommended_actions(alert_id=None, shipment_id=None, unique_key="0"):
    """
    Renders action buttons depending on alert/shipment context.
    unique_key ensures Streamlit never complains about duplicate widgets.
    """

    # Nothing to show
    if not alert_id and not shipment_id:
        return

    st.write("")  
    st.markdown("### Recommended Actions")

    col1, col2 = st.columns(2)

    with col1:
        rerun = st.button(
            "🔄 Re-run RCA",
            key=f"re_run_{unique_key}"
        )
    with col2:
        email = st.button(
            "📧 Send Email Summary",
            key=f"email_{unique_key}"
        )

    if rerun:
        st.session_state.chat_history.append({
            "sender": "user",
            "text": "__RE_RUN_RCA__"
        })
        st.rerun()

    if email:
        st.session_state.chat_history.append({
            "sender": "user",
            "text": "__SEND_EMAIL_REPORT__"
        })
        st.rerun()
