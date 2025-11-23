import streamlit as st

def trace_timeline(trace_text: str):
    """
    Converts trace text into a clean step-by-step timeline UI.
    """
    if not trace_text:
        st.info("No reasoning trace available.")
        return

    st.markdown("### 📝 Reasoning Timeline")

    steps = [s.strip() for s in trace_text.split(".") if s.strip()]

    for i, step in enumerate(steps, start=1):
        st.markdown(
            f"""
            <div style='margin-bottom:12px;'>
                <div style='
                    background:#1e293b; 
                    color:white; 
                    padding:8px 12px; 
                    border-radius:6px; 
                    display:inline-block;
                '>
                    Step {i}
                </div>
                <div style='padding:8px 12px; background:#fff; border-radius:6px;'>
                    {step}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
