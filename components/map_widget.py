import streamlit as st
import pandas as pd

def shipment_map(shipment_id: str):
    st.markdown(f"### 🌍 Shipment Map — {shipment_id}")

    # IMPORTANT: clear floating divs from chat bubbles
    st.markdown("<div style='clear:both;'></div>", unsafe_allow_html=True)

    # Fake coordinates for demo (Mumbai)
    df = pd.DataFrame({
        "lat": [19.0760],
        "lon": [72.8777],
    })

    # Always ensure float types
    df["lat"] = df["lat"].astype(float)
    df["lon"] = df["lon"].astype(float)

    # Render map
    st.map(df, zoom=8)
