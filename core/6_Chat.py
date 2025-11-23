# core/6_Chat.py
import streamlit as st
from components.chat_ui import chat_ui

st.header("💬 Chat with RCA Agent")
st.write("Interactive RCA engine with timelines, maps and operational actions.")

chat_ui()
