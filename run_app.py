import streamlit as st
from dotenv import load_dotenv
import os, json

# ---- Load environment variables ----
load_dotenv()

# ---- Streamlit config ----
st.set_page_config(
    page_title="CPG Supply Chain AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Custom styling ----
st.markdown("""
<style>
    body, .stApp { background-color:#0e1117; color:#d7e3f0; }
    .block-container { padding-top: 1rem; }
    section[data-testid="stSidebar"] {
        background-color: #111418 !important;
        border-right: 1px solid #2b3035;
    }
    .sidebar-title { font-size:1.2rem; font-weight:600; color:#6ab0f8; margin-bottom:0.8rem; }
</style>
""", unsafe_allow_html=True)

# ---- Simple authentication ----
def login_widget():
    """Renders login form if user not logged in."""
    if "user" not in st.session_state:
        st.session_state.user = None

    if st.session_state.user is None:
        with st.form("login_form"):
            st.markdown("## 🔐 Login to CPG AI Dashboard")
            username = st.text_input("👤 Username")
            password = st.text_input("🔑 Password", type="password")
            submit = st.form_submit_button("Login")

            if submit:
                os.makedirs("config", exist_ok=True)
                try:
                    users = json.load(open("config/users.json"))
                except FileNotFoundError:
                    users = {"admin": "admin123", "user": "user123"}
                    json.dump(users, open("config/users.json", "w"), indent=2)

                if username in users and users[username] == password:
                    st.session_state.user = username
                    st.success(f"✅ Welcome, {username}!")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials")
        st.stop()

login_widget()
username = st.session_state.user

# ---- Sidebar navigation ----
st.sidebar.markdown('<core class="sidebar-title">🚛 CPG Supply Chain AI</core>', unsafe_allow_html=True)

# Define all core
all_pages = {
    "Dashboard": "1_Dashboard.py",
    "Alerts": "2_Alerts.py",
    "Admin / Data Integrator": "3_Admin_Data_Integrator.py",
    "Admin / Guardrails": "4_Admin_Guardrails.py",
    "Admin / Metrics (RAGAS)": "5_Admin_Metrics.py",
    "Chat Assistant": "6_Chat.py",
}

# ---- Role-based filtering ----
admin_only_pages = [p for p in all_pages if p.startswith("Admin")]
if username != "admin":
    pages = {k: v for k, v in all_pages.items() if not k.startswith("Admin")}
else:
    pages = all_pages

# ---- Sidebar menu ----
choice = st.sidebar.radio("Navigate", list(pages.keys()))
page_file = pages[choice]
page_path = os.path.join("core", page_file)

# ---- Logout button ----
st.sidebar.markdown("---")
if st.sidebar.button("🚪 Logout"):
    st.session_state.user = None
    st.rerun()

# ---- Load core dynamically ----
if not os.path.exists(page_path):
    st.error(f"❌ Page file not found: `{page_path}`")
    st.stop()

# ---- Security for Admin Pages ----
if choice.startswith("Admin") and username != "admin":
    st.error("🚫 You do not have permission to access this core.")
    st.stop()

# ---- Execute the selected core ----
try:
    with open(page_path, "r", encoding="utf8") as f:
        code = compile(f.read(), page_path, 'exec')
        exec(code, globals())
except Exception as e:
    st.error(f"⚠️ Error loading core `{choice}`:\n\n{e}")
