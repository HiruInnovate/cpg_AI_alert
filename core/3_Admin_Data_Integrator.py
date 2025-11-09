import streamlit as st
from services.ingestion import save_upload, process_file

st.header("🗂 Admin — Data Integrator")

st.markdown("""
Upload **any** of the following data types to enrich your RCA pipeline:

- 🟦 **Shipment Data** (`shipment_id`, `event_type`, `carrier`, etc.)
- 🟩 **News Data** (`headline`, `title`, `body`)
- 🟨 **Social Data** (`tweet`, `post`, or `content`)
- 🟧 **Logistics Performance Data** (`carrier_name`, `on_time_delivery_rate_percent`, etc.)

Once uploaded, the system will automatically detect the data type, normalize it,
and index it into the appropriate vector store namespace.
""")

uploaded = st.file_uploader(
    "Upload file for ingestion",
    type=["csv", "json", "txt"],
    help="Upload your dataset (Shipments, News, Social, or Logistics data)."
)

if uploaded:
    with st.spinner("🔍 Analyzing and processing your file..."):
        path = save_upload(uploaded)
        res = process_file(path)
    st.success("✅ File successfully processed!")
    st.json(res)

st.divider()

st.markdown("""
### 🧠 Tips
- Upload **`shipment_events.json`** for shipment data
- Upload **`logistics_performance.json`** for carrier performance data
- Upload **news** or **social** CSV/JSON files to enrich external context  
""")
