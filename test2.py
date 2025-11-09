from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
import pandas as pd
import os
import openai
import plotly.express as px

# -----------------------------
# 1️⃣ Load API Key
# -----------------------------
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found. Please set it in your .env file or environment variables.")

openai.api_key = api_key
os.environ["OPENAI_API_KEY"] = api_key  # ensure downstream libs see it

# -----------------------------
# 2️⃣ RCA ALERT DATASET
# -----------------------------
data_samples = {
    'question': [
        'What is the cause of this alert and what are the recommendations to mitigate : {"alert_id": "ALRT_80775", "shipment_id": "SHP1006", "summary": "Detected delay for shipment SHP1006 at Retail_Delhi", "created_at": "2025-11-07T10:45:07.813968", "severity": "High", "status": "Active"}',
        'What is the cause of this alert and what are the recommendations to mitigate : {"alert_id": "ALRT_50014", "shipment_id": "SHP1003", "summary": "Detected delay for shipment SHP1003 at Port_NaviMumbai", "created_at": "2025-11-07T10:45:07.789270", "severity": "Low", "status": "Active"}',
        'What is the cause of this alert and what are the recommendations to mitigate : {"alert_id": "ALRT_85593", "shipment_id": "SHP1013", "summary": "Detected delay for shipment SHP1013 at Factory_01", "created_at": "2025-11-07T10:44:07.376020", "severity": "High", "status": "Active"}'
    ],
    'answer': [
        'The delay for shipment SHP1006 at Retail_Delhi is primarily caused by traffic issues affecting the carrier EcomExpress. Recommended actions: Implement real-time traffic monitoring for routes; Consider alternative routes during peak traffic hours; Enhance communication with EcomExpress for proactive updates.',
        'The delay for shipment SHP1003 is primarily due to it awaiting unloading at the warehouse, compounded by common delays associated with the carrier EcomExpress. Recommended actions: Coordinate with the warehouse to expedite unloading process; Monitor weather conditions and traffic reports to anticipate further delays; Review carrier performance and consider alternative carriers for future shipments if delays persist.',
        'The delay in shipment SHP1013 is attributed to a driver shift delay from Safexpress, reflecting broader operational challenges in driver scheduling. Recommended actions: Engage with Safexpress to improve driver scheduling and availability; Consider alternative carriers for critical shipments to mitigate future delays; Implement a contingency plan for driver shortages, including partnerships with local logistics providers.'
    ],
    'contexts': [
        [
            'Shipment SHP1006 by EcomExpress — Vehicle delayed due to traffic',
            'Shipment SHP1006 by EcomExpress — Vehicle delayed due to traffic',
            'Shipment by EcomExpress — Moderate weather impact, otherwise consistent in South India',
            'Common delay reasons: Heavy rain reported on route, Vehicle delayed due to traffic, Awaiting unloading at warehouse'
        ],
        [
            'Shipment SHP1003 by EcomExpress — Awaiting unloading at warehouse',
            'Shipment SHP1003 by EcomExpress — Awaiting unloading at warehouse',
            'Shipment by EcomExpress — Moderate weather impact, otherwise consistent in South India',
            'Common delay reasons: Heavy rain reported on route, Vehicle delayed due to traffic, Awaiting unloading at warehouse'
        ],
        [
            'Shipment SHP1013 by Safexpress — Driver shift delay',
            'Shipment SHP1013 by Safexpress — Driver shift delay',
            'Shipment by Safexpress — Good network reliability, minor scheduling issues at urban depots',
            'Common delay reasons: Driver shift delay, Awaiting unloading at warehouse'
        ]
    ],
    'ground_truth': [
        'The delay for shipment SHP1006 at Retail_Delhi is primarily caused by traffic issues affecting the carrier EcomExpress.',
        'The delay for shipment SHP1003 is primarily due to it awaiting unloading at the warehouse, compounded by common delays associated with the carrier EcomExpress.',
        'The delay in shipment SHP1013 is attributed to a driver shift delay from Safexpress, reflecting broader operational challenges in driver scheduling.'
    ]
}

dataset = Dataset.from_dict(data_samples)

# -----------------------------
# 3️⃣ Configure LLM and Embeddings
# -----------------------------
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    max_tokens=2048
)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# -----------------------------
# 4️⃣ Evaluate RAGAS Metrics
# -----------------------------
print("🚀 Running RAGAS evaluation on RCA alert dataset...")

try:
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, context_precision, answer_relevancy],
        llm=llm,
        embeddings=embeddings
    )

    df = results.to_pandas()
    print("\n✅ RAGAS Evaluation Complete!\n")
    print(df)

    # Save results
    output_file = "ragas_rca_alert_scores.csv"
    df.to_csv(output_file, index=False)
    print(f"📁 Results saved to {output_file}")

    # df is the DataFrame returned by results.to_pandas()
    # It typically has one row with columns like:
    # ['user_input', 'retrieved_contexts', 'response', 'reference',
    #  'faithfulness', 'context_precision', 'answer_relevancy', ...]
    # We'll extract the metric columns and plot them.

    # metrics we want to display (only those present in df)
    metric_cols = [c for c in ["faithfulness", "context_precision", "answer_relevancy"] if c in df.columns]

    if not metric_cols:
        print("No metric columns found in result dataframe. Columns are:", df.columns.tolist())
    else:
        # take first row (ragas returns aggregated results in a single row)
        scores_series = df.loc[0, metric_cols].astype(float)

        # prepare a tidy dataframe for plotting
        plot_df = scores_series.reset_index()
        plot_df.columns = ["Metric", "Score"]

        import plotly.express as px

        fig = px.bar(
            plot_df,
            x="Metric",
            y="Score",
            text="Score",
            color="Metric",
            color_discrete_sequence=px.colors.qualitative.Bold,
            title="📊 RAGAS Evaluation Metrics — RCA Alert Analysis"
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.update_layout(yaxis_range=[0, 1], uniformtext_minsize=8, uniformtext_mode='hide')
        fig.show()

        # also show as printed table
        print("\nMetric scores:\n", plot_df.to_string(index=False))


except Exception as e:
    print("❌ Error during RAGAS evaluation:", e)
