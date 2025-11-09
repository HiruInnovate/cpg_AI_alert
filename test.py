from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_correctness
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
import pandas as pd
import openai
import os

# --- Step 1: Load API Key ---
load_dotenv()  # Loads variables from .env file if available

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found. Please set it in your .env or environment variables.")

# Configure both OpenAI SDKs
openai.api_key = api_key
os.environ["OPENAI_API_KEY"] = api_key  # ensures ragas/langchain pick it up

# --- Step 2: Prepare Sample Dataset ---
data_samples = {
    "question": [
        "When was the first super bowl?",
        "Who won the most super bowls?"
    ],
    "answer": [
        "The first superbowl was held on Jan 15, 1967",
        "The most super bowls have been won by The New England Patriots"
    ],
    "contexts": [
        [
            "The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles,"
        ],
        [
            "The Green Bay Packers...Green Bay, Wisconsin.",
            "The Packers compete...Football Conference"
        ]
    ],
    "ground_truth": [
        "The first superbowl was held on January 15, 1967",
        "The New England Patriots have won the Super Bowl a record six times"
    ]
}

dataset = Dataset.from_dict(data_samples)

# --- Step 3: Define Models for RAGAS ---
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    max_tokens=2048
)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# --- Step 4: Run Evaluation ---
print("🚀 Running RAGAS evaluation...")

score = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_correctness],
    llm=llm,
    embeddings=embeddings
)

# --- Step 5: Save Results ---
df = score.to_pandas()
print("✅ RAGAS Evaluation Complete!")
print(df)

df.to_csv("score.csv", index=False)
print("📁 Saved to score.csv")
