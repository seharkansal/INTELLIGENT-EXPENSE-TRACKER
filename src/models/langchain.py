import json
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.document_loaders import JSONLoader
from dotenv import load_dotenv

load_dotenv()
# primary_df = pd.read_csv("/home/sehar/INTELLIGENT-EXPENSE-TRACKER/data/external/new_transactions_features.csv")
# transactions_json = primary_df.to_dict(orient="records")
# with open("/home/sehar/INTELLIGENT-EXPENSE-TRACKER/data/processed/transaction_level.json", "w") as f:
#     json.dump(transactions_json, f, indent=4, default=str)
"""
# Load forecast data
with open("/home/sehar/INTELLIGENT-EXPENSE-TRACKER/data/processed/category_forecasts.json", "r") as f:
    forecast_json = json.load(f)

# print(forecast_json.keys())  # -> dict_keys(['Food', 'Grocery', 'Utility'])

# 3️⃣ Generate automated alerts
# -------------------------------
# alerts = []

# for _, row in primary_df.iterrows():
#     category = row["category"]
#     month = row["month"].strftime("%B %Y")
    
#     if row["spike_anomaly"]:
#         alerts.append(f"Spike anomaly detected in {category} on {row['date'].strftime('%Y-%m-%d')}")
    
#     if row["budget_anomaly"]:
#         alerts.append(f"Budget exceeded in {category} for {month}")
    
#     if row["other_anomaly"]:
#         alerts.append(f"Other anomaly in {category} on {row['date'].strftime('%Y-%m-%d')}")
"""
CHUNK_DIR = Path("monthly_chunks")
SUMMARY_DIR = Path("summaries")
SUMMARY_DIR.mkdir(exist_ok=True)

def build_vector_store():
    chunk_texts = []
    month_refs = []
    for month_file in CHUNK_DIR.iterdir():
        with open(month_file, "r") as f:
            data = json.load(f)
        chunk_texts.append(str(data))
        month_refs.append(month_file.stem)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_texts(chunk_texts, embeddings, metadatas=[{"month": m} for m in month_refs])
    return vector_store

# Step 1: build or update vector store for retrieval
vector_store = build_vector_store()

prompt = PromptTemplate(
    input_variables=["data_json"],
    template="""
You are a financial analyst.

Analyze the transactions from the JSON data below:
{data_json}

Format like this:
<Category Name>
Monthly spend: $X vs budget $Y → Z% overspend/underspend (state if budget anomaly).
Flagged transactions: list any anomalies with merchant name and reason.
Rolling average spend: $A (std B) → explain if current spend is within/outside normal range.

Repeat for each category.

Overall summary:
- Total spend: $TOTAL
- Categories exceeding budget: list them with % over
- Recommendations: actionable advice in 1–2 sentences

Provide your response in clear structured points.
"""
)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)
chain = LLMChain(llm=llm, prompt=prompt)
# Step 2: retrieval & LLM analysis (example)
# 8️⃣ Function to retrieve relevant chunks and run LLM
def analyze_month(month_str):
    # Retrieve chunks for that month
    results = vector_store.similarity_search(month_str, k=1)  # adjust k if multiple chunks
    data_json = results[0].page_content
    # Run LLM
    return chain.run(data_json=data_json)

# 9️⃣ Example usage
month_to_analyze = "2023-10"
response = analyze_month(month_to_analyze)
print(f"Analysis for {month_to_analyze}:\n", response)