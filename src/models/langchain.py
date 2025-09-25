import pandas as pd
import json
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
# Load primary transaction-level dataset
from dotenv import load_dotenv

load_dotenv()
primary_df = pd.read_csv("/home/sehar/INTELLIGENT-EXPENSE-TRACKER/data/external/new_transactions_features.csv", parse_dates=["date"])
transactions_json = primary_df.to_dict(orient="records")
with open("/home/sehar/INTELLIGENT-EXPENSE-TRACKER/data/processed/transaction_level.json", "w") as f:
    json.dump(transactions_json, f, indent=4, default=str)

# print(primary_df.head())

"""
4️⃣ Format Data for LangChain
a) Transaction-level JSON
"""

# Load forecast data
with open("/home/sehar/INTELLIGENT-EXPENSE-TRACKER/data/processed/category_forecasts.json", "r") as f:
    forecast_json = json.load(f)

# print(forecast_json.keys())  # -> dict_keys(['Food', 'Grocery', 'Utility'])

# print(json.dumps(transactions_json[:5], indent=4, default=str))

#5️⃣ Call LangChain to Generate Insights / Alerts / Recommendations
langchain_context = {
    "transactions": transactions_json,
    # "forecasts": forecast_json
}
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

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

# Example prompt
prompt = PromptTemplate(
    input_variables=["data_json"],
    template="""
You are a financial assistant. Use the following JSON dataset of transactions and forecasts:

{data_json}

Generate:
1. Monthly summary
2. Category-level insights
3. Alerts for anomalies
4. Recommendations to adjust budgets

Respond in concise, readable English.
"""
)

"""
You are a financial assistant. You are given a dataset of transaction records with the following columns:
date, merchant_clean, amount, category, source, month, cumulative_category_spend, monthly_spend_category,
budget, budget_utilization, monthly_total_spend, other_ratio, rolling_mean, rolling_std,
z_score, spike_anomaly, budget_anomaly, other_anomaly, any_anomaly

You are also given monthly category-level trends.

Tasks:
1. Summarize anomalies (spike, budget, other).
2. Generate insights based on cumulative spend, budget utilization, rolling metrics.
3. Recommend adjustments or warnings per category.
4. Identify unusual trends or overspending.
5. Provide a short summary of the current month's financial health.

Here is the data:
{data}

Provide your response in clear structured points.

"""

"""
You are a financial assistant. You are given:
1. Transaction-level data with cumulative spend, anomalies, budgets, rolling metrics.
2. Monthly category-level trends.
3. Automated alerts based on anomalies and budget utilization.

Your tasks:
1. Summarize anomalies for each category.
2. Provide actionable recommendations for overspending or underspending categories.
3. Highlight trends and forecast context from monthly category trends.
4. Suggest optimal budget adjustments and preventive measures.
5. Present a concise summary of the financial health of the month.

Here is the dataset:
{data}

Alerts:
{alerts}

Respond in a structured format per category.
"""

chain = LLMChain(llm=llm, prompt=prompt)
response = chain.run(data_json=langchain_context)
print(response)