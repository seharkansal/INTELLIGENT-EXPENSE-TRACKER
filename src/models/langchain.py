import pandas as pd
import json
# Load primary transaction-level dataset
primary_df = pd.read_csv("/home/sehar/INTELLIGENT-EXPENSE-TRACKER/data/external/new_transactions_features.csv", parse_dates=["date"])

# print(primary_df.head())

"""
4️⃣ Format Data for LangChain
a) Transaction-level JSON
"""
transactions_json = primary_df.to_dict(orient="records")

# Load forecast data
with open("/home/sehar/INTELLIGENT-EXPENSE-TRACKER/data/processed/category_forecasts.json", "r") as f:
    forecast_json = json.load(f)

print(forecast_json.keys())  # -> dict_keys(['Food', 'Grocery', 'Utility'])

# print(json.dumps(transactions_json[:5], indent=4, default=str))

#5️⃣ Call LangChain to Generate Insights / Alerts / Recommendations
