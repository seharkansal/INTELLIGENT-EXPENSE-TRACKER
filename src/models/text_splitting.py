from collections import defaultdict
from datetime import datetime
import pandas as pd
import json
from pathlib import Path

def split_transactions_by_month(transactions):
    monthly_chunks = defaultdict(list)

    for tx in transactions:
        # Parse date safely, handles "YYYY-MM-DD" and "YYYY-MM-DD HH:MM:SS"
        dt = pd.to_datetime(tx["date"])
        month_key = dt.strftime("%Y-%m")

        monthly_chunks[month_key].append(tx)

    # Build structured chunks (month + transactions)
    chunks = [
        {"month": m, "transactions": monthly_chunks[m]}
        for m in sorted(monthly_chunks.keys())
    ]
    return chunks

# with open("/home/sehar/INTELLIGENT-EXPENSE-TRACKER/data/processed/transaction_level.json", "r") as f:
#     transactions_json = json.load(f)

# chunks = split_transactions_by_month(transactions_json)

# # Save to JSON file
# with open("/home/sehar/INTELLIGENT-EXPENSE-TRACKER/data/processed/monthly_chunks.json", "w") as f:
#     json.dump(chunks, f, indent=4)

# # Show how the chunks look
# print(json.dumps(chunks, indent=4))


"""
all_summaries = []

for month_json in all_months_chunks:
    summary = llm_chain.run(data_json=month_json)
    all_summaries.append(summary)

# Now do a second LLM call with summaries
final_response = llm_chain.run(data_json="\n".join(all_summaries))
print(final_response)

"""

# ----------------------------
# CONFIG
# ----------------------------
CHUNK_DIR = Path("monthly_chunks")
CHUNK_DIR.mkdir(exist_ok=True)
VECTOR_STORE_PATH = Path("faiss_index")
VECTOR_STORE_PATH.mkdir(exist_ok=True)
# ----------------------------
# 1️⃣ Load transactions CSV
# ----------------------------
df = pd.read_csv("/home/sehar/INTELLIGENT-EXPENSE-TRACKER/data/external/new_transactions_features.csv", parse_dates=["date"])

# Load big JSON
# with open("/home/sehar/INTELLIGENT-EXPENSE-TRACKER/data/processed/monthly_chunks.json") as f:
#     all_transactions = json.load(f)

# # Iterate over the list
# for month_dict in all_transactions:
#     month = month_dict["month"]
#     transactions = month_dict["transactions"]
#     month_path = CHUNK_DIR / f"{month}.json"
#     with open(month_path, "w") as f:
#         json.dump(transactions, f, indent=4)

"""
apending new transactions:
Existing month:
The month JSON already exists in month_chunks/.
we just need to load it, append the new transactions, deduplicate, and save.

New month:
The month JSON does not exist yet.
we create a new JSON file, write the transactions into it, and save it in month_chunks/.
"""
def process_new_transactions(new_transactions):
    for tx_dict in new_transactions:
        month_path = CHUNK_DIR / f"{tx_dict['month']}.json"

        if month_path.exists():
            with open(month_path, "r") as f:
                month_data = json.load(f)
        else:
            month_data = []

        print("Processing transaction:")
        print(tx_dict)
        print("---")

        month_data.append(tx_dict)

        # Deduplicate
        seen = set()
        unique_transactions = []
        for t in month_data:
            key = (t["date"], t["merchant_clean"], t["amount"])
            if key not in seen:
                seen.add(key)
                unique_transactions.append(t)

        with open(month_path, "w") as f:
            json.dump(unique_transactions, f, indent=4)

    print("New transactions processed successfully.")

# 2️⃣ Select new transactions
mask = df["source"] == "new"
new_df = df[mask]

if new_df.empty:
    print("No new transactions found in the dataframe.")
else:
    # Convert to list of dicts
    new_transactions = []
    for _, tx in new_df.iterrows():
        tx_dict = tx.to_dict()
        if isinstance(tx_dict["date"], pd.Timestamp):
            tx_dict["date"] = tx_dict["date"].strftime("%Y-%m-%d")  # make JSON-safe
        tx_dict["month"] = tx_dict["date"][:7]  # works since now it's a string
        new_transactions.append(tx_dict)

    # Append to month JSONs
    process_new_transactions(new_transactions)
    # 5️⃣ Mark these rows as historical in the dataframe
    df.loc[mask, "source"] = "historical"

    # 6️⃣ Save back to the CSV
    df.to_csv("/home/sehar/INTELLIGENT-EXPENSE-TRACKER/data/external/new_transactions_features.csv", index=False)
    print(f"Marked {mask.sum()} transactions as historical in CSV.")

