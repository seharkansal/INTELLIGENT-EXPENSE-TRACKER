import pandas as pd
from pathlib import Path
import datetime
from src.logger import logging
import pandas as pd
from pathlib import Path
import os

def create_features(df):
    # Convert date
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.strftime('%B %Y')
    # df = df.sort_values('date').reset_index(drop=True)

    # # Monthly spend per category
    # # Cumulative monthly spend per category
    df['cumulative_category_spend'] = df.groupby(['month', 'category'])['amount'].cumsum()
    df['monthly_spend_category'] = df.groupby(['month', 'category'])['amount'].transform('sum')

    # # Budget mapping and utilization
    budgets = {
        "Food": 120,
        "Shopping": 0,
        "Grocery": 110,
        "Bills": 300,
        "Utility": 39,
        "trip": 10,
        "cashback": 10,
        "Other": 100
    }
    df['budget'] = df['category'].map(budgets).fillna(0)
    # # Budget utilization (positive ratio)
    df['budget_utilization'] = df['monthly_spend_category'].abs() / df['budget'].replace(0, 1)  # avoid div by 0

    df['monthly_total_spend'] = (
        df.assign(expense_only = df['amount'].where(df['amount'] < 0, 0))
        .groupby('month')['expense_only']
        .transform('sum')
)
    # # Ratio of "Other" category transactions
    df['other_ratio'] = df.groupby('month')['category'].transform(lambda x: (x=='Other').sum()/len(x))

    # # 2. Round numeric columns for readability
    # numeric_cols = ["amount", "monthly_spend_category", "budget", 
    #                 "budget_utilization", "monthly_total_spend", "other_ratio"]
    # df[numeric_cols] = df[numeric_cols].round(2)

    df["merchant_clean"] = df["merchant_clean"].str.strip()

    # # Rolling mean and std for anomaly detection
    df['rolling_mean'] = df.groupby('category')['amount'].transform(lambda x: x.rolling(5, min_periods=1).mean()).round(2)
    df['rolling_std'] = df.groupby('category')['amount'].transform(lambda x: x.rolling(5, min_periods=1).std().fillna(0)).round(2)

    # # Z-score for anomaly detection
    df['z_score'] = ((df['amount'] - df['rolling_mean']) / df['rolling_std'].replace(0, 1)).round(2)

    numeric_cols = ["amount", "monthly_spend_category", "budget", 
                    "budget_utilization","cumulative_category_spend", "monthly_total_spend", "other_ratio","rolling_mean", "rolling_std", "z_score"]
    df[numeric_cols] = df[numeric_cols].round(2)

    # # Keep desired columns and order
    # df = df[['date', 'merchant_clean', 'amount', 'category', 'month', 'monthly_spend_category',
    #         'cumulative_category_spend', 'budget', 'budget_utilization',
    #         'monthly_total_spend', 'other_ratio', 'rolling_mean', 'rolling_std', 'z_score']]

    return df

def detect_anomalies(df, z_thresh=3, budget_thresh=1, other_ratio_thresh=0.3, combine_logic='any'):
    # Spike anomaly
    df['spike_anomaly'] = df['z_score'].abs() > z_thresh
    
    # Budget anomaly
    df['budget_anomaly'] = df['budget_utilization'] > budget_thresh
    
    # Other category anomaly
    monthly_other_ratio = df.groupby('month')['category'].transform(
        lambda x: (x=='Other').sum() / len(x))
    df['other_anomaly'] = monthly_other_ratio > other_ratio_thresh
    
    # Combine anomalies
    if combine_logic == 'any':
        df['any_anomaly'] = df[['spike_anomaly','budget_anomaly','other_anomaly']].any(axis=1)
    elif combine_logic == 'majority':
        df['any_anomaly'] = df[['spike_anomaly','budget_anomaly','other_anomaly']].sum(axis=1) >= 2
    else:
        raise ValueError("combine_logic must be 'any' or 'majority'")
    
    return df

def evaluate_metrics(df, ground_truth_col='any_anomaly'):
    predicted = df['any_anomaly']
    ground_truth = df[ground_truth_col]

    TP = ((predicted==True) & (ground_truth==True)).sum()
    FP = ((predicted==True) & (ground_truth==False)).sum()
    FN = ((predicted==False) & (ground_truth==True)).sum()
    TN = ((predicted==False) & (ground_truth==False)).sum()

    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

    return {'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN, 'Precision': precision, 'Recall': recall, 'F1': f1}

# # Paths
feature_path = Path("/home/sehar/INTELLIGENT-EXPENSE-TRACKER/data/external/new_transactions_features.csv")
raw_path=Path("/home/sehar/INTELLIGENT-EXPENSE-TRACKER/data/interim/processed_data.csv")
# historic_anomalies_path = Path("/home/sehar/INTELLIGENT-EXPENSE-TRACKER/data/external/historic_anomalies.csv")
new_anomalies_path = Path("/home/sehar/INTELLIGENT-EXPENSE-TRACKER/data/external/new_anomalies.csv")
new_email_path = Path("/home/sehar/INTELLIGENT-EXPENSE-TRACKER/data/interim/new_email_transaction_data.csv")

def main():
    # 1.  Load historical features
    raw_transcation = pd.read_csv(raw_path)
    print("raw data loaded")
    
    # 2️⃣ Load new email transactions
    df_new = pd.read_csv(new_email_path)
    print("email data loaded")
    
    if df_new.empty:
        logging.info("No new email transactions.")
        print("No new email transactions.")
        return

    # Run feature engineering on new transactions
    # df_new_features = create_features(df_new)

    # 3️⃣ Append new transactions to historical raw data
     # 3️⃣ Keep only new rows (source=='new') from email alerts
    df_new_only = df_new[df_new['source'] == 'new'].copy()
    raw_transcation['source'] = 'historical'
    combined_raw = pd.concat([raw_transcation, df_new_only]).reset_index(drop=True)

    # combined_raw = pd.concat([raw_transcation, df_new]).reset_index(drop=True)
    combined_raw = combined_raw.drop_duplicates(subset=["date", "merchant_clean", "amount"], keep="last")
    print(combined_raw.head())

    combined_raw.to_csv(raw_path, index=False)
    # Sort by date ascending (oldest first)
    # # combined_raw = combined_raw.sort_values(by="date").reset_index(drop=True)

    # # # 4️⃣ Run feature engineering on full dataset
    combined_features = create_features(combined_raw)
    # # combined_features.to_csv(feature_path, index=False)

    # # # After anomaly detection and saving
    # # combined_features.loc[combined_features['source'] == 'new', 'source'] = 'historical'

    # # #5️⃣ Detect anomalies on full feature dataset
    anamoly_dataset = detect_anomalies(combined_features)
    # #storing historic anamolies
    anamoly_dataset.to_csv(feature_path, index=False)

    anomalies_new = (combined_features['any_anomaly'].astype(bool)) | (combined_features['source'] == 'new')

    print(f"Detected {len(anomalies_new)} anomalies in new transactions.")

    # # Step 2: Append new anomalies to existing anomalies CSV
    if new_anomalies_path.exists() and os.path.getsize(new_anomalies_path) > 0:
        try:
            existing_anomalies = pd.read_csv(new_anomalies_path)
        except pd.errors.EmptyDataError:
            existing_anomalies = pd.DataFrame(columns=anomalies_new.columns)
    else:
        existing_anomalies = pd.DataFrame(columns=anomalies_new.columns)

    # # Combine and remove duplicates
    combined_anomalies = pd.concat([existing_anomalies, anomalies_new], ignore_index=True)
    combined_anomalies = combined_anomalies.drop_duplicates(
        subset=['date', 'merchant_clean', 'amount', 'category'], keep='last'
    )

    # # Save updated anomalies dataset
    # combined_anomalies.to_csv(new_anomalies_path, index=False)
    logging.info(f"✅ Updated anomalies dataset saved with {len(combined_anomalies)} rows.")

    # # Step 3: Update 'source' for new transactions to historical
    # combined_features.loc[combined_features['source'] == 'new', 'source'] = 'historical'

    # # Step 4: Save updated features dataset
    combined_features.to_csv(feature_path, index=False)
    logging.info(f"✅ Updated features dataset saved with {len(combined_features)} rows.")

    print("Anomalies processing complete.")

    # Set your labeled anomalies as ground truth
    df = pd.read_csv(feature_path, parse_dates=['date'])
    print(df.head())
    df['ground_truth'] = df['any_anomaly']

    # Try different thresholds
    z_scores = [2.5, 3, 3.5]
    budget_thresholds = [1, 1.1]
    other_ratios = [0.2, 0.3, 0.4]

    results = []

    for z in z_scores:
        for b in budget_thresholds:
            for o in other_ratios:
                temp_df = detect_anomalies(df.copy(), z_thresh=z, budget_thresh=b, other_ratio_thresh=o, combine_logic='any')
                metrics = evaluate_metrics(temp_df, ground_truth_col='ground_truth')
                metrics.update({'z_thresh': z, 'budget_thresh': b, 'other_ratio_thresh': o})
                results.append(metrics)

    # Convert results to DataFrame for easy comparison
    results_df = pd.DataFrame(results)
    # print(results_df.sort_values(by='F1', ascending=False))

if __name__ == "__main__":
    main()
