import pandas as pd
from pathlib import Path

def create_features(df):
    # Convert date
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M')
    # Convert 'month' to a more readable format
    if str(df['month'].dtype) == 'period[M]':
        df['month'] = df['month'].dt.to_timestamp().dt.strftime('%B %Y')

    # Monthly spend per category
    # Cumulative monthly spend per category
    df['cumulative_category_spend'] = df.groupby(['month', 'category'])['amount'].cumsum()
    df['monthly_spend_category'] = df.groupby(['month', 'category'])['amount'].transform('sum')

    # Budget mapping and utilization
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
    # Budget utilization (positive ratio)
    df['budget_utilization'] = df['monthly_spend_category'].abs() / df['budget'].replace(0, 1)  # avoid div by 0

    # Monthly total spend
    df['monthly_total_spend'] = df.groupby('month')['amount'].transform('sum')

    # Ratio of "Other" category transactions
    df['other_ratio'] = df.groupby('month')['category'].transform(lambda x: (x=='Other').sum()/len(x))

    # 2. Round numeric columns for readability
    numeric_cols = ["amount", "monthly_spend_category", "budget", 
                    "budget_utilization", "monthly_total_spend", "other_ratio"]
    df[numeric_cols] = df[numeric_cols].round(2)

    df["merchant_clean"] = df["merchant_clean"].str.strip()

    # Rolling mean and std for anomaly detection
    df = df.sort_values('date')
    df['rolling_mean'] = df.groupby('category')['amount'].transform(lambda x: x.rolling(5, min_periods=1).mean()).round(2)
    df['rolling_std'] = df.groupby('category')['amount'].transform(lambda x: x.rolling(5, min_periods=1).std().fillna(0)).round(2)

    # Z-score for anomaly detection
    df['z_score'] = ((df['amount'] - df['rolling_mean']) / df['rolling_std'].replace(0, 1)).round(2)

    # Keep desired columns and order
    df = df[['date', 'merchant_clean', 'amount', 'category', 'month', 'monthly_spend_category',
            'cumulative_category_spend', 'budget', 'budget_utilization',
            'monthly_total_spend', 'other_ratio', 'rolling_mean', 'rolling_std', 'z_score']]

    return df

if __name__ == "__main__":
    # Input: cleaned transactions
    input_path = Path("/home/sehar/INTELLIGENT-EXPENSE-TRACKER/data/interim/train_processed.csv")
    output_path = Path("data/processed/transactions_features.csv")

    df = pd.read_csv(input_path)
    df_features = create_features(df)

    # Ensure output folder exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(output_path, index=False)
    print(f"✅ Feature-engineered dataset saved to {output_path}")

    """
    monthly_spend_category → total spend per category per month.

    cumulative_category_spend → running total, useful to see mid-month spikes.

    budget_utilization → positive ratio of spending vs budget.

    rolling_mean & rolling_std → for anomaly detection using z-score.

    z_score → identifies outlier transactions.

    other_ratio → shows category vs rest of spending in the month.

    Month in words → human-readable.
    """
