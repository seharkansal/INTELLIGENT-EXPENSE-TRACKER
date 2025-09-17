"""
Visualizations:
Trends: is spending generally increasing, decreasing, or flat?
Seasonality: do certain months or cycles have higher spending?
outliers: large spikes or drops that might distort your forecast.
Category patterns: which categories are more volatile vs stable?
"""
#1️⃣ Aggregate per 15th-to-15th cycle
import pandas as pd

# Load features dataset
df = pd.read_csv("data/external/new_transactions_features.csv")
df['date'] = pd.to_datetime(df['date'])

# Define 15th-to-15th month cycle
def assign_cycle(date):
    if date.day >= 15:
        return pd.Timestamp(year=date.year, month=date.month, day=15)
    else:
        prev_month = date.month - 1 if date.month > 1 else 12
        year = date.year if date.month > 1 else date.year - 1
        return pd.Timestamp(year=year, month=prev_month, day=15)

df['cycle_start'] = df['date'].apply(assign_cycle)

# print(df.head(10))

# Aggregate spend per category per cycle
agg = df.groupby(['cycle_start', 'category'])['amount'].sum().reset_index()

#2️⃣ Basic line plots per category
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12,6))
sns.lineplot(data=agg, x='cycle_start', y='amount', hue='category', marker='o')
plt.title("15th-to-15th Cycle Spend per Category")
plt.ylabel("Total Spend ($)")
plt.xlabel("Cycle Start Date")
plt.xticks(rotation=45)
plt.grid(True)
# plt.show()

#3️⃣ Highlight anomalies
# Merge anomaly info
agg_anom = df[df['any_anomaly']].groupby(['cycle_start', 'category'])['amount'].sum().reset_index()
sns.scatterplot(data=agg_anom, x='cycle_start', y='amount', hue='category', s=100, marker='X')
plt.show()
