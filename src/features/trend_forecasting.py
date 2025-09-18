import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
# from src.features import anamoly_detection

df = pd.read_csv("/home/sehar/INTELLIGENT-EXPENSE-TRACKER/data/external/new_transactions_features.csv")
df['month'] = pd.to_datetime(df['month'], format='%B %Y')
'''
Converts the month column (currently a string like "September 2023") to a datetime object so that time series models can understand the ordering.
'''
#3️⃣ Aggregate Total Monthly Spend
# Assume 'category' or 'source' can help identify credit payments
expenses_df = df[df['amount'] < 0]  # only actual spending
monthly_total = expenses_df.groupby('month')['amount'].sum().sort_index()

monthly_total = monthly_total.asfreq('MS').fillna(method='ffill')

"""
groupby('month') → combines multiple rows for the same month into one number per month.
.asfreq('MS') → sets the time index frequency to month start, filling any missing months.
.fillna(method='ffill') → fills missing months with the previous month’s value to avoid gaps.
"""
#4️⃣ Forecast Total Monthly Spend (Prophet)
prophet_df = monthly_total.reset_index().rename(columns={'month':'ds','amount':'y'})
model_total = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
model_total.fit(prophet_df)

future_total = model_total.make_future_dataframe(periods=6, freq='MS')
forecast_total = model_total.predict(future_total)
print(prophet_df.head())
# ------------------------------
# 5️⃣ Pivot Categories to Wide Format
# ------------------------------
category_monthly = df.pivot_table(
    index='month', 
    columns='category', 
    values='monthly_spend_category', 
    aggfunc='max'
).sort_index()

category_monthly = category_monthly.asfreq('MS').fillna(0)  # fill missing months

print(category_monthly.head())

# ------------------------------
# 6️⃣ Forecast Each Category Separately
# ------------------------------
category_forecasts = {}
for cat in category_monthly.columns:
    cat_df = category_monthly[cat].reset_index().rename(columns={'month':'ds', cat:'y'})
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(cat_df)
    future = model.make_future_dataframe(periods=6, freq='MS')
    forecast = model.predict(future)
    category_forecasts[cat] = forecast

# ------------------------------
# 7️⃣ Visualize Total Monthly Spend
# ------------------------------
plt.figure(figsize=(12,5))
plt.plot(monthly_total.index, monthly_total.values, label='Actual', marker='o')
plt.plot(forecast_total['ds'], forecast_total['yhat'], label='Forecast', linestyle='--')
plt.fill_between(forecast_total['ds'], forecast_total['yhat_lower'], forecast_total['yhat_upper'], color='gray', alpha=0.2)
plt.title('Total Monthly Spend Forecast')
plt.xlabel('Month')
plt.ylabel('Spend ($)')
plt.legend()
plt.show()

# ------------------------------
# 8️⃣ Side-by-Side Category Visualization
# ------------------------------
categories = category_monthly.columns
num_categories = len(categories)
fig, axes = plt.subplots(1, num_categories, figsize=(5*num_categories,5), sharey=True)

for i, cat in enumerate(categories):
    axes[i].plot(category_monthly.index, category_monthly[cat], label='Actual', marker='o')
    forecast = category_forecasts[cat]
    axes[i].plot(forecast['ds'], forecast['yhat'], label='Forecast', linestyle='--')
    axes[i].fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='gray', alpha=0.2)
    axes[i].set_title(f'{cat} Spend')
    axes[i].set_xlabel('Month')
    axes[i].set_ylabel('Spend ($)')
    axes[i].legend()

plt.tight_layout()
plt.show()

# ------------------------------
# 9️⃣ Optional: Highlight Anomalies in Total Spend
# ------------------------------
if 'spike_anomaly' in df.columns:
    anomalies = df[df['spike_anomaly']==True].groupby('month')['monthly_total_spend'].max()
    plt.figure(figsize=(12,5))
    plt.plot(monthly_total.index, monthly_total.values, label='Actual', marker='o')
    plt.scatter(anomalies.index, anomalies.values, color='red', label='Spike Anomaly', s=100)
    plt.title('Total Monthly Spend with Anomalies')
    plt.xlabel('Month')
    plt.ylabel('Spend ($)')
    plt.legend()
    plt.show()
