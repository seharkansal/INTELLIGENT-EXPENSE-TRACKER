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

.max() → in your data, the total spend per month is repeated per transaction, so max() just takes one value.

.asfreq('MS') → sets the time index frequency to month start, filling any missing months.

.fillna(method='ffill') → fills missing months with the previous month’s value to avoid gaps.
"""
#4️⃣ Forecast Total Monthly Spend (Prophet)
prophet_df = monthly_total.reset_index().rename(columns={'month':'ds','monthly_total_spend':'y'})
# model_total = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
# model_total.fit(prophet_df)

# future_total = model_total.make_future_dataframe(periods=6, freq='MS')
# forecast_total = model_total.predict(future_total)
print(prophet_df)

