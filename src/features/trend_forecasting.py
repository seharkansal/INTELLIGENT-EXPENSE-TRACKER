import pandas as pd
import numpy as np
from prophet import Prophet
# from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import json

# ------------------------------
# 1️⃣ Create Category-Level Monthly Data
# ------------------------------
df = pd.read_csv("/home/sehar/INTELLIGENT-EXPENSE-TRACKER/data/external/new_transactions_features.csv")
df['month'] = pd.to_datetime(df['month'], format='%B %Y')

category_monthly = df.pivot_table(
    index='month',
    columns='category',
    values='monthly_spend_category',
    aggfunc='max' 
).sort_index()

category_monthly = category_monthly.asfreq('MS').fillna(0)
category_monthly.to_csv("/home/sehar/INTELLIGENT-EXPENSE-TRACKER/data/external/category_spend_monthly")
# print(category_monthly)

holidays = pd.DataFrame({
    'holiday': 'high_spend',
    'ds': pd.to_datetime(['2025-05-01', '2025-06-01', '2024-10-01']),  # months with spikes
    'lower_window': 0,  # include this day
    'upper_window': 0
})

# ------------------------------
# 2️⃣ Forecast Each Category
# ------------------------------
category_forecasts = {}
# metrics = {}

for cat in ["Food", "Grocery", "Utility"]:
    print(f"\nForecasting category: {cat}")

    # Prepare data
    cat_df = category_monthly[cat].reset_index().rename(columns={'month': 'ds', cat: 'y'})
    cat_df['y_smooth'] = cat_df['y'].rolling(2, center=True).median()
    # print(cat_df.head())
    
    
    # Prophet model
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,changepoint_prior_scale=0.3,holidays=holidays)
    model.fit(cat_df)
    
    # Forecast (extend 3 months ahead for test comparison)
    future = model.make_future_dataframe(periods=3, freq='MS')
    forecast = model.predict(future)
    
    # Save forecast
    category_forecasts[cat] = forecast

forecast_json = {}
for cat, forecast in category_forecasts.items():
    forecast_json[cat] = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(5).to_dict(orient="records")

    # Save or pretty print
# print(json.dumps(forecast_json, indent=4, default=str))
# Save forecast_json to a file
with open("/home/sehar/INTELLIGENT-EXPENSE-TRACKER/data/processed/category_forecasts.json", "w") as f:
    json.dump(forecast_json, f, indent=4, default=str)
    # print(category_forecasts)

# fig1 = model.plot(forecast)
# plt.title(f"Forecast for {cat}")
# plt.show()

# fig2 = model.plot_components(forecast)
# plt.title(f"Components for {cat}")
# plt.show()

