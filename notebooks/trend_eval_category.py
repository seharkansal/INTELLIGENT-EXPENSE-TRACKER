import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

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
print(category_monthly)

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
metrics = {}

for cat in category_monthly.columns:
    print(f"\nForecasting category: {cat}")

    # Prepare data
    cat_df = category_monthly[cat].reset_index().rename(columns={'month': 'ds', cat: 'y'})
    cat_df['y_smooth'] = cat_df['y'].rolling(2, center=True).median()
    # print(cat_df.head())
    
    # Train/test split (last 3 months as test)
    train = cat_df.iloc[:-3]
    test = cat_df.iloc[-3:]
    
    # Prophet model
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,changepoint_prior_scale=0.3,holidays=holidays)
    model.fit(train)
    
    # Forecast (extend 3 months ahead for test comparison)
    future = model.make_future_dataframe(periods=3, freq='MS')
    forecast = model.predict(future)
    
    # Save forecast
    category_forecasts[cat] = forecast
    
    # ------------------------------
    # 3️⃣ Evaluate Forecast
    # ------------------------------
    y_pred = forecast['yhat'].iloc[-3:].values
    y_true = test['y'].values

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics[cat] = {"MAE": mae, "RMSE": rmse}

    print(f"  MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    # ------------------------------
    # 4️⃣ Plot Actual vs Forecast
    # ------------------------------
    # plt.figure(figsize=(10, 5))
    # plt.plot(cat_df['ds'], cat_df['y'], label="Actual")
    # plt.plot(forecast['ds'], forecast['yhat'], label="Forecast")
    # plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.2)
    # plt.axvline(test['ds'].iloc[0], color='red', linestyle='--', label="Train/Test Split")
    # plt.title(f"Category: {cat}")
    # plt.legend()
    # plt.show()

# ------------------------------
# 5️⃣ Summarize Metrics
# ------------------------------
metrics_df = pd.DataFrame(metrics).T
print("\nCategory-level Forecast Accuracy:")
print(metrics_df)
