# ------------------------------
# Prophet Forecasting with Outlier Handling & Plotly Visualization
# ------------------------------

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import plotly.graph_objs as go
import plotly.io as pio

pio.renderers.default = "browser"

# 1️⃣ Load Data
df = pd.read_csv("/home/sehar/INTELLIGENT-EXPENSE-TRACKER/data/external/new_transactions_features.csv")
df['month'] = pd.to_datetime(df['month'], format='%B %Y')

# 2️⃣ Aggregate Total Monthly Spend (only expenses)
expenses_df = df[df['amount'] < 0]
monthly_total = expenses_df.groupby('month')['amount'].sum().sort_index()

# Fill missing months
monthly_total = monthly_total.asfreq('MS').fillna(method='ffill')

holidays = pd.DataFrame({
    'holiday': 'high_spend',
    'ds': pd.to_datetime(['2025-05-01', '2025-06-01', '2024-10-01']),  # months with spikes
    'lower_window': 0,  # include this day
    'upper_window': 0
})

# 3️⃣ Handle Extreme Outliers
prophet_df = monthly_total.reset_index().rename(columns={'month':'ds', 'amount':'y'})

# Cap extreme negative spends
lower_cap = -1000  # you can adjust
upper_cap = -10

prophet_df['y_clean'] = prophet_df['y'].clip(lower=lower_cap, upper=upper_cap)
prophet_df = prophet_df[prophet_df['ds'] != '2025-06-01']


# Optional: smooth minor spikes with rolling median
prophet_df['y_smooth'] = prophet_df['y_clean'].rolling(3, center=True).median()
prophet_df['y_smooth'].fillna(method='bfill', inplace=True)
prophet_df['y_smooth'].fillna(method='ffill', inplace=True)

print(prophet_df)

# 4️⃣ Split Train/Test for Backtesting
train = prophet_df.iloc[:-3]
test = prophet_df.iloc[-3:]

# 5️⃣ Fit Prophet Model on Cleaned Data
model = Prophet(yearly_seasonality=True, holidays=holidays,changepoint_prior_scale=0.6)
model.fit(train[['ds', 'y_smooth']].rename(columns={'y_smooth':'y'}))

# 6️⃣ Make Future Predictions
future = model.make_future_dataframe(periods=3, freq='MS')
forecast = model.predict(future)

# 7️⃣ Backtesting Metrics
y_pred = forecast['yhat'].iloc[-3:].values
y_true = test['y_smooth'].values
mae = mean_absolute_error(y_true, y_pred)
rmse = root_mean_squared_error(y_true, y_pred)
print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

# 8️⃣ Visualization with Plotly
fig = go.Figure()

# Actuals
fig.add_trace(go.Scatter(
    x=prophet_df['ds'],
    y=prophet_df['y'],
    mode='lines+markers',
    name='Actual Spend',
    line=dict(color='black')
))

# Forecast
fig.add_trace(go.Scatter(
    x=forecast['ds'],
    y=forecast['yhat'],
    mode='lines',
    name='Forecast',
    line=dict(color='blue')
))

# Test Actuals Highlight
fig.add_trace(go.Scatter(
    x=test['ds'],
    y=test['y'],
    mode='markers',
    name='Test Actuals',
    marker=dict(color='red', size=10)
))

fig.update_layout(
    title='Monthly Spend Forecast vs Actuals',
    xaxis_title='Month',
    yaxis_title='Monthly Spend ($)',
    legend=dict(x=0.02, y=0.98)
)

fig.show()
