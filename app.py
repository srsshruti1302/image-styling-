import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import timedelta

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="InsightForge BI Dashboard",
    layout="wide"
)

st.title("📊 InsightForge: LSTM-Driven GenAI Business Intelligence Dashboard")

# -------------------------------
# Upload Company Data
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload Company CSV File",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload a company CSV file to continue.")
    st.stop()

df = pd.read_csv(uploaded_file)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# -------------------------------
# Feature Selection
# -------------------------------
features = [
    'Sales',
    'Orders',
    'Customers',
    'MarketingSpend',
    'Returns'
]

target = 'Revenue'

# -------------------------------
# KPI Section
# -------------------------------
latest_revenue = df[target].iloc[-1]
avg_revenue = df[target].mean()
growth = ((df[target].iloc[-1] - df[target].iloc[-2]) /
          df[target].iloc[-2]) * 100

col1, col2, col3 = st.columns(3)

col1.metric("💰 Latest Revenue", f"{latest_revenue:.2f}")
col2.metric("📊 Average Revenue", f"{avg_revenue:.2f}")
col3.metric("📈 Growth (%)", f"{growth:.2f}%")

# -------------------------------
# Scale Data
# -------------------------------
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(df[features])
y_scaled = scaler_y.fit_transform(df[[target]])

# -------------------------------
# Create Sequences
# -------------------------------
def create_sequences(X, y, window=10):
    X_seq, y_seq = [], []
    for i in range(len(X) - window):
        X_seq.append(X[i:i+window])
        y_seq.append(y[i+window])
    return np.array(X_seq), np.array(y_seq)

window = 10
X, y = create_sequences(X_scaled, y_scaled, window)

# -------------------------------
# LSTM Model
# -------------------------------
model = Sequential([
    LSTM(64, return_sequences=True,
         input_shape=(window, len(features))),
    LSTM(32),
    Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mse'
)

model.fit(X, y, epochs=20, batch_size=16, verbose=0)

# -------------------------------
# Forecast Future Revenue
# -------------------------------
future_days = 15
last_seq = X_scaled[-window:]
forecast = []

current_seq = last_seq.copy()

for _ in range(future_days):
    pred = model.predict(
        current_seq.reshape(1, window, len(features)),
        verbose=0
    )
    forecast.append(pred[0][0])

    next_features = current_seq[-1]
    current_seq = np.vstack([current_seq[1:], next_features])

forecast = scaler_y.inverse_transform(
    np.array(forecast).reshape(-1, 1)
)

future_dates = [
    df['Date'].iloc[-1] + timedelta(days=i+1)
    for i in range(future_days)
]

# -------------------------------
# Visualization Section
# -------------------------------
st.subheader("📉 Revenue Trend & Forecast")

fig, ax = plt.subplots()
ax.plot(df['Date'], df['Revenue'], label="Historical Revenue")
ax.plot(future_dates, forecast,
        linestyle='--', label="Forecast Revenue")

ax.set_xlabel("Date")
ax.set_ylabel("Revenue")
ax.legend()

st.pyplot(fig)

# -------------------------------
# GenAI-Style Insight Panel
# -------------------------------
trend = "increasing" if forecast.mean() > latest_revenue else "decreasing"

st.markdown("### 🤖 AI-Generated Business Insights")
st.write(f"""
• The overall revenue trend is **{trend}**  
• Predicted average revenue for next {future_days} days is **{forecast.mean():.2f}**  
• Recent growth rate is **{growth:.2f}%**  
• Business performance remains stable with moderate variability  

📌 **Recommendation:**  
Monitor marketing spend and customer acquisition to sustain growth.
""")

# -------------------------------
# Raw Data View
# -------------------------------
with st.expander("📂 View Uploaded Data"):
    st.dataframe(df)
