import streamlit as st
import yfinance as yf
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

st.title("📊 Optimized Currency Exchange Rate Predictor (LSTM + HOA)")

currency_pair = st.text_input("Enter Currency Pair (e.g., USDINR=X, EURUSD=X):", "USDINR=X")
start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2023-01-01"))
forecast_days = st.slider("How many future days to forecast?", 1, 10, 3)

if st.button("Predict"):
    try:
        model_path = "optimized_lstm_currency_model.keras"
        if not os.path.exists(model_path):
            st.error("Model not found. Please train and save the optimized LSTM model.")
        else:
            model = tf.keras.models.load_model(model_path)

            df = yf.download(currency_pair, start=start_date, end=end_date)
            df = df[['Close']].dropna()

            if len(df) < 60:
                st.error("Not enough data to run predictions (need at least 60 data points).")
            else:
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(df.values)

                X = []
                for i in range(60, len(scaled_data)):
                    X.append(scaled_data[i - 60:i])
                X = np.array(X).reshape(-1, 60, 1)

                preds_scaled = model.predict(X)
                preds = scaler.inverse_transform(preds_scaled).flatten()
                actual = df['Close'].values[60:].flatten()
                dates = df.index[60:]

                st.subheader("📈 Actual vs Predicted Prices")
                results_df = pd.DataFrame({
                    "Date": dates,
                    "Actual Price": actual,
                    "Predicted Price": preds
                })
                st.dataframe(results_df.set_index("Date"))
                st.line_chart(results_df.set_index("Date"))

                st.subheader("🔮 Forecasted Prices for Next Days")

                last_window = scaled_data[-60:].reshape(1, 60, 1)
                forecast = []

                for _ in range(forecast_days):
                    next_pred = model.predict(last_window)[0][0]
                    forecast.append(next_pred)
                    last_window = np.append(last_window[:, 1:, :], [[[next_pred]]], axis=1)

                forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
                future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)

                forecast_df = pd.DataFrame({
                    "Date": future_dates,
                    "Forecasted Price": forecast
                })
                st.dataframe(forecast_df.set_index("Date"))

                fig, ax = plt.subplots()
                ax.plot(df.index[-30:], df['Close'].values[-30:], label="Recent Prices")
                ax.plot(forecast_df["Date"], forecast_df["Forecasted Price"], linestyle="--", marker='o', color="orange", label="Forecast")
                ax.set_title(f"{currency_pair} Forecast")
                ax.legend()
                st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred: {e}")