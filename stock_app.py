import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
from keras.models import load_model
import yfinance as yf
from datetime import datetime

# Load the models and scaler
def load_models_and_scaler():
    short_term_model = load_model("short_term_lstm_model.h5")
    long_term_model = load_model("long_term_lstm_model.h5")
    scaler = joblib.load("scaler.save")  # Load the scaler
    return short_term_model, long_term_model, scaler

# Function to fetch data from Yahoo Finance for a given stock and date
def fetch_stock_data(stock_symbol, start_date, end_date):
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    return data

# Function to make predictions based on user-selected date
def make_prediction(date_input):
    # Load models and scaler
    short_term_model, long_term_model, scaler = load_models_and_scaler()

    # Fetch historical data for Coca-Cola (KO) from Yahoo Finance up to the selected date
    historical_data = fetch_stock_data('KO', '2010-01-01', date_input.strftime('%Y-%m-%d'))
    
    # Prepare the lag features and target (here it's assumed the 'Close' column is used)
    historical_data = historical_data[['Close']]  # Use only the 'Close' column for predictions
    lag_features = historical_data.tail(12).values.reshape(-1, 1)  # Last 12 values

    # Rescale the features using the scaler
    lag_features_scaled = scaler.transform(lag_features)

    # Prepare the input data for LSTM models
    lag_features_scaled = lag_features_scaled.reshape(1, 12, 1)  # Adjust shape for LSTM

    # Make predictions using both short-term and long-term models
    short_term_pred = short_term_model.predict(lag_features_scaled)[0][0]
    long_term_pred = long_term_model.predict(lag_features_scaled)[0][0]

    # Inverse transform to get actual values
    short_term_pred = scaler.inverse_transform([[short_term_pred]])[0][0]
    long_term_pred = scaler.inverse_transform([[long_term_pred]])[0][0]

    return short_term_pred, long_term_pred

# Streamlit UI setup
st.title("🥤 Coca-Cola (KO) Stock Price Prediction")
st.markdown("### Select a Date to Get Predictions")

# Date picker for user to select a date
date_input = st.date_input("Select Date", min_value=datetime(2010, 1, 1), max_value=datetime.today())

# Display predictions when the date is selected
if date_input:
    short_term_pred, long_term_pred = make_prediction(date_input)
    
    st.markdown(f"### Predictions for {date_input.strftime('%Y-%m-%d')}")
    st.markdown(f"📊 **Short-Term Predicted Close Price**: ${short_term_pred:.2f}")
    st.markdown(f"📈 **Long-Term Predicted Close Price**: ${long_term_pred:.2f}")
