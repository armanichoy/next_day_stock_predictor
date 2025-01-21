import streamlit as st
import yfinance as yf
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Import your custom functions here
from src.functions import load_tables, transform_table, train_test_model

st.title("Stock Prediction App")
st.sidebar.header("User Input")

# User Input
stock_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL)", value="AAPL")
timeframe = st.sidebar.selectbox("Select Timeframe", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "max"])
predict_button = st.sidebar.button("Predict")

if predict_button:
    try:
        # Load data
        stock_df = load_tables(stock_symbol)
        st.write(f"Loaded data for {stock_symbol}.")
        st.write(stock_df.tail())

        # Transform the data
        full_table, last_day, transformed_table = transform_table(stock_df)
        st.write("Transformed Data:")
        st.write(transformed_table.head())

        # Train and test the model
        st.write("Training the model...")
        result, model, importances = train_test_model(transformed_table)

        # Display results
        st.write("Prediction Results:")
        def directions(direction):
            if direction == 0:
                return "No Change"
            elif direction == 1:
                return "Up"
            else:
                return "Down"
        st.text(directions(model.predict(last_day.drop(['target']).to_frame().T)[0]))
        
        # Display feature importance
        importance_df = pd.DataFrame({
            'Feature': ['Close', 'std_23days', 'z_23days', 'mu_23days', 'month', 'dow', 
                        'pct_top_wick', 'pct_body', 'pct_bottom_wick', 'pct_gap_up_down',
                        'z_top_30days', 'z_body_30days', 'z_bottom_30days', 'z_gap_30days'],
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        st.write("Feature Importances:")
        st.write(importance_df)

    except Exception as e:
        st.error(f"An error occurred: {e}")
