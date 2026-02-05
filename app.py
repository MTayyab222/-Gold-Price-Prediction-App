import streamlit as st
import pickle
import pandas as pd
import numpy as np


# Page config
st.set_page_config(page_title="Gold Price Predictor", page_icon="💰")

# Title
st.title("💰 Gold Price Prediction App")
st.write("Predict gold prices using Machine Learning!")

# Load model
@st.cache_resource
def load_model():
    with open('gold_price_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Sidebar for inputs
st.sidebar.header("Enter Market Data:")

# Input fields (adjust according to your features)
spx = st.sidebar.number_input("SPX (S&P 500)", value=4500.0, step=10.0)
uso = st.sidebar.number_input("USO (Oil Price)", value=75.0, step=1.0)
slv = st.sidebar.number_input("SLV (Silver Price)", value=24.0, step=0.5)
eur_usd = st.sidebar.number_input("EUR/USD", value=1.10, step=0.01)

# Prediction button
if st.sidebar.button("🔮 Predict Gold Price"):
    # Create input dataframe
    input_data = pd.DataFrame({
        'SPX': [spx],
        'USO': [uso],
        'SLV': [slv],
        'EUR/USD': [eur_usd]
    })
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Display result
    st.success(f"### Predicted Gold Price: ${prediction[0]:.2f}")
    
    # Additional info
    st.info(f"""
    **Input Parameters:**
    - S&P 500 (SPX): {spx}
    - Oil Price (USO): {uso}
    - Silver Price (SLV): {slv}
    - EUR/USD: {eur_usd}
    """)

# Footer
st.markdown("---")
st.write("Built with ❤️ using Random Forest Regressor | R² Score: 0.989")