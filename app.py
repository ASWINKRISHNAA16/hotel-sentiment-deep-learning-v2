import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from PIL import Image

# Load artifacts
model = load_model("hotel_sentiment_dl_model.h5")
with open("scaler_dl.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("columns_dl.pkl", "rb") as f:
    model_columns = pickle.load(f)

# Load and display image
image = Image.open("pic.jpeg")

st.image(image, use_column_width=True)
st.title("🏨 Hotel Review Sentiment Prediction")

st.markdown("""
Please enter the required feature values below to predict whether the hotel review sentiment is **Positive (Satisfied)** or **Negative (Not Satisfied)**.
""")

# Generate input widgets dynamically based on your columns
user_input = {}
for col in model_columns:
    # Since features are numeric and scaled in [0,1], accept input in that range or original range if known
    # Let's assume some reasonable default ranges - you can adjust accordingly
    user_input[col] = st.number_input(f"{col}", value=0.5, min_value=0.0, max_value=10.0, step=0.1)

# Convert input to DataFrame
input_df = pd.DataFrame([user_input])

# Scale the input - since your model expects scaled input
input_scaled = scaler.transform(input_df)

if st.button("Predict Sentiment"):
    pred_prob = model.predict(input_scaled)[0][0]
    pred_class = int(pred_prob > 0.5)

    if pred_class == 1:
        st.success(f"Predicted Sentiment: Positive (Satisfied) with probability {pred_prob:.2f}")
    else:
        st.error(f"Predicted Sentiment: Negative (Not Satisfied) with probability {1 - pred_prob:.2f}")
