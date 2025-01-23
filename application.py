import os
import gdown
import pandas as pd
import streamlit as st
from src.pipelines.prediction_training import CustomData,PredictPipeline


# Function to download model from Google Drive
def download_model():
    url = "https://drive.google.com/uc?id=11OujRjqt161MqpWwS8Q8mU9oFHHbqLp2"
    output = "model.pkl"
    gdown.download(url, output, quiet=False)


# Streamlit UI
st.title("Diamond Price Prediction")

# Download the model if it's not already downloaded
if not os.path.exists("model.pkl"):
    with st.spinner("Loading Model ..."):
        download_model()

# Initialize model and pipeline
predict_pipeline = PredictPipeline()


# Form for manual data input
st.header("Enter Diamond Information:")

carat = st.number_input("Carat", min_value=0.0, step=0.1)
depth = st.number_input("Depth", min_value=0.0, step=0.1)
table = st.number_input("Table", min_value=0.0, step=0.1)
x = st.number_input("X", min_value=0.0, step=0.1)
y = st.number_input("Y", min_value=0.0, step=0.1)
z = st.number_input("Z", min_value=0.0, step=0.1)

cut = st.selectbox("Cut", options=["Fair", "Good", "Very Good", "Ideal", "Premium"])
color = st.selectbox("Color", options=["D", "E", "F", "G", "H", "I", "J"])
clarity = st.selectbox(
    "Clarity", options=["SI1", "SI2", "VS1", "VS2", "VVS1", "VVS2", "IF", "FL"]
)

# Prediction button
if st.button("Predict"):
    # Create CustomData object with input values
    data = CustomData(
        carat=carat,
        depth=depth,
        table=table,
        x=x,
        y=y,
        z=z,
        cut=cut,
        color=color,
        clarity=clarity,
    )

    # Get the data as DataFrame
    final_new_data = data.get_data_as_dataframe()

    # Make prediction
    pred = predict_pipeline.predict(final_new_data)
    result = round(pred[0], 2)

    # Display result
    st.success(f"Predicted Diamond Price: ${result}")
