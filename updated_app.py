import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load data
data = pd.read_csv("framingham.csv")

# Load model and pre-processing files
with open("heart_disease_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

with open("feature_names.pkl", "rb") as features_file:
    feature_names = pickle.load(features_file)

# Title
st.title("Heart Disease Prediction App")

# User input fields
def user_input():
    st.sidebar.header("Enter Patient Data")

    input_data = {}
    for feature in feature_names:
        input_data[feature] = st.sidebar.number_input(f"{feature}", value=0.0)

    df = pd.DataFrame(input_data, index=[0])
    return df

input_df = user_input()

# Show input
st.subheader("Input Data")
st.write(input_df)

# Prediction
if st.button("Predict"):
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)
    prediction_proba = model.predict_proba(scaled_input)

    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

    st.subheader("Prediction Probability")
    st.write({
        "Low Risk Probability": f"{prediction_proba[0][0]*100:.2f}%",
        "High Risk Probability": f"{prediction_proba[0][1]*100:.2f}%"
    })
