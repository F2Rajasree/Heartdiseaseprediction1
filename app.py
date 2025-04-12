import streamlit as st
import numpy as np
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Setup page layout and theme
st.set_page_config(
    page_title="Heart Disease Prediction",
    layout="wide",
    page_icon="🫀"
)

# Load model and scaler
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")
data = pd.read_csv("framingham.csv")
data.fillna(data.mean(), inplace=True)

st.title("🫀 Heart Disease Risk Predictor")
st.markdown("### Predict your 10-year risk of coronary heart disease (CHD)")

st.markdown("---")

# Input section
feature_names = [
    "Age", "Gender", "Education", "Current Smoker", "Cigs Per Day", "BP Meds",
    "Prevalent Stroke", "Prevalent Hypertension", "Diabetes", "Total Cholesterol",
    "Systolic BP", "Diastolic BP", "BMI", "Heart Rate", "Glucose"
]

cols = st.columns(3)
user_inputs = []

for i, feature in enumerate(feature_names):
    with cols[i % 3]:
        val = st.number_input(f"{feature}", value=0.0)
        user_inputs.append(val)

st.markdown("---")

# Predict button
if st.button("🔍 Predict"):
    features_array = np.array(user_inputs).reshape(1, -1)
    features_scaled = scaler.transform(features_array)
    result = model.predict(features_scaled)[0]

    st.subheader("🧠 Prediction Result:")
    st.success("🔴 At Risk" if result == 1 else "🟢 Low Risk")

    # Bar Chart
    st.markdown("### 📊 Input Feature Values")
    fig_bar, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x=feature_names, y=user_inputs, palette="coolwarm", ax=ax)
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig_bar)

    # Correlation Heatmap
    st.markdown("### 🔥 Feature Correlation Heatmap")
    fig_heat, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig_heat)