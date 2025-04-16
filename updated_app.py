import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Load Dataset
file_path = r"C:\Users\Harshitha Reddy\OneDrive\Documents\RTP_Project\New\framingham.csv"
data = pd.read_csv(file_path)

# Load Model, Scaler & Feature Names
model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')

# Streamlit UI
st.title("Heart Disease Prediction Web App")
st.write("### Please enter the person's details for prediction")

# User input for each feature
inputs = {}
for feature in feature_names:
    if feature == 'age':  # Example for age input
        inputs[feature] = st.number_input(f"Enter {feature}:", min_value=1, max_value=100, value=50)
    elif feature == 'sex':  # Binary feature (gender)
        inputs[feature] = st.selectbox(f"Select {feature} (0 = Female, 1 = Male):", [0, 1])
    elif feature == 'totChol':  # Total Cholesterol input
        inputs[feature] = st.number_input(f"Enter {feature} (Total Cholesterol):", min_value=100, max_value=600, value=200)
    elif feature == 'sysBP':  # Systolic Blood Pressure input
        inputs[feature] = st.number_input(f"Enter {feature} (Systolic Blood Pressure):", min_value=80, max_value=200, value=120)
    elif feature == 'diaBP':  # Diastolic Blood Pressure input
        inputs[feature] = st.number_input(f"Enter {feature} (Diastolic Blood Pressure):", min_value=60, max_value=130, value=80)
    elif feature == 'BMI':  # Body Mass Index input
        inputs[feature] = st.number_input(f"Enter {feature} (Body Mass Index):", min_value=10, max_value=50, value=25)
    elif feature == 'heartRate':  # Heart Rate input
        inputs[feature] = st.number_input(f"Enter {feature} (Heart Rate):", min_value=50, max_value=200, value=70)
    elif feature == 'glucose':  # Glucose input
        inputs[feature] = st.number_input(f"Enter {feature} (Glucose):", min_value=50, max_value=250, value=100)
    else:
        inputs[feature] = st.number_input(f"Enter {feature}:")  # For any other feature

# Convert inputs to a DataFrame
input_data = pd.DataFrame(inputs, index=[0])

# Standardize the input features (same as during training)
input_data_scaled = scaler.transform(input_data)

# Prediction Function
def predict_heart_disease(features):
    """Predict heart disease based on input features."""
    prediction = model.predict(features)
    return "✅ Low Risk" if prediction[0] == 0 else "🧠 At Risk"

# Prediction Output
if st.button("Predict Heart Disease Risk"):
    prediction = predict_heart_disease(input_data_scaled)
    st.write(f"### Prediction: {prediction}")

    # ================== VISUALIZATIONS ==================
    
    # Feature Relationships (Scatter Plots)
    st.write("### Feature Relationships (Scatter Plots)")

    # Example: Age vs Total Cholesterol
    plt.figure(figsize=(12, 5))
    sns.scatterplot(x=data['age'], y=data['totChol'], hue=data['TenYearCHD'], palette='coolwarm', alpha=0.7)
    plt.scatter(inputs['age'], inputs['totChol'], color='black', marker='x', s=100, label='Selected Person')
    plt.legend()
    plt.title("Age vs Total Cholesterol")
    st.pyplot(plt)

    # Age vs Systolic Blood Pressure
    plt.figure(figsize=(12, 5))
    sns.scatterplot(x=data['age'], y=data['sysBP'], hue=data['TenYearCHD'], palette='coolwarm', alpha=0.7)
    plt.scatter(inputs['age'], inputs['sysBP'], color='black', marker='x', s=100, label='Selected Person')
    plt.legend()
    plt.title("Age vs Systolic Blood Pressure")
    st.pyplot(plt)

    # Age vs BMI (Body Mass Index)
    plt.figure(figsize=(12, 5))
    sns.scatterplot(x=data['age'], y=data['BMI'], hue=data['TenYearCHD'], palette='coolwarm', alpha=0.7)
    plt.scatter(inputs['age'], inputs['BMI'], color='black', marker='x', s=100, label='Selected Person')
    plt.legend()
    plt.title("Age vs BMI")
    st.pyplot(plt)

    # ================== HISTOGRAMS ==================
    st.write("### Feature Distributions (Histograms)")

    for feature in feature_names:
        plt.figure(figsize=(12, 5))
        sns.histplot(data[feature], kde=True, bins=20, color='skyblue', edgecolor='black')
        plt.title(f"Distribution of {feature}")
        st.pyplot(plt)

    # ================== CORRELATION HEATMAP ==================
    st.write("### Feature Correlation Heatmap")

    # Calculate the correlation matrix
    correlation_matrix = data.corr()

    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap")
    st.pyplot(plt)
