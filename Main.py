import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import graphviz
import sys

# Fix UnicodeEncodeError for emojis on Windows
sys.stdout.reconfigure(encoding='utf-8')

# Sklearn Modules
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# Load Dataset
file_path = r"C:\Users\Rajasree\Desktop\Heart Disease Model\framingham.csv"
data = pd.read_csv(file_path)

# Handle Missing Values (Replace with Mean)
data.fillna(data.mean(), inplace=True)

# Define Features and Target
X = data.drop(columns=['TenYearCHD'])
y = data['TenYearCHD']

# Apply SMOTE to Balance the Dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Standardize Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest Model
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate Model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("\U0001F4CA Model Accuracy:", accuracy)
print("\n\U0001F4CA Classification Report:\n", classification_report(y_test, y_pred))

# Save Model & Scaler
joblib.dump(model, 'heart_disease_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# ================== FUNCTION: Real-time Prediction ==================
def predict_heart_disease(features):
    """Predict heart disease based on input features."""
    model = joblib.load('heart_disease_model.pkl')
    scaler = joblib.load('scaler.pkl')
    features = np.array(features).reshape(1, -1)
    features = scaler.transform(features)
    prediction = model.predict(features)
    return "\U0001FA7A At Risk" if prediction[0] == 1 else "✅ Low Risk"

# Interactive Feature: Choose a Person for Prediction
print("\nChoose a person from the test set for prediction (Enter an index between 0 and", len(X_test) - 1, "):")
person_index = int(input("Enter index: "))

# Ensure index is within range
if 0 <= person_index < len(X_test):
    sample_data = X_test.iloc[person_index].values
    print("\n\U0001F4CA Real-time Prediction:", predict_heart_disease(sample_data))

    # ================== VISUALIZATIONS FOR SELECTED PERSON ==================
    person_data = X_test.iloc[person_index]

    # Feature Distribution (Bar Chart for Selected Person)
    plt.figure(figsize=(12, 5))
    sns.barplot(x=person_data.index, y=person_data.values, palette='coolwarm')
    plt.xticks(rotation=45)
    plt.title(f"Feature Values for Selected Person (Index {person_index})")
    plt.show()

    # Feature Relationships (Scatter Plots)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.scatterplot(x=data['age'], y=data['totChol'], hue=data['TenYearCHD'], palette='coolwarm', alpha=0.7)
    plt.scatter(person_data['age'], person_data['totChol'], color='black', marker='x', s=100, label='Selected Person')
    plt.legend()
    plt.title("Age vs Total Cholesterol")

    plt.subplot(1, 2, 2)
    sns.scatterplot(x=data['age'], y=data['sysBP'], hue=data['TenYearCHD'], palette='coolwarm', alpha=0.7)
    plt.scatter(person_data['age'], person_data['sysBP'], color='black', marker='x', s=100, label='Selected Person')
    plt.legend()
    plt.title("Age vs Systolic Blood Pressure")

    plt.tight_layout()
    plt.show()

    # Feature Correlation Heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.show()
else:
    print("\n❌ Invalid index. Please enter a valid number within range.")

