

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- LOAD SAVED MODEL & PREPROCESSING ---
model = joblib.load('credit_score_model.pkl')   # Trained classifier (e.g. LogisticRegression/RandomForest)
scaler = joblib.load('scaler.pkl')               # Same scaler used in training (StandardScaler) - normalizes inputs
columns = joblib.load('column.pkl')
try:
    columns = list(columns)
except TypeError:
    columns = list(columns) if hasattr(columns, '__iter__') else []
# Match scaler fit-time names (spaces, not underscores)
columns = [
    c.replace('Number_of_Children', 'Number of Children')
     .replace('Marital_Status', 'Marital Status')
     .replace('Home_Ownership', 'Home Ownership')
    for c in columns
]


# --- UI: TITLE & DESCRIPTION ---
st.title("💳 Credit Score Predictor")
st.write("Predict credit score using ML")

# --- UI: INPUT FIELDS (user enters patient values) ---
# Each value is sent to the model. Higher risk generally: older age, high CPK, low EF, low platelets, high creatinine, low sodium, short follow-up.
Age = st.slider("Age", 18, 70, 25)
Gender = st.selectbox("Gender", ["Male", "Female"])
if Gender == "Male":
    Gender = 1
else:
    Gender = 0
Income = st.number_input("Income", 10000, 200000, 50000)
Education = st.selectbox(
    "Education",
    ["High School", "Bachelor", "Master", "PhD"]
)

Marital_Status = st.selectbox(
    "Marital_Status",
    ["Single", "Married"]
)

Number_of_Children = st.slider("Number_of _Children", 0, 5, 0)


Home_Ownership = st.selectbox(
    "Home_Ownership",
    ["Rent", "Own"]
)

# Convert input to dataframe
input_dict = {
    "Age": Age,
    "Income": Income,
    "Number of Children": Number_of_Children,
}

input_df = pd.DataFrame([input_dict])


categorical = {
    f"Education_{Education}": 1,
    f"Marital Status_{Marital_Status}": 1,
    f"Home Ownership_{Home_Ownership}": 1,
}

for col in columns:
    if col not in input_df.columns:
        input_df[col] = categorical.get(col, 0)

input_df = input_df[columns]

# Scale
input_scaled = scaler.transform(input_df)

# Prediction (0 = Average, 1 = High, 2 = Low)
SCORE_LABELS = {0: "Average", 1: "High", 2: "Low"}
if st.button("Predict Credit Score"):
    prediction = model.predict(input_scaled)[0]
    label = SCORE_LABELS.get(int(prediction), str(prediction))
    st.success(f"Predicted Credit Score: **{label}**")