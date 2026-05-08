import streamlit as st
import pandas as pd
import joblib


# Load the trained model
@st.cache_resource  # Caches the model so it doesn't reload on every button click
def load_model():
    return joblib.load('fraud_model_rf.pkl')


model = load_model()

# Build the UI
st.title("Credit Card Fraud Detection System")
st.write("Upload a CSV of transaction data to scan for potential fraud.")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the uploaded data
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(data.head())

    if st.button("Run Fraud Detection"):
        # Ensure we drop the 'Class' column if the test data happens to include it
        if 'Class' in data.columns:
            features = data.drop('Class', axis=1)
        else:
            features = data

        # Make predictions
        predictions = model.predict(features)
        probabilities = model.predict_proba(features)[:, 1]

        # Append results to the dataframe
        results = data.copy()
        results['Fraud_Prediction'] = predictions
        results['Fraud_Probability (%)'] = (probabilities * 100).round(2)

        st.write("### Detection Results")
        # Display transactions flagged as fraud (Prediction == 1)
        fraudulent_txns = results[results['Fraud_Prediction'] == 1]

        if len(fraudulent_txns) > 0:
            st.error(f"⚠️ Alert! Detected {len(fraudulent_txns)} fraudulent transactions.")
            st.dataframe(fraudulent_txns)
        else:
            st.success("✅ No fraudulent transactions detected in this batch.")