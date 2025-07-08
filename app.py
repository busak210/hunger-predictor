import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open("xgb_model.pkl", "rb"))

st.title("Hunger & Undernourishment Predictor")

st.write("Upload a CSV file or enter values manually:")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    input_data = pd.read_csv(uploaded_file)
else:
    st.subheader("Manual Input")
    val1 = st.number_input("Feature 1", value=0.0)
    val2 = st.number_input("Feature 2", value=0.0)
    val3 = st.number_input("Feature 3", value=0.0)
    val4 = st.number_input("Feature 4", value=0.0)
    input_data = pd.DataFrame([[val1, val2, val3, val4]], columns=["Feature1", "Feature2", "Feature3", "Feature4"])

if st.button("Predict"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Value: {prediction[0]:.2f}")