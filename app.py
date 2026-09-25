import streamlit as st
import requests

st.title("Medical ICD-10 Coding Assistant")

note = st.text_area("Enter clinical note:")

if st.button("Predict"):
    response = requests.post("http://127.0.0.1:8000/predict", json={"note": note})
    result = response.json()

    st.write("Predicted ICD Code:", result["predicted_code"])
    st.write("Confidence:", result["confidence"])