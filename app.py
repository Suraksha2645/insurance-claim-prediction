import streamlit as st
import pandas as pd
import joblib

model = joblib.load("best_model.pkl")

st.title("Insurance Claim Prediction System")

st.write("Enter customer details")

age = st.number_input("Age", min_value=18, max_value=100)

sex_text = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

sex = 1 if sex_text == "Male" else 0

height_cm = st.number_input(
    "Height (cm)",
    min_value=100.0,
    max_value=250.0,
    value=170.0
)

weight_kg = st.number_input(
    "Weight (kg)",
    min_value=20.0,
    max_value=300.0,
    value=70.0
)

height_m = height_cm / 100

bmi = weight_kg / (height_m ** 2)

st.write(f"Calculated BMI: {bmi:.2f}")

children = st.number_input(
    "Number of Children",
    min_value=0,
    max_value=10
)

smoker_text = st.selectbox(
    "Smoker",
    ["No", "Yes"]
)

smoker = 1 if smoker_text == "Yes" else 0

region_text = st.selectbox(
    "Region",
    [
        "Northeast",
        "Northwest",
        "Southeast",
        "Southwest"
    ]
)

region_mapping = {
    "Northeast": 0,
    "Northwest": 1,
    "Southeast": 2,
    "Southwest": 3
}

region = region_mapping[region_text]

charges = st.number_input(
    "Medical Charges",
    min_value=0.0
)

if st.button("Predict"):

    BMI_Age = bmi * age
    Family_Size = children + 1

    data = pd.DataFrame({
        "age":[age],
        "sex":[sex],
        "bmi":[bmi],
        "children":[children],
        "smoker":[smoker],
        "region":[region],
        "charges":[charges],
        "BMI_Age":[BMI_Age],
        "Family_Size":[Family_Size]
    })

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    if prediction == 1:
        st.success("Insurance Claim Predicted")
    else:
        st.error("No Insurance Claim Predicted")

    st.write(
        f"Probability of Claim: {probability*100:.2f}%"
    )