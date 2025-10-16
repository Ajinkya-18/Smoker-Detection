import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import load_model, preprocess_data


st.set_page_config(page_title="Smoker Classification", layout="wide")
st.title("🚬 Smoker Detection App")
st.write("This app predicts whether a person is a smoker or not based on their health data.")


@st.cache_resource
def load_models():
    model = load_model('models/balanced_RFC.joblib')

    return model


st.sidebar.header("Input Features")

input_features = {}

col1, col2, col3 = st.sidebar.columns(3)

input_features['age'] = col1.number_input('Age', min_value=1, max_value=100, value=30)
input_features['height(cm)'] = col1.number_input('Height (cm)', min_value=50, max_value=250, value=170)
input_features['weight(kg)'] = col1.number_input('Weight (kg)', min_value=20, max_value=200, value=70) 
input_features['waist(cm)'] = col1.number_input('Waist (cm)', min_value=30.0, max_value=200.0, value=80.0)
input_features['systolic'] = col1.number_input('Systolic Blood Pressure', min_value=40, max_value=250, value=120)
input_features['relaxation'] = col1.number_input('Diastolic Blood Pressure', min_value=20, max_value=200, value=80)
input_features['fasting blood sugar'] = col1.number_input('Fasting Blood Sugar (mg/dL)', min_value=50, max_value=400, value=90)

input_features['Cholesterol'] = col2.number_input('Cholesterol', min_value=50, max_value=400, value=180)
input_features['triglyceride'] = col2.number_input('Triglyceride', min_value=30, max_value=500, value=150)
input_features['HDL'] = col2.number_input('HDL Cholesterol', min_value=20, max_value=200, value=50)
input_features['LDL'] = col2.number_input('LDL Cholesterol', min_value=20, max_value=300, value=100)
input_features['hemoglobin'] = col2.number_input('Hemoglobin (g/dL)', min_value=5.0, max_value=20.0, value=12.0)
input_features['serum creatinine'] = col2.number_input('Serum Creatinine', min_value=0.1, max_value=15.0, value=1.0)
input_features['AST'] = col2.number_input('AST', min_value=5, max_value=300, value=30)

input_features['ALT'] = col3.number_input('ALT', min_value=5, max_value=300, value=30)
input_features['Gtp'] = col3.number_input('Gtp', min_value=5, max_value=300, value=30)
input_features['Urine protein'] = col3.selectbox('Urine Protein', [1, 2, 3, 4, 5, 6], index=0)
input_features['eyesight(left)'] = col3.number_input('Eyesight (left)', min_value=0.0, max_value=10.0, value=1.0)
input_features['eyesight(right)'] = col3.number_input('Eyesight (right)', min_value=0.0, max_value=10.0, value=1.0)
input_features['hearing(left)'] = col3.selectbox('Hearing (left)', [1, 2], index=0)
input_features['hearing(right)'] = col3.selectbox('Hearing (right)', [1, 2], index=0)
input_features['dental caries'] = col3.selectbox('Dental Caries', [0, 1], index=0)


if st.sidebar.button('Predict'):
    input_df = pd.DataFrame(input_features, index=[0])

    processed_df = preprocess_data(input_df.copy(), mode="inference")

    model = load_models()

    prediction = model.predict(processed_df)
    prediction_proba = model.predict_proba(processed_df)


    st.subheader("Prediction")

    if prediction[0] == 1:
        st.warning("The model predicts: **Smoker**")

    else:
        st.success("The model predicts: **Non-Smoker**")


    st.subheader("Prediction Probablility")
    st.write(f"**Probability of being a smoker:** {prediction_proba[0][1]:.2f}")
    st.write(f"**Probability of being a non-smoker:** {prediction_proba[0][0]:.2f}")





