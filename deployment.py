import streamlit as st
import requests
import pandas as pd
from streamlit_option_menu import option_menu
import streamlit_lottie as st_lottie
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
model=joblib.load(open("HealthRisk",'rb'))

# Function to load animations
def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Prediction function
def predict(features):
    prediction = model.predict(features)
    return prediction

# Page configuration
st.set_page_config(
    page_title='Health Risk Prediction',
    page_icon=':hospital:',
    initial_sidebar_state='collapsed'
)

# Sidebar menu
with st.sidebar:
    choose = option_menu(
        None,
        ["Home", "Graphs", "About", "Contact"],
        icons=["house", "bar-chart", "book", "envelope"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#f0f2f6"},
            "icon": {"color": "#6c757d", "font-size": "25px"},
            "nav-link": {"font-size": "16px", "text-align": "left"},
            "nav-link-selected": {"background-color": "#02ab21"}
        }
    )

# Home page
if choose == "Home":
    st.write('# Health Risk Prediction')
    st.subheader('Enter your details for health risk prediction')
    
    # User inputs
    age = st.number_input("Enter your age:", min_value=0)
    blood_pressure = st.text_input("Enter your blood pressure (Systolic/Diastolic):")
    glucose = st.number_input("Enter your glucose level:", min_value=0)
    insulin = st.number_input("Enter your insulin level:", min_value=0)
    bmi = st.number_input("Enter your BMI:", min_value=0.0, format="%.1f")
    diabetes_pedigree = st.number_input("Enter your diabetes pedigree function:", min_value=0.0, format="%.2f")
    pregnancies = st.number_input("Enter number of pregnancies:", min_value=0)
    skin_thickness = st.number_input("Enter your skin thickness:", min_value=0)
    
    if blood_pressure:
        try:
            systolic_bp, diastolic_bp = map(float, blood_pressure.split('/'))
        except ValueError:
            st.error("Please enter blood pressure in the format 'Systolic/Diastolic'.")
            systolic_bp, diastolic_bp = 0, 0
    else:
        systolic_bp, diastolic_bp = 0, 0

    additional_features = [0] * 22  # Replace this with the actual preprocessing steps to generate these features

    features = np.array([age, glucose, insulin, bmi, diabetes_pedigree, pregnancies, skin_thickness, systolic_bp, diastolic_bp] + additional_features).reshape(1, -1)
   
    
    # Prediction
    if st.button("Predict"):
        result = predict(features)
        st.write(f'The predicted health risk is: **{result[0]}**')


# About page
elif choose == "About":
    st.write("# About")
    st.write("This app provides health risk predictions based on user input.")

# Contact page
elif choose == "Contact":
    st.write("# Contact")
    st.write("For inquiries, please contact us at contact@example.com")

# Load the datasets (for displaying purposes only)
diabetes_df = pd.read_csv("Healthcare-Diabetes.csv")
heart_attack_df = pd.read_csv("heart_attack_prediction_dataset.csv")

