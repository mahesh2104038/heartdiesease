import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import numpy as np
 
st.write(f"## Disease Prediction Application")
st.write("Author: Bhawani Shankar")
 
 
# Loaded the model
heat_disease_model = pickle.load(open("model/heart_disease_model1.sav", "rb"))
diabetes_model = pickle.load(open("model\diabetes_model.sav", "rb"))
with st.sidebar:
    selected = option_menu("Choose the Disease", ["Heart Disease Prediction", "Diabetes Prediction"])
 
 
if selected == "Heart Disease Prediction":
    st.write(f"## Heart Disease Prediction")
 
    # Code of Heart disease prediction
    # Heart disease
    st.write("Please Provide your Details")
    # Created the columns
    col1, col2  = st.columns(2)
    # Taking User Inputs
    with col1:
        age = st.text_input("Type your age")
        sex = st.text_input("Type your age for Male:1 and Female: 0")
        cp = st.text_input("Type your cp")
        trestbps = st.text_input("Type your trestbps")
        chol = st.text_input("Type your chol")
        fbs = st.text_input("Type your fbs")
        restecg = st.text_input("Type your restecg")
 
    with col2:
        thalach = st.text_input("Type your thalach")
        exang = st.text_input("Type your exang")
        oldpeak = st.text_input("Type your oldpeak")
        slope = st.text_input("Type your slope")
        ca = st.text_input("Type your ca")
        thal = st.text_input("Type your thal")
 
    if st.button("Predict"):
        data = [age, sex, cp, trestbps, chol, fbs, restecg,thalach, exang, oldpeak, slope, ca, thal]
        data_array = np.array(data, dtype=float).reshape(1,-1)
        prediction = heat_disease_model.predict(data_array)
        st.write(f"## Prediction: {prediction}")
 
 
 
if selected == "Diabetes Prediction":
    st.write(f"## Diabetes Prediction")
    if selected == "Diabetes Prediction":
     st.write(f"## Diabetes Prediction")
     # Take User inputs
     col1, col2, col3= st.columns(3)
   
     with col1:
        Pregnancies = st.text_input('No of Pregnancies')
        SkinThickness = st.text_input('Skin Thickness value')
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
     with col2:
        Glucose = st.text_input('Glucose Level')
        Insulin = st.text_input('Insulin Level')
        Age = st.text_input('Age')
     with col3:
        BloodPressure = st.text_input('Blood Pressure value')
        BMI = st.text_input('BMI')
    if st.button("Predict"):
        data = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age ]
        data_array = np.array(data, dtype=float).reshape(1,-1)
        prediction = diabetes_model.predict(data_array)
        st.write(f"## Prediction: {prediction}")