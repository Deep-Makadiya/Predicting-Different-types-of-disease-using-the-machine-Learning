import os
import pickle
import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# loading the saved models
diabetes_model = pickle.load(open(f'{working_dir}/saved_models/diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open(f'{working_dir}/saved_models/heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open(f'{working_dir}/saved_models/parkinsons_model.sav', 'rb'))

# sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person'],
                           default_index=0)

# Function to load test data from CSV
def load_test_data(file):
    return pd.read_csv(file)

# Function to make predictions and display results
def make_predictions(model, test_data):
    predictions = model.predict(test_data)
    return predictions

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    # page title
    st.title('Diabetes Prediction using ML')

    # Upload CSV file containing test data
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        test_data = load_test_data(uploaded_file)
        st.write(test_data)

        # code for Prediction
        diab_diagnosis = pd.DataFrame(index=test_data.index)

        # creating a button for Prediction
        if st.button('Diabetes Test Result'):
            diab_predictions = make_predictions(diabetes_model, test_data)
            diab_diagnosis['Diabetes Diagnosis'] = ['The person is diabetic' if pred == 1 else 'The person is not diabetic' for pred in diab_predictions]

        st.write(diab_diagnosis)

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    # page title
    st.title('Heart Disease Prediction using ML')

    # Upload CSV file containing test data
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        test_data = load_test_data(uploaded_file)
        st.write(test_data)

        # code for Prediction
        heart_diagnosis = pd.DataFrame(index=test_data.index)

        # creating a button for Prediction
        if st.button('Heart Disease Test Result'):
            heart_predictions = make_predictions(heart_disease_model, test_data)
            heart_diagnosis['Heart Disease Diagnosis'] = ['The person is having heart disease' if pred == 1 else 'The person does not have any heart disease' for pred in heart_predictions]

        st.write(heart_diagnosis)

# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":
    # page title
    st.title("Parkinson's Disease Prediction using ML")

    # Upload CSV file containing test data
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        test_data = load_test_data(uploaded_file)
        st.write(test_data)

        # code for Prediction
        parkinsons_diagnosis = pd.DataFrame(index=test_data.index)

        # creating a button for Prediction
        if st.button("Parkinson's Test Result"):
            parkinsons_predictions = make_predictions(parkinsons_model, test_data)
            parkinsons_diagnosis["Parkinson's Disease Diagnosis"] = ["The person has Parkinson's disease" if pred == 1 else "The person does not have Parkinson's disease" for pred in parkinsons_predictions]

        st.write(parkinsons_diagnosis)
