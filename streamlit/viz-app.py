import streamlit as st
import pandas as pd
import numpy as np
import pickle 
import base64 


@st.cache(suppress_st_warning=True)

# Create a sidebar to allow user select page 
# Exploration - contains 2 plots from task 1; Prediction - contains 1 plot from model predictions
st.sidebar.title('Select the page to display visualisation')
app_mode = st.sidebar.selectbox('Select Page',['Exploration','Prediction'])

if app_mode is 'Exploration':
    st.title('Exploration')  
    st.subheader('Dataset')
    st.caption('The dataset contains responses from the 2020 Stack Overflow Developer survey, which is among the largest and most comprehensive survey of software developers. Below is a complete list of attributes that correspond to the questions included in the survey.')
	# load the dataset 
    data=pd.read_csv('cleaned_survey.csv')
    st.write(data.head())
	# Chart 1
    st.subheader('Column pair #1: ...')
    st.write('Chart goes here')

	# Chart 2 
    st.subheader('Column pair #2: ...')
    st.write('Chart goes here')
elif app_mode is 'Prediction': 
    st.title('Multi-label classification')
    st.write('This application predicts income range based on user-input features')
    # load the data 
    data = get_user_input()
    st.subheader('Input features')
    st.write(data.head())

    # Clean & encode input data

    # Load model

    # Make prediction on cleaned data  

    # Display the prediction 

    # Prediction 
    st.subheader('Prediction')
    st.write('Prediction goes here')

    # Chart 
    st.subheader('Chart')
    st.write('Chart goes here')







