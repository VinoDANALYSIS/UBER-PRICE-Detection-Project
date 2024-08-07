import streamlit as st
from datetime import datetime,timedelta
import sklearn
import pandas as pd 
import numpy as np
import pickle

model = pickle.load(open('xgb_model.pkl','rb'))

st.title('Uber Fares Prediction ')
st.sidebar.header('Please select the information of the journey')

# Function for input
def user_report():
    passenger_count = st.sidebar.number_input('Passenger Count', min_value=1, max_value=7, step=1)
    year = st.sidebar.number_input('Year of Journey', min_value=2009, max_value=2015, step=1)
    
    months = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 7: 'July', 8: 'August', 
              9: 'September', 10: 'October', 11: 'November', 12: 'December'}
    month = st.sidebar.selectbox('Month of Journey', options=list(months.keys()), format_func=lambda x: months[x])
    
    day = st.sidebar.number_input('Day of Journey', 1, 30, 1)
    
    days = {0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday'}
    day_of_name = st.sidebar.selectbox('Day of Name', options=list(days.keys()), format_func=lambda x: days[x])
    
    weeks = {0: 'First Week', 1: 'Second Week', 2: 'Third Week', 3: 'Fourth Week'}
    day_of_week = st.sidebar.selectbox('Day of Week', options=list(weeks.keys()), format_func=lambda x: weeks[x])
    
    hour = st.sidebar.number_input('Hour of Journey', 0, 23, 1)
    mins = st.sidebar.number_input('Minutes of Journey', 0, 59, 1)
    sec = st.sidebar.number_input('Seconds of Journey', 0, 59, 1)
    
    distance_travelled = st.sidebar.number_input('Distance of Journey (in kilometers)', min_value=1.0, max_value=100.0, step=0.5)
    
    car_models = {0: 'Mini', 1: 'XL', 2: 'XUV'}
    car_model = st.sidebar.selectbox('Car Model', options=list(car_models.keys()), format_func=lambda x: car_models[x])
    
    user_report_data = {
        'passenger_count': passenger_count,
        'year': year,
        'month': month,
        'day': day,
        'day_of_name': day_of_name,
        'day_of_week': day_of_week,
        'hour': hour,
        'mins': mins,
        'sec': sec,
        'distance_travelled': distance_travelled,
        'car_model': car_model
    }
    
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data

user_data = user_report()
st.header('Please check the information provided by you')
st.write(user_data)

# Fares Prediction
if st.button("Calculate Fares"):
    fare = model.predict(user_data)
    st.subheader('Fares for the journey would be')
    st.subheader(f"${np.round(fare[0], 2)}")
