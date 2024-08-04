import streamlit as st
from datetime import datetime,timedelta
import sklearn
import pandas as pd 
import numpy as np
import pickle

model = pickle.load(open('xgb_model.pkl','rb'))

st.title('Uber Fares Prediction ')
st.sidebar.header('Please select the information of the journey')

#function for input
def user_report():
    passenger_count = st.sidebar.number_input('Passenger Count',min_value=1,max_value=7,step=1)
    year = st.sidebar.number_input('Year of Journey',min_value=2009,max_value=2015,step=1)
    month = st.sidebar.number_input('Month of Journey',min_value=1, max_value=12, step=1)
    day = st.sidebar.number_input('Day of Journey',1,30,1)
    Day_of_Name = st.sidebar.number_input('Day of Name',0,6,0)
    Day_of_Week=st.sidebar.number_input('Day of Week',0,6,0)
    hour = st.sidebar.number_input('Hour of Journey',0,23,1)
    mins = st.sidebar.number_input('Minutes of Journey',0,59,1)
    sec=st.sidebar.number_input('Seconds of Journey',0,59,1)
    distance_travelled = st.sidebar.number_input('Distance  of Journey(In Kilometers)',min_value=1.0,max_value=100.0,step=0.5) 
    car_model=st.sidebar.number_input('Car Model',min_value=0,max_value=2,step=0)
   
    user_report_data={
    'passenger_count':passenger_count,
    'year':year,
    'month':month,
    'day': day,
    'day_of_name':Day_of_Name,
    'Day_of_Week':Day_of_Week,
    'hour': hour,
    'mins': mins,
    'sec':sec,
    'distance_travelled' : distance_travelled,
    'car_model':car_model
    }

    report_data = pd.DataFrame(user_report_data,index=[0])
    return report_data

user_data = user_report()
st.header('Please check the information provided by you')
st.write(user_data)

## fares Prediction
if (st.button("Calculate Fares")):
    fare = model.predict(user_data)
    st.subheader('Fares for the journey would be')
    st.subheader(f"${np.round(fare[0], 2)}")
