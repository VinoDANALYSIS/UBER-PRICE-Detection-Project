#Uploading and Downloading the uber file in AWS s3 service
!pip install boto3 -q
import boto3
access_key = "access_key"
secret_key = "secret_key"
s3_client = boto3.client('s3',aws_access_key_id=access_key,aws_secret_access_key=secret_key)
buckets=s3_client.list_buckets()
s3_file= s3_client.list_objects_v2(Bucket='myawsuberbucket')
File=('C:/Users/hp/Desktop/uber.csv')

#Upload
s3_client.upload_file(File,'myawsuberbucket','uber.csv')

#Download
s3_client.download_file('myawsuberbucket','uber.csv','s3file.csv')

import pandas as pd
df=pd.read_csv("C:/Users/hp/s3file.csv")

df.info()
df_uber=df.copy()

#convert the pickup_datetime to datetime format
df_uber['pickup_datetime']=pd.to_datetime(df_uber['pickup_datetime'])

#Feature Extraction
df_uber['Year'] = df_uber['pickup_datetime'].dt.year
df_uber['Month']= df_uber['pickup_datetime'].dt.month
df_uber['Day']= df_uber['pickup_datetime'].dt.day
df_uber['Day_of_Name']=df_uber['pickup_datetime'].dt.day_name()
df_uber['Day_of_Week']= df_uber['pickup_datetime'].dt.dayofweek
df_uber['Hour'] = df_uber['pickup_datetime'].dt.hour
df_uber['Minute'] = df_uber['pickup_datetime'].dt.minute
df_uber['Second'] = df_uber['pickup_datetime'].dt.second

import numpy as np
# Drop rows with NaN values
df.dropna(subset=['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'], inplace=True)

# Filter invalid passenger counts
df = df[(df['passenger_count'] > 0) & (df['passenger_count'] <= 7)]

# Fill zero passenger counts with the median value
median_passenger_count = df['passenger_count'].median()
df['passenger_count'] = df['passenger_count'].replace(0, median_passenger_count)

# Drop unnecessary columns
df.drop(columns=['Unnamed: 0', 'key', 'pickup_datetime'], inplace=True)

# Drop latitude and longitude outliers
df = df[(df['pickup_latitude'].between(-90, 90)) & (df['dropoff_latitude'].between(-90, 90))]
df = df[(df['pickup_longitude'].between(-180, 180)) & (df['dropoff_longitude'].between(-180, 180))]

import geopy
from geopy.distance import geodesic

def calculate_distance(row):
    start = (row['pickup_latitude'], row['pickup_longitude'])
    end = (row['dropoff_latitude'], row['dropoff_longitude'])
    return geodesic(start, end).kilometers

df['Distance_km'] = df.apply(calculate_distance, axis=1)

#Removing Outliers
from scipy.stats import zscore

# Calculate Z-scores
df['fare_amount_zscore'] = zscore(df['fare_amount'])
df['distance_km_zscore'] = zscore(df['Distance_km'])

# Define threshold for outliers
threshold = 3

# Identify outliers
outliers_fare = df[(df['fare_amount_zscore']> threshold)]
outliers_distance = df[(df['distance_km_zscore']> threshold)]

print("Number of fare amount outliers:", len(outliers_fare))
print("Number of distance outliers:", len(outliers_distance))

#Cleaned Data--After removing the Outliers from Data[Fare Amount,Distance]
data = df[(df['fare_amount_zscore'] <= threshold) & (df['distance_km_zscore'] <= threshold)]

# Drop Z-score columns
data.drop(columns=['fare_amount_zscore', 'distance_km_zscore'], inplace=True)

#Car Model
def car_model(value):
    if value > 0 and value < 3:
        return "MINI"
    elif value > 2 and value < 5:
        return "XL"
    elif value >= 5:
        return "XUV"

data['car_model'] = data['passenger_count'].map(car_model)

#Removed the Fare Amount <= 0
data.drop(data[data['fare_amount']<=0].index,inplace=True)

!pip install mysql-connector-python
#Connectind AWS to MYSQL

import mysql.connector as db

my_db=db.connect(host="database.craqaua4exat.ap-south-1.rds.amazonaws.com",
                    user="admin",
                    password="Shastik2901",
                    port=3306,database='UBER')

writer=my_db.cursor()

#RDS
from sqlalchemy import create_engine
from sqlalchemy import text
connection="mysql+mysqlconnector://admin:Shastik2901@database.craqaua4exat.ap-south-1.rds.amazonaws.com:3306/UBER"
engine=create_engine(connection,echo=True)
#Pushing the Data to SQL
data.to_sql('UBER_Data',con=engine,if_exists='append',index=False)
#Reading the Data
DF=pd.read_sql_table('UBER_Data',con=engine)
DF.head()

#Converting the non-numerical values into Numerical Values for Model training
DF=data.copy()
import sklearn
from sklearn.preprocessing import LabelEncoder
encoding=LabelEncoder()
DF['car_model']=encoding.fit_transform(DF['car_model'])
DF['Day_of_Name']=encoding.fit_transform(DF['Day_of_Name'])

#Analysis
#Analysis
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Fare amount distribution before and after cleaning
plt.figure(figsize=(10, 6))
sns.histplot(DF['fare_amount'], kde=True)
plt.title('Distribution of Fare Amount')
plt.xlabel('Fare Amount ($)')
plt.ylabel('Frequency')
plt.show()

# Trips by car model
plt.figure(figsize=(10, 6))
sns.countplot(x='car_model', data=DF)
plt.title('Number of Trips by Car Model')
plt.xlabel('Car Model')
plt.ylabel('Number of Trips')
plt.show()

# Correlation matrix
correlation_matrix = DF.corr()
plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Trips by Day of Week
plt.figure(figsize=(10, 6))
sns.countplot(x=DF['Day_of_Name'])
plt.title('Number of Trips by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Number of Trips')
plt.show()

# Trips by Hour of Day
plt.figure(figsize=(10, 6))
sns.countplot(x=DF['Hour'])
plt.title('Number of Trips by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Trips')
plt.show()

#Model Selection
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Prepare data for training
X = DF[['passenger_count', 'Year', 'Month', 'Day', 'Day_of_Name', 'Day_of_Week', 'Hour', 'Minute', 'Second', 'Distance_km', 'car_model']]
y = DF['fare_amount']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
lr_mae = mean_absolute_error(y_test, y_pred_lr)
lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
lr_r2 = r2_score(y_test, y_pred_lr)
print(f'Linear Regression - MAE: {lr_mae}, RMSE: {lr_rmse}, R2: {lr_r2}')

# Train and evaluate Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
rf_r2 = r2_score(y_test, y_pred_rf)
print(f'Random Forest - MAE: {rf_mae}, RMSE: {rf_rmse}, R2: {rf_r2}')

# Train and evaluate XGBoost Regressor
xgb_model = XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
xgb_mae = mean_absolute_error(y_test, y_pred_xgb)
xgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
xgb_r2 = r2_score(y_test, y_pred_xgb)
print(f'XGBoost - MAE: {xgb_mae}, RMSE: {xgb_rmse}, R2: {xgb_r2}')

#Output
Linear Regression - MAE: 2.426881515437828, RMSE: 4.204793996331803, R2: 0.5604169216266717
Random Forest - MAE: 1.832262955844019, RMSE: 2.832326186565462, R2: 0.8005481566278051
XGBoost - MAE: 1.7465241825918647, RMSE: 2.7766591667313194, R2: 0.8083112318930146

#WHY XGB MODEL IS BEST
Based on the evaluation metrics, we select the best model (e.g., XGBoost) for deployment.

#PICKLE

# Save the XGBoost model using pickle
import pickle
pickle.dump(xgb_model, open('xgb_model.pkl', 'wb'))

14.Deploying the Model with Streamlit

Create a Streamlit app to allow users to input journey details and get fare predictions.
