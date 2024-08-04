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
#Latitude and Londigute should not be zero
df_uber['pickup_longitude']=df_uber['pickup_longitude'].replace(0,np.nan)
df_uber['pickup_latitude']=df_uber['pickup_latitude'].replace(0,np.nan)
df_uber['dropoff_longitude']=df_uber['dropoff_longitude'].replace(0,np.nan)
df_uber['dropoff_latitude']=df_uber['dropoff_latitude'].replace(0,np.nan)

#Dropping the Null value from Pickup Latitude,Longitude and Dropoff Latitude,Longitude
df_uber.dropna(subset=['pickup_longitude'],inplace=True)
df_uber.dropna(subset=['pickup_latitude'],inplace=True)
df_uber.dropna(subset=['dropoff_longitude'],inplace=True)
df_uber.dropna(subset=['dropoff_latitude'],inplace=True)

#Minimum and Maximum Values of Passenger Count

df_uber['passenger_count'].min(),df_uber['passenger_count'].max()

#Removing the Passenger count which is not valid
df_uber.drop(df_uber[df_uber['passenger_count']>7].index,inplace=True)

Mean=df_uber['passenger_count'].mean()
Median=df_uber['passenger_count'].median()
Mode=df_uber['passenger_count'].mode()
print(Mean,Median,Mode)

#Filling the Passenger Count Zero Values with Median Value
df_uber['passenger_count']=df_uber['passenger_count'].replace(0,Median)

#Finding the Null Value
df_uber.isnull().sum()

#Removing the latitude and lonngitude value which is greter than 90 and less than -90 as it is not valid
df_uber.drop(df_uber[df_uber['pickup_latitude']>90].index,inplace=True)
df_uber.drop(df_uber[df_uber['pickup_latitude']<-90].index,inplace=True)
df_uber.drop(df_uber[df_uber['dropoff_latitude']>90].index,inplace=True)
df_uber.drop(df_uber[df_uber['dropoff_latitude']<-90].index,inplace=True)
df_uber.drop(df_uber[df_uber['pickup_longitude']>90].index,inplace=True)
df_uber.drop(df_uber[df_uber['pickup_longitude']<-90].index,inplace=True)
df_uber.drop(df_uber[df_uber['dropoff_longitude']>90].index,inplace=True)
df_uber.drop(df_uber[df_uber['dropoff_longitude']<-90].index,inplace=True)

# Calculate distance using geodesic distance
from geopy.distance import geodesic

def calculate_distance(row):
    start = (row['pickup_latitude'], row['pickup_longitude'])
    end = (row['dropoff_latitude'], row['dropoff_longitude'])
    return geodesic(start, end).kilometers

df_uber['Distance_km'] = df_uber.apply(calculate_distance, axis=1)

#Car Model
def car_model(value):
    if value>0 and value<3:
        return "MINI"
    elif value>2 and value<5:
        return "XL"
    elif value>=5:
        return "XUV"
df_uber['car_model']=df_uber['passenger_count'].map(car_model)

#Maximum and Minimum Value of Fare_Amount
df_uber['fare_amount'].min(),df_uber['fare_amount'].max()

#Removing the value which is less than and equalto zero from Fare Amount
df_uber.drop(df_uber[df_uber['fare_amount']<=0].index,inplace=True)

#Maximum and Minimum Value of Distance_km
df_uber['Distance_km'].min(),df_uber['Distance_km'].max()

Mean_D=df_uber['Distance_km'].mean()
Median_D=df_uber['Distance_km'].median()
Mode_D=df_uber['Distance_km'].mode()
print(Mean_D,Median_D,Mode_D)

#Filling the Distance_km Zero Values with Median Value
df_uber['Distance_km']=df_uber['Distance_km'].replace(0,Median_D)

#Dropping the unnecessary columns
df_uber.drop(['Unnamed: 0','key','pickup_datetime'],axis=1,inplace=True)

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
df_uber.to_sql('UBER_Data',con=engine,if_exists='append',index=False)
#Reading the Data
DF=pd.read_sql_table('UBER_Data',con=engine)
DF.head()

#Converting the non-numerical values into Numerical Values for Model training
#Label Encoding
from sklearn.preprocessing import LabelEncoder
encoding=LabelEncoder()
DF['car_model']=encoding.fit_transform(DF['car_model'])
DF['Day_of_Name']=encoding.fit_transform(DF['Day_of_Name'])

#Outlier Detection

from scipy.stats import zscore

# Calculate Z-scores
DF['fare_amount_zscore'] = zscore(DF['fare_amount'])
DF['distance_km_zscore'] = zscore(DF['Distance_km'])

# Define threshold for outliers
threshold = 3

# Identify outliers
outliers_fare = DF[(DF['fare_amount_zscore']> threshold)]
outliers_distance = DF[(DF['distance_km_zscore']> threshold)]

print("Number of fare amount outliers:", len(outliers_fare))
print("Number of distance outliers:", len(outliers_distance))

#Cleaned Data--After removing the Outliers from Data[Fare Amount,Distance]
data = DF[(DF['fare_amount_zscore'] <= threshold) & (DF['distance_km_zscore'] <= threshold)]

# Drop Z-score columns
data.drop(columns=['fare_amount_zscore', 'distance_km_zscore'], inplace=True)

#Analysis
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#Dropped Data Percentage
Original_Data=len(df)
Cleaned_Data=len(data)
Dropped_Data=Original_Data-Cleaned_Data
#Plot
plt.figure(figsize=(6, 4))
plt.pie([Original_Data,Dropped_Data],explode=(0.1,0),labels=['Original_Data','Dropped_Data'],
        colors=['blue','pink'],autopct='%1.1f%%',startangle=160)
plt.axis('equal')
plt.title('Proportion of Original and Dropped Data')
plt.show()


# Fare Amount Distribution
#Before Removing Outliers
plt.figure(figsize=(10, 6))
sns.histplot(df['fare_amount'], kde=True)
plt.title('Distribution of Fare Amount-Before Removing Outliers')
plt.xlabel('Fare Amount ($)')
plt.ylabel('Frequency')
plt.show()
#After Removing Outliers
plt.figure(figsize=(10, 6))
sns.histplot(data['fare_amount'], kde=True)
plt.title('Distribution of Fare Amount-After Removing Outliers')
plt.xlabel('Fare Amount ($)')
plt.ylabel('Frequency')
plt.show()

# Trips by Car Model
plt.figure(figsize=(10, 6))
sns.countplot(x=data['car_model'])
plt.title('Number of Trips by Car Model')
plt.xlabel('Car Model')
plt.ylabel('Number of Trips')
plt.show()

# Trips by Day of Week
plt.figure(figsize=(10, 6))
sns.countplot(x=data['Day_of_Name'])
plt.title('Number of Trips by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Number of Trips')
plt.show()

# Trips by Hour of Day
plt.figure(figsize=(10, 6))
sns.countplot(x=data['Hour'])
plt.title('Number of Trips by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Trips')
plt.show()

# Correlation matrix
correlation_matrix = data.corr()

# Plot the correlation matrix
plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Amount Distribution
#Before Cleaning
plt.figure(figsize=(10, 6))
sns.boxplot(df['fare_amount'])
plt.title('Original data')
plt.show()
#After Cleaning
plt.figure(figsize=(10, 6))
sns.boxplot(data['fare_amount'])
plt.title('Cleaned Data')
plt.show()

data.shape

#Model Selection
x=data[['passenger_count','Year','Month','Day','Day_of_Name','Day_of_Week','Hour','Minute','Second','Distance_km','car_model']].values
y=data['fare_amount'].values
#Reshape
x=np.array(x)
y=np.array(y)

#Train Test Split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#Linear Regression
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)

#Error Metrics
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
y_pred =model.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2Score = r2_score(y_test,y_pred)
print(f'LR RMSE: {mse}')
print(f'LR MAE: {mae}')
print(f'LR RMSE: {rmse}')
print(f'LR R2Score: {r2Score}')

LR RMSE: 17.73560282497412
LR MAE: 2.4344724748407174
LR RMSE: 4.211365909651419
LR R2Score: 0.5754280699638983


#Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# Initialize and train the Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(x_train, y_train)
# Predict and evaluate
y_pred = rf_model.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2Score=r2_score(y_test, y_pred)
print(f'Random Forest RMSE: {mse}')
print(f'Random Forest MAE: {mae}')
print(f'Random Forest RMSE: {rmse}')
print(f'Random Forest Score: {r2Score}')
Random Forest RMSE: 7.875996179291093
Random Forest MAE: 1.822877051029778
Random Forest RMSE: 2.806420527877298
Random Forest Score: 0.8075611462666388

!pip install xgboost
import xgboost as xgb
from xgboost import XGBRegressor
# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# Initialize and train the Random Forest Regressor
xgb_model = XGBRegressor(random_state=42)
xgb_model.fit(x_train, y_train)
# Predict and evaluate
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
y_pred = xgb_model.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2Score=r2_score(y_test, y_pred)
print(f'XGB MSE: {mse}')
print(f'XGB MAE: {mae}')
print(f'XGB RMSE: {rmse}')
print(f'XGB Score: {r2Score}')
XGB MSE: 7.66136833059755
XGB MAE: 1.7455537413922755
XGB RMSE: 2.7679176885517296
XGB Score: 0.8128052749129226

#WHY XGB MODEL IS BEST
#1. Performance Metrics
Using the XGBoost model, you might observe lower RMSE (Root Mean Squared Error) and 
MAE (Mean Absolute Error) compared to other models. This indicates that XGBoost is 
more accurate in predicting fare amounts.

#PICKLE

import pickle
pickle.dump(xgb_model,open('xgb_model.pkl','wb'))

