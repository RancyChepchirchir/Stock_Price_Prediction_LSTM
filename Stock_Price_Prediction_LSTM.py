# Import libraries
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import warnings
warnings.filterwarnings('ignore')

# stock data sets
# AAPL
AAPL_daily = pd.read_csv("AAPL_daily.csv")
AAPL_daily = AAPL_daily.set_index('Date')
AAPL_weekly = pd.read_csv("AAPL_weekly.csv")
AAPL_weekly = AAPL_weekly.set_index('Date')
AAPL_monthly = pd.read_csv("AAPL_monthly.csv")
AAPL_monthly = AAPL_monthly.set_index('Date')
# MSFT
MSFT_daily = pd.read_csv("MSFT_daily.csv")
MSFT_daily = MSFT_daily.set_index('Date')
MSFT_weekly = pd.read_csv("MSFT_weekly.csv")
MSFT_weekly = MSFT_weekly.set_index('Date')
MSFT_monthly = pd.read_csv("MSFT_monthly.csv")
MSFT_monthly = MSFT_monthly.set_index('Date')
# Meta
META_daily = pd.read_csv("META_daily.csv")
META_daily = META_daily.set_index('Date')
META_weekly = pd.read_csv("META_weekly.csv")
META_weekly = META_weekly.set_index('Date')
META_monthly = pd.read_csv("META_monthly.csv")
META_monthly = META_monthly.set_index('Date')

# Sorting & Standardizing the datasets
# Creating a new dataframe with only the 'Close' column
AAPL_daily_data, MSFT_daily_data, META_daily_data = AAPL_daily.filter(['Close']), MSFT_daily.filter(['Close']), META_daily.filter(['Close'])
AAPL_weekly_data, MSFT_weekly_data, META_weekly_data = AAPL_weekly.filter(['Close']), MSFT_weekly.filter(['Close']), META_weekly.filter(['Close'])
AAPL_monthly_data, MSFT_monthly_data, META_monthly_data = AAPL_monthly.filter(['Close']), MSFT_monthly.filter(['Close']), META_monthly.filter(['Close'])
# Converting the dataframe to a numpy array
AAPL_daily_dataset, MSFT_daily_dataset, META_daily_dataset = AAPL_daily_data.values, MSFT_daily_data.values, META_daily_data.values
AAPL_weekly_dataset, MSFT_weekly_dataset, META_weekly_dataset = AAPL_weekly_data.values, MSFT_weekly_data.values, META_weekly_data.values
AAPL_monthly_dataset, MSFT_monthly_dataset, META_monthly_dataset = AAPL_monthly_data.values, MSFT_monthly_data.values, META_monthly_data.values

# Get/Compute the number of rows to train the daily models on
training_AAPL_daily_data_len = math.ceil( len(AAPL_daily_dataset) *.8)
training_MSFT_daily_data_len = math.ceil( len(MSFT_daily_dataset) *.8)
training_META_daily_data_len = math.ceil( len(META_daily_dataset) *.8)
# Get/Compute the number of rows to train the weekly models on
training_AAPL_weekly_data_len = math.ceil( len(AAPL_weekly_dataset) *.8)
training_MSFT_weekly_data_len = math.ceil( len(MSFT_weekly_dataset) *.8)
training_META_weekly_data_len = math.ceil( len(META_weekly_dataset) *.8)
# Get/Compute the number of rows to train the monthly models on
training_AAPL_monthly_data_len = math.ceil( len(AAPL_monthly_dataset) *.8)
training_MSFT_monthly_data_len = math.ceil( len(MSFT_monthly_dataset) *.8)
training_META_monthly_data_len = math.ceil( len(META_monthly_dataset) *.8)

# here we are Scaling the all of the data to be values between 0 and 1 
scaler = MinMaxScaler(feature_range=(0, 1)) 
scaled_AAPL_daily_data, scaled_MSFT_daily_data, scaled_META_daily_data = scaler.fit_transform(AAPL_daily_dataset), scaler.fit_transform(MSFT_daily_dataset), scaler.fit_transform(META_daily_dataset)
scaled_AAPL_weekly_data, scaled_MSFT_weekly_data, scaled_META_weekly_data = scaler.fit_transform(AAPL_weekly_dataset), scaler.fit_transform(MSFT_weekly_dataset), scaler.fit_transform(META_weekly_dataset)
scaled_AAPL_monthly_data, scaled_MSFT_monthly_data, scaled_META_monthly_data = scaler.fit_transform(AAPL_monthly_dataset), scaler.fit_transform(MSFT_monthly_dataset), scaler.fit_transform(META_monthly_dataset)

#Creating the daily scaled training data set
train_AAPL_daily_data = scaled_AAPL_daily_data[0:training_AAPL_daily_data_len  , : ]
train_MSFT_daily_data = scaled_MSFT_daily_data[0:training_MSFT_daily_data_len  , : ]
train_META_daily_data = scaled_META_daily_data[0:training_META_daily_data_len  , : ]
#Creating the weekly scaled training data set
train_AAPL_weekly_data = scaled_AAPL_weekly_data[0:training_AAPL_weekly_data_len  , : ]
train_MSFT_weekly_data = scaled_MSFT_weekly_data[0:training_MSFT_weekly_data_len  , : ]
train_META_weekly_data = scaled_META_weekly_data[0:training_META_weekly_data_len  , : ]
#Creating the monthly scaled training data set
train_AAPL_monthly_data = scaled_AAPL_monthly_data[0:training_AAPL_monthly_data_len  , : ]
train_MSFT_monthly_data = scaled_MSFT_monthly_data[0:training_MSFT_monthly_data_len  , : ]
train_META_monthly_data = scaled_META_monthly_data[0:training_META_monthly_data_len  , : ]

#Spliting the daily data into x_train and y_train data sets
X_train_AAPL_daily, y_train_AAPL_daily = [], []
for i in range(60,len(train_AAPL_daily_data)):
    X_train_AAPL_daily.append(train_AAPL_daily_data[i-60:i,0])
    y_train_AAPL_daily.append(train_AAPL_daily_data[i,0])
X_train_MSFT_daily, y_train_MSFT_daily = [], []
for i in range(60,len(train_MSFT_daily_data)):
    X_train_MSFT_daily.append(train_MSFT_daily_data[i-60:i,0])
    y_train_MSFT_daily.append(train_MSFT_daily_data[i,0])  
X_train_META_daily, y_train_META_daily = [], []
for i in range(60,len(train_META_daily_data)):
    X_train_META_daily.append(train_META_daily_data[i-60:i,0])
    y_train_META_daily.append(train_META_daily_data[i,0])    

#Spliting the weekly data into x_train and y_train data sets
X_train_AAPL_weekly, y_train_AAPL_weekly = [], []
for i in range(60,len(train_AAPL_weekly_data)):
    X_train_AAPL_weekly.append(train_AAPL_weekly_data[i-60:i,0])
    y_train_AAPL_weekly.append(train_AAPL_weekly_data[i,0])
X_train_MSFT_weekly, y_train_MSFT_weekly = [], []
for i in range(60,len(train_MSFT_weekly_data)):
    X_train_MSFT_weekly.append(train_MSFT_weekly_data[i-60:i,0])
    y_train_MSFT_weekly.append(train_MSFT_weekly_data[i,0])  
X_train_META_weekly, y_train_META_weekly = [], []
for i in range(60,len(train_META_weekly_data)):
    X_train_META_weekly.append(train_META_weekly_data[i-60:i,0])
    y_train_META_weekly.append(train_META_weekly_data[i,0]) 

#Spliting the monthly data into x_train and y_train data sets
X_train_AAPL_monthly, y_train_AAPL_monthly = [], []
for i in range(60,len(train_AAPL_monthly_data)):
    X_train_AAPL_monthly.append(train_AAPL_monthly_data[i-60:i,0])
    y_train_AAPL_monthly.append(train_AAPL_monthly_data[i,0])
X_train_MSFT_monthly, y_train_MSFT_monthly = [], []
for i in range(60,len(train_MSFT_monthly_data)):
    X_train_MSFT_monthly.append(train_MSFT_monthly_data[i-60:i,0])
    y_train_MSFT_monthly.append(train_MSFT_monthly_data[i,0])  
X_train_META_monthly, y_train_META_monthly = [], []
for i in range(60,len(train_META_monthly_data)):
    X_train_META_monthly.append(train_META_monthly_data[i-60:i,0])
    y_train_META_monthly.append(train_META_monthly_data[i,0])     

# Here we are Converting daily x_train and y_train to numpy arrays
X_train_AAPL_daily, y_train_AAPL_daily = np.array(X_train_AAPL_daily), np.array(y_train_AAPL_daily)
X_train_MSFT_daily, y_train_MSFT_daily = np.array(X_train_MSFT_daily), np.array(y_train_MSFT_daily)
X_train_META_daily, y_train_META_daily = np.array(X_train_META_daily), np.array(y_train_META_daily)

# Here we are Converting weekly x_train and y_train to numpy arrays
X_train_AAPL_weekly, y_train_AAPL_weekly = np.array(X_train_AAPL_weekly), np.array(y_train_AAPL_weekly)
X_train_MSFT_weekly, y_train_MSFT_weekly = np.array(X_train_MSFT_weekly), np.array(y_train_MSFT_weekly)
X_train_META_weekly, y_train_META_weekly = np.array(X_train_META_weekly), np.array(y_train_META_weekly)

# Here we are Converting monthly x_train and y_train to numpy arrays
X_train_AAPL_monthly, y_train_AAPL_monthly = np.array(X_train_AAPL_monthly), np.array(y_train_AAPL_monthly)
X_train_MSFT_monthly, y_train_MSFT_monthly = np.array(X_train_MSFT_monthly), np.array(y_train_MSFT_monthly)
X_train_META_monthly, y_train_META_monthly = np.array(X_train_META_monthly), np.array(y_train_META_monthly)

# Here we are reshaping the daily data into the shape accepted by the LSTM
X_train_AAPL_daily = np.reshape(X_train_AAPL_daily, (X_train_AAPL_daily.shape[0],X_train_AAPL_daily.shape[1],1))
X_train_MSFT_daily = np.reshape(X_train_MSFT_daily, (X_train_MSFT_daily.shape[0],X_train_MSFT_daily.shape[1],1))
X_train_META_daily = np.reshape(X_train_META_daily, (X_train_META_daily.shape[0],X_train_META_daily.shape[1],1))

# Here we are reshaping the weekly data into the shape accepted by the LSTM
X_train_AAPL_weekly = np.reshape(X_train_AAPL_weekly, (X_train_AAPL_weekly.shape[0],X_train_AAPL_weekly.shape[1],1))
X_train_MSFT_weekly = np.reshape(X_train_MSFT_weekly, (X_train_MSFT_weekly.shape[0],X_train_MSFT_weekly.shape[1],1))
X_train_META_weekly = np.reshape(X_train_META_weekly, (X_train_META_weekly.shape[0],X_train_META_weekly.shape[1],1))

# Here we are reshaping the monthly data into the shape accepted by the LSTM
X_train_AAPL_monthly = np.reshape(X_train_AAPL_monthly, (X_train_AAPL_monthly.shape[0],X_train_AAPL_monthly.shape[1],1))
X_train_MSFT_monthly = np.reshape(X_train_MSFT_monthly, (X_train_MSFT_monthly.shape[0],X_train_MSFT_monthly.shape[1],1))
X_train_META_monthly = np.reshape(X_train_META_monthly, (X_train_META_monthly.shape[0],X_train_META_monthly.shape[1],1))

# Build and train the LSTM model:
# Build the LSTM network model for daily Apple stock
model = Sequential()
model.add(LSTM(units=50, return_sequences=True,input_shape=(X_train_AAPL_daily.shape[1],1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))
# here we are Compiling the model
model.compile(optimizer='adam', loss='mean_squared_error')
# here we are training the model
model.fit(X_train_AAPL_daily, y_train_AAPL_daily, batch_size=5, epochs=25, verbose=0)

# Building the LSTM network model for daily Microsoft stock
model = Sequential()
model.add(LSTM(units=50, return_sequences=True,input_shape=(X_train_MSFT_daily.shape[1],1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))
# here we are Compiling the model
model.compile(optimizer='adam', loss='mean_squared_error')
# here we are training the model
model.fit(X_train_MSFT_daily, y_train_MSFT_daily, batch_size=1, epochs=1, verbose=0)

# Building the LSTM network model for daily Meta stock
model = Sequential()
model.add(LSTM(units=50, return_sequences=True,input_shape=(X_train_META_daily.shape[1],1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))
# here we are Compiling the model
model.compile(optimizer='adam', loss='mean_squared_error')
# here we are training the model
model.fit(X_train_META_daily, y_train_META_daily, batch_size=1, epochs=1, verbose=0)

# Building the LSTM network model for weekly Apple stock
model = Sequential()
model.add(LSTM(units=50, return_sequences=True,input_shape=(X_train_AAPL_weekly.shape[1],1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))
# here we are Compiling the model
model.compile(optimizer='adam', loss='mean_squared_error')
# here we are training the model
model.fit(X_train_AAPL_weekly, y_train_AAPL_weekly, batch_size=5, epochs=25, verbose=0)

# Building the LSTM network model for weekly Microsoft stock
model = Sequential()
model.add(LSTM(units=50, return_sequences=True,input_shape=(X_train_MSFT_weekly.shape[1],1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))
# here we are Compiling the model
model.compile(optimizer='adam', loss='mean_squared_error')
# here we are training the model
model.fit(X_train_MSFT_weekly, y_train_MSFT_weekly, batch_size=1, epochs=1, verbose=0)

# Building the LSTM network model for weekly Meta stock
model = Sequential()
model.add(LSTM(units=50, return_sequences=True,input_shape=(X_train_META_weekly.shape[1],1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))
# here we are Compiling the model
model.compile(optimizer='adam', loss='mean_squared_error')
# here we are training the model
model.fit(X_train_META_weekly, y_train_META_weekly, batch_size=1, epochs=1, verbose=0)

# Building the LSTM network model for monthly Apple stock
model = Sequential()
model.add(LSTM(units=50, return_sequences=True,input_shape=(X_train_AAPL_monthly.shape[1],1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))
# here we are Compiling the model
model.compile(optimizer='adam', loss='mean_squared_error')
# here we are training the model
model.fit(X_train_AAPL_monthly, y_train_AAPL_monthly, batch_size=5, epochs=25, verbose=0)

# Building the LSTM network model for monthly Microsoft stock
model = Sequential()
model.add(LSTM(units=50, return_sequences=True,input_shape=(X_train_MSFT_monthly.shape[1],1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))
# here we are Compiling the model
model.compile(optimizer='adam', loss='mean_squared_error')
# here we are training the model
model.fit(X_train_MSFT_monthly, y_train_MSFT_monthly, batch_size=1, epochs=1, verbose=0)

# Building the LSTM network model for monthly Meta stock
model = Sequential()
model.add(LSTM(units=50, return_sequences=True,input_shape=(X_train_META_monthly.shape[1],1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))
# here we are Compiling the model
model.compile(optimizer='adam', loss='mean_squared_error')
# here we are training the model
model.fit(X_train_META_monthly, y_train_META_monthly, batch_size=1, epochs=1, verbose=0)

# Testing and Prediction
# Daily testing datasets
test_AAPL_daily_data = scaled_AAPL_daily_data[training_AAPL_daily_data_len - 60: , : ]
test_MSFT_daily_data = scaled_MSFT_daily_data[training_MSFT_daily_data_len - 60: , : ]
test_META_daily_data = scaled_META_daily_data[training_META_daily_data_len - 60: , : ]
# Weekly testing datasets
test_AAPL_weekly_data = scaled_AAPL_weekly_data[training_AAPL_weekly_data_len - 60: , : ]
test_MSFT_weekly_data = scaled_MSFT_weekly_data[training_MSFT_weekly_data_len - 60: , : ]
test_META_weekly_data = scaled_META_weekly_data[training_META_weekly_data_len - 60: , : ]
# Monthly testing datasets
test_AAPL_monthly_data = scaled_AAPL_monthly_data[training_AAPL_monthly_data_len - 60: , : ]
test_MSFT_monthly_data = scaled_MSFT_monthly_data[training_MSFT_monthly_data_len - 60: , : ]
test_META_monthly_data = scaled_META_monthly_data[training_META_monthly_data_len - 60: , : ]

#Creating the X_test and y_test for daily data sets
X_test_AAPL_daily_data, y_test_AAPL_daily_data = [], AAPL_daily_dataset[training_AAPL_daily_data_len : , : ]
X_test_MSFT_daily_data, y_test_MSFT_daily_data = [], MSFT_daily_dataset[training_MSFT_daily_data_len : , : ]
X_test_META_daily_data, y_test_META_daily_data = [], META_daily_dataset[training_META_daily_data_len : , : ]
#Creating the X_test and y_test for weekly data sets
X_test_AAPL_weekly_data, y_test_AAPL_weekly_data = [], AAPL_weekly_dataset[training_AAPL_weekly_data_len : , : ]
X_test_MSFT_weekly_data, y_test_MSFT_weekly_data = [], MSFT_weekly_dataset[training_MSFT_weekly_data_len : , : ]
X_test_META_weekly_data, y_test_META_weekly_data = [], META_weekly_dataset[training_META_weekly_data_len : , : ]
#Creating the X_test and y_test for monthly data sets
X_test_AAPL_monthly_data, y_test_AAPL_monthly_data = [], AAPL_monthly_dataset[training_AAPL_monthly_data_len : , : ]
X_test_MSFT_monthly_data, y_test_MSFT_monthly_data = [], MSFT_monthly_dataset[training_MSFT_monthly_data_len : , : ]
X_test_META_monthly_data, y_test_META_monthly_data = [], META_monthly_dataset[training_META_monthly_data_len : , : ]
# Daily data
for i in range(60,len(test_AAPL_daily_data)):
    X_test_AAPL_daily_data.append(test_AAPL_daily_data[i-60:i,0])
for i in range(60,len(test_MSFT_daily_data)):
    X_test_MSFT_daily_data.append(test_MSFT_daily_data[i-60:i,0])
for i in range(60,len(test_META_daily_data)):
    X_test_META_daily_data.append(test_META_daily_data[i-60:i,0])  
# Weekly data
for i in range(60,len(test_AAPL_weekly_data)):
    X_test_AAPL_weekly_data.append(test_AAPL_weekly_data[i-60:i,0])
for i in range(60,len(test_MSFT_weekly_data)):
    X_test_MSFT_weekly_data.append(test_MSFT_weekly_data[i-60:i,0])
for i in range(60,len(test_META_weekly_data)):
    X_test_META_weekly_data.append(test_META_weekly_data[i-60:i,0]) 
# Monthly data
for i in range(60,len(test_AAPL_monthly_data)):
    X_test_AAPL_monthly_data.append(test_AAPL_monthly_data[i-60:i,0])
for i in range(60,len(test_MSFT_monthly_data)):
    X_test_MSFT_monthly_data.append(test_MSFT_monthly_data[i-60:i,0])
for i in range(60,len(test_META_monthly_data)):
    X_test_META_monthly_data.append(test_META_monthly_data[i-60:i,0])     

# here we are converting X_test to arrays
X_test_AAPL_daily_data, X_test_MSFT_daily_data, X_test_META_daily_data = np.array(X_test_AAPL_daily_data), np.array(X_test_MSFT_daily_data), np.array(X_test_META_daily_data)
X_test_AAPL_weekly_data, X_test_MSFT_weekly_data, X_test_META_weekly_data = np.array(X_test_AAPL_weekly_data), np.array(X_test_MSFT_weekly_data), np.array(X_test_META_weekly_data)
X_test_AAPL_monthly_data, X_test_MSFT_monthly_data, X_test_META_monthly_data = np.array(X_test_AAPL_monthly_data), np.array(X_test_MSFT_monthly_data), np.array(X_test_META_monthly_data)

# reshape the daily data into the shape/dimension accepted by the LSTM  
X_test_AAPL_daily_data = np.reshape(X_test_AAPL_daily_data, (X_test_AAPL_daily_data.shape[0],X_test_AAPL_daily_data.shape[1],1))
X_test_MSFT_daily_data = np.reshape(X_test_MSFT_daily_data, (X_test_MSFT_daily_data.shape[0],X_test_MSFT_daily_data.shape[1],1))
X_test_META_daily_data = np.reshape(X_test_META_daily_data, (X_test_META_daily_data.shape[0],X_test_META_daily_data.shape[1],1))
# reshape the daily data into the shape/dimension accepted by the LSTM  
X_test_AAPL_weekly_data = np.reshape(X_test_AAPL_weekly_data, (X_test_AAPL_weekly_data.shape[0],X_test_AAPL_weekly_data.shape[1],1))
X_test_MSFT_weekly_data = np.reshape(X_test_MSFT_weekly_data, (X_test_MSFT_weekly_data.shape[0],X_test_MSFT_weekly_data.shape[1],1))
X_test_META_weekly_data = np.reshape(X_test_META_weekly_data, (X_test_META_weekly_data.shape[0],X_test_META_weekly_data.shape[1],1))
# reshape the daily data into the shape/dimension accepted by the LSTM  
X_test_AAPL_monthly_data = np.reshape(X_test_AAPL_monthly_data, (X_test_AAPL_monthly_data.shape[0],X_test_AAPL_monthly_data.shape[1],1))
X_test_MSFT_monthly_data = np.reshape(X_test_MSFT_monthly_data, (X_test_MSFT_monthly_data.shape[0],X_test_MSFT_monthly_data.shape[1],1))
X_test_META_monthly_data = np.reshape(X_test_META_monthly_data, (X_test_META_monthly_data.shape[0],X_test_META_monthly_data.shape[1],1))

# now we are getting the models predicted price values (daily)
predictions_AAPL_daily, predictions_MSFT_daily, predictions_META_daily = model.predict(X_test_AAPL_daily_data), model.predict(X_test_MSFT_daily_data), model.predict(X_test_META_daily_data)
predictions_AAPL_daily, predictions_MSFT_daily, predictions_META_daily = scaler.inverse_transform(predictions_AAPL_daily), scaler.inverse_transform(predictions_MSFT_daily), scaler.inverse_transform(predictions_META_daily) #Undo scaling
# now we are getting the models predicted price values (weekly)
predictions_AAPL_weekly, predictions_MSFT_weekly, predictions_META_weekly = model.predict(X_test_AAPL_weekly_data), model.predict(X_test_MSFT_weekly_data), model.predict(X_test_META_weekly_data)
predictions_AAPL_weekly, predictions_MSFT_weekly, predictions_META_weekly = scaler.inverse_transform(predictions_AAPL_weekly), scaler.inverse_transform(predictions_MSFT_weekly), scaler.inverse_transform(predictions_META_weekly)
# now we are getting the models predicted price values (monthly)
predictions_AAPL_monthly, predictions_MSFT_monthly, predictions_META_monthly = model.predict(X_test_AAPL_monthly_data), model.predict(X_test_MSFT_monthly_data), model.predict(X_test_META_monthly_data)
predictions_AAPL_monthly, predictions_MSFT_monthly, predictions_META_monthly = scaler.inverse_transform(predictions_AAPL_monthly), scaler.inverse_transform(predictions_MSFT_monthly), scaler.inverse_transform(predictions_META_monthly)

# here we are calculaing the value of RMSE (daily models)
rmse_AAPL_daily = np.sqrt(np.mean(((predictions_AAPL_daily - y_test_AAPL_daily_data)**2)))
rmse_MSFT_daily = np.sqrt(np.mean(((predictions_MSFT_daily - y_test_MSFT_daily_data)**2)))
rmse_META_daily = np.sqrt(np.mean(((predictions_META_daily - y_test_META_daily_data)**2)))
# here we are calculaing the value of RMSE (weekly models)
rmse_AAPL_weekly = np.sqrt(np.mean(((predictions_AAPL_weekly - y_test_AAPL_weekly_data)**2)))
rmse_MSFT_weekly = np.sqrt(np.mean(((predictions_MSFT_weekly - y_test_MSFT_weekly_data)**2)))
rmse_META_weekly = np.sqrt(np.mean(((predictions_META_weekly - y_test_META_weekly_data)**2)))
# here we are calculaing the value of RMSE (monthly models)
rmse_AAPL_monthly = np.sqrt(np.mean(((predictions_AAPL_monthly - y_test_AAPL_monthly_data)**2)))
rmse_MSFT_monthly = np.sqrt(np.mean(((predictions_MSFT_monthly - y_test_MSFT_monthly_data)**2)))
rmse_META_monthly = np.sqrt(np.mean(((predictions_META_monthly - y_test_META_monthly_data)**2)))

# Create the daily data for visuals
train_AAPL_daily_data, valid_AAPL_daily_data = AAPL_daily_data[:training_AAPL_daily_data_len], AAPL_daily_data[training_AAPL_daily_data_len:]
train_MSFT_daily_data, valid_MSFT_daily_data = MSFT_daily_data[:training_MSFT_daily_data_len], MSFT_daily_data[training_MSFT_daily_data_len:]
train_META_daily_data, valid_META_daily_data = META_daily_data[:training_META_daily_data_len], META_daily_data[training_META_daily_data_len:]
valid_AAPL_daily_data['Predictions'], valid_MSFT_daily_data['Predictions'], valid_META_daily_data['Predictions']= predictions_AAPL_daily, predictions_MSFT_daily, predictions_META_daily

# Create the weekly data for visuals
train_AAPL_weekly_data, valid_AAPL_weekly_data = AAPL_weekly_data[:training_AAPL_weekly_data_len], AAPL_weekly_data[training_AAPL_weekly_data_len:]
train_MSFT_weekly_data, valid_MSFT_weekly_data = MSFT_weekly_data[:training_MSFT_weekly_data_len], MSFT_weekly_data[training_MSFT_weekly_data_len:]
train_META_weekly_data, valid_META_weekly_data = META_weekly_data[:training_META_weekly_data_len], META_weekly_data[training_META_weekly_data_len:]
valid_AAPL_weekly_data['Predictions'], valid_MSFT_weekly_data['Predictions'], valid_META_weekly_data['Predictions']= predictions_AAPL_weekly, predictions_MSFT_weekly, predictions_META_weekly

# Create the monthly data for visuals
train_AAPL_monthly_data, valid_AAPL_monthly_data = AAPL_monthly_data[:training_AAPL_monthly_data_len], AAPL_monthly_data[training_AAPL_monthly_data_len:]
train_MSFT_monthly_data, valid_MSFT_monthly_data = MSFT_monthly_data[:training_MSFT_monthly_data_len], MSFT_monthly_data[training_MSFT_monthly_data_len:]
train_META_monthly_data, valid_META_monthly_data = META_monthly_data[:training_META_monthly_data_len], META_monthly_data[training_META_monthly_data_len:]
valid_AAPL_monthly_data['Predictions'], valid_MSFT_monthly_data['Predictions'], valid_META_monthly_data['Predictions']= predictions_AAPL_monthly, predictions_MSFT_monthly,predictions_META_monthly

# Daily predictions
#Visualize the data
plt.figure(figsize=(16,8))
plt.title('Apple LSTM Model for Daily Stock Prices')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train_AAPL_daily_data['Close'])
plt.plot(valid_AAPL_daily_data[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

predicted_AAPL_daily = pd.DataFrame(predictions_AAPL_daily)
predicted_AAPL_daily.to_csv('predicted_closing_price_AAPL_daily.csv')

#Visualize the data
plt.figure(figsize=(16,8))
plt.title('Microsoft LSTM Model for Daily Stock Prices')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train_MSFT_daily_data['Close'])
plt.plot(valid_MSFT_daily_data[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

predicted_MSFT_daily = pd.DataFrame(predictions_MSFT_daily)
predicted_MSFT_daily.to_csv('predicted_closing_price_MSFT_daily.csv')

#Visualize the data
plt.figure(figsize=(16,8))
plt.title('Meta LSTM Model for Daily Stock Prices')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train_META_daily_data['Close'])
plt.plot(valid_META_daily_data[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

predicted_META_daily = pd.DataFrame(predictions_META_daily)
predicted_META_daily.to_csv('predicted_closing_price_META_daily.csv')

# Weekly Predictions
#Visualize the data
plt.figure(figsize=(16,8))
plt.title('Apple LSTM Model for Weekly Stock Prices')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train_AAPL_weekly_data['Close'])
plt.plot(valid_AAPL_weekly_data[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

predicted_AAPL_weekly = pd.DataFrame(predictions_AAPL_weekly)
predicted_AAPL_weekly.to_csv('predicted_closing_price_AAPL_weekly.csv')

#Visualize the data
plt.figure(figsize=(16,8))
plt.title('Microsoft LSTM Model for Weekly Stock Prices')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train_MSFT_weekly_data['Close'])
plt.plot(valid_MSFT_weekly_data[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

predicted_MSFT_weekly = pd.DataFrame(predictions_MSFT_weekly)
predicted_MSFT_weekly.to_csv('predicted_closing_price_MSFT_weekly.csv')

#Visualize the data
plt.figure(figsize=(16,8))
plt.title('Meta LSTM Model for Weekly Stock Prices')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train_META_weekly_data['Close'])
plt.plot(valid_META_weekly_data[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

predicted_META_weekly = pd.DataFrame(predictions_META_weekly)
predicted_META_weekly.to_csv('predicted_closing_price_META_weekly.csv')

#Visualize the data
plt.figure(figsize=(16,8))
plt.title('Apple LSTM Model for Monthly Stock Prices')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train_AAPL_monthly_data['Close'])
plt.plot(valid_AAPL_monthly_data[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

predicted_AAPL_monthly = pd.DataFrame(predictions_AAPL_monthly)
predicted_AAPL_monthly.to_csv('predicted_closing_price_AAPL_monthly.csv')

#Visualize the data
plt.figure(figsize=(16,8))
plt.title('Microsoft LSTM Model for Monthly Stock Prices')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train_MSFT_monthly_data['Close'])
plt.plot(valid_MSFT_monthly_data[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

predicted_MSFT_monthly = pd.DataFrame(predictions_MSFT_monthly)
predicted_MSFT_monthly.to_csv('predicted_closing_price_MSFT_monthly.csv')

#Visualize the data
plt.figure(figsize=(16,8))
plt.title('Meta LSTM Model for Monthly Stock Prices')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train_META_monthly_data['Close'])
plt.plot(valid_META_monthly_data[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

predicted_META_monthly = pd.DataFrame(predictions_META_monthly)
predicted_META_monthly.to_csv('predicted_closing_price_META_monthly.csv')

