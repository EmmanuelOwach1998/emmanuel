#%%
#author information.
"""
Created on Wed Apr  1 13:31:38 2020

This program uses an artificial recurrent neural network called Long Short Term Memory to predict the closing price of stock of an institution(Amazon) using past 6 month(180 days) stock prices.

"""

#%%
#import the needed libraries to use.
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#%%
#get the stock quote of the companies to use for a particular time period and display it.
amazondataset = pd.read_csv('AMZN.csv')
amazondataset

#%%
#Get the number of rows an columns in the whole amazon dataset.
amazondataset.shape

#%%
#visualize the closing price history.
plt.figure(figsize=(13,8))
plt.title('Close price history')
plt.plot(amazondataset['Close'])
plt.xlabel('Days elapsed', fontsize = 18)
plt.ylabel('Closing Price in USD ($)',fontsize = 18)
plt.show

#%%
#Create a new dataframe from the amazon dataset with only the 'Close' column.
amazonclose  = amazondataset.filter(['Close'])

#Convert the amazonclose dataframe to a numpy array.
amazonarray = amazonclose.values

#Get the number of rows to train the model on.
training_amazonarray = math.ceil(len(amazonarray) * .80 )
training_amazonarray

#%%
#Scale the data.
amazonscaler = MinMaxScaler(feature_range=(0,1))
amazonscaled_data = amazonscaler.fit_transform(amazonarray)
amazonscaled_data

#%%
#Create the training dataset.

#Create the scaled training data set
train_data = amazonscaled_data[0:training_amazonarray , :]

#Split the data into x_train(independent variables) and y_train datasets(dependent variables).
x_train = []
y_train = []

for i in range(180, len(train_data)):
    x_train.append(train_data[i-180:i, 0])
    y_train.append(train_data[i, 0])
    if i <= 180:
        print(x_train)
        print(y_train)
        print()
        
#%%
#Convert the x_train and y_train to numpy arrays.
x_train, y_train = np.array(x_train), np.array(y_train)

#%%
#Reshape the data.
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train.shape

#%%
#Build the LSTM model.
model = Sequential()
model.add(LSTM(170, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(170, return_sequences=False))
model.add(Dense(85))
model.add(Dense(1))

#%%
#Compile the model.
model.compile(optimizer='adam', loss='mean_squared_error')

#%%
#Train the model.
model.fit(x_train, y_train, batch_size = 1, epochs = 1)

#%%
#Create the testing dataset.
#Create a new arraycontaining scaled values
test_data = amazonscaled_data[training_amazonarray - 180: , :]

#Create the datasets x_test and y_test.
x_test = []
y_test = amazonarray[training_amazonarray:, :]
for i in range(180, len(test_data)):
    x_test.append(test_data[i-180:i, 0])
    
#%%
#Convert the data to a numpy array.
x_test = np.array(x_test)

#%%
#Reshape the data.
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#%%
#Get the model's predicted price values.
predictions = model.predict(x_test)
predictions = amazonscaler.inverse_transform(predictions)

#%%
#Get the root mean squared error(RMSE)
rmse = np.sqrt( np.mean(predictions - y_test)**2 )  
rmse

#%%
#Plot the data.
train = amazonclose[:training_amazonarray]
valid = amazonclose[training_amazonarray:]
valid['Predictions'] = predictions

#Visualize the model.
plt.figure(figsize=(13,8))
plt.title('Actual and predictive price model.')
plt.xlabel('Days elapsed', fontsize=18)
plt.ylabel('Close price in USD($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

#%%
#Show the valid actual and predicted price of the already collected dataset.
valid

#%%
#Get the predicted close price for the next day. 

#Get the amazon stock quote with valid price information for the set timeframe.
amazonquote = pd.read_csv('AMZN.csv')

#Create a new dataframe from the amazonquote dataset with only the 'Close' column.
amazonquote_close = amazonquote.filter(['Close'])

#Get the last 6 months (180 days) closing price values and convert the dataframe to an arrray.
last180days = amazonquote_close[-180:].values

#Scale the data to be values between 0 and 1.
last180days_scaled = amazonscaler.transform(last180days)

#Create an empty list
X_test = []

#Append the past 6 months to the new list created.
X_test.append(last180days_scaled)

#Convert the A_test dataset to a numpy array.
X_test = np.array(X_test)
 
#Reshape the data.
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1], 1))

#Get the predicted scaled price.
pred_price = model.predict(X_test)

#Undo the scaling.
pred_price = amazonscaler.inverse_transform(pred_price)

#Print the predicted price.
print(pred_price)

#%%
#Get the exact close price for the next day.
amazonexact = web.DataReader('AMZN', data_source = 'yahoo', start = '2020-04-06', end = '2020-04-06')
print(amazonexact['Close'])

#%%
#assign amazon exact value as a float.
amazonexactfloat = amazonexact.iat[0,3]

#%%
#Calculate the level of accuracy of the LSTM Model and output it.
accuracy = ((pred_price/amazonexactfloat) * 100)
print(accuracy)

#%%
#Create a dataframe containing actual and predicted for the next day.
nextday = (amazonexact['Close']) 
nextday['Predicted price'] = pred_price
nextday['Accuracy'] = accuracy

#Display the 2 prices(actual and estimated) and the level of accuracy.
nextday

#%%
