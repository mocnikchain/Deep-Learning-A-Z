import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset_train = pd.read_csv('Part 3 - Recurrent Neural Networks/Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range= (0,1))
training_set_scaled = sc.fit_transform(training_set)

y_train = []
X_train = []
# Creating datta structure with 60 timestamps and 1 output (days)
for i in range(60, len(training_set_scaled)):
    print(i)
    X_train.append(training_set_scaled[i-60:i, 0]) #60 not included
    y_train.append(training_set_scaled[i, 0]) # only 60 row

X_train, y_train = np.array(X_train), np.array(y_train)
#For an LSTM layer, you need the shape like (NumberOfExamples, TimeSteps, FeaturesPerStep)
# Reshaping into 3 dimenzions # #1. num of stock prices 2. num of timestamps, 3. number of indicators/predictors
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)) 
############################################################################
# Part 2 - building the RNN
#Importing libs and pckges
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
# initialize the RNN
regressor = Sequential()
#Adding the first LSTM layer and some froppout regularisation
# units = neurons, return_sequences=True vsi razn zadnjega 
regressor.add(LSTM(units = 50, return_sequences=True, input_shape = (X_train.shape[1], 1)))
# how many neurons to be dropped in order of regularisation 20% - 0.2 
# --> 10 neurons will be dropped
regressor.add(Dropout(0.2))

#Adding a second LSTM layer and some froppout regularisation
regressor.add(LSTM(units = 50, return_sequences=True))
regressor.add(Dropout(0.2))
#Adding a third LSTM layer and some froppout regularisation
regressor.add(LSTM(units = 50, return_sequences=True))
regressor.add(Dropout(0.2))
#Adding a fourth LSTM layer and some froppout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))
#Adding a output layer
regressor.add(Dense(units=1))

# Compiling the RNN - Configures the model for training.
# we will use adam optimizer instead of default rmsprop
regressor.compile(optimizer= "adam", loss="mean_squared_error")
# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size=32) #epoch = poizkusi, batches = skupina

#############################################################################
#Part 3 - Make predictions and visualizing the results
# Getting the real stock prices of 2017
dataset_test = pd.read_csv('Part 3 - Recurrent Neural Networks/Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017 
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
# all inputs we need to predict stock prices of January 2017
inputs = dataset_total[len(dataset_total)- len(dataset_test)-60:].values
# reshape
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
# Creating datta structure with 60 timestamps and 1 output (days)
for i in range(60, 80): #imamo 20 podatkov v january (test) in gledamo 60 nazaj
    X_test.append(inputs[i-60:i, 0]) #60 not included
X_test = np.array(X_test) 
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1)) 
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualize the results
plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()