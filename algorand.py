import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import math
from statsmodels.tsa.stattools import adfuller, kpss # Stationarity and detrending (ADF/KPSS)
from numpy import log
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import pacf
from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error
from keras.layers import LSTM, SimpleRNN
import tensorflow as tf
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# Load dataset
file_path = 'algorand.csv'
dataset = pd.read_csv(file_path)

# First view of the dataset
dataset.head()
dataset.describe()

# Check for null values
dataset.isnull().sum()

start_date = pd.to_datetime(dataset.date[0])
end_date = pd.to_datetime(dataset.date.values[-1])
dataset['date'] = pd.to_datetime(dataset['date'])
dataset.tail()

# Plotting Few price data
top_plt = plt.subplot2grid((5,4), (0, 0), rowspan=3, colspan=4)
top_plt.plot(dataset.date, dataset["price"])
plt.title('Historical stock prices of Algorand [21-06-2019 to 03-03-2023]')
bottom_plt = plt.subplot2grid((5,4), (3,0), rowspan=1, colspan=4)
bottom_plt.bar(dataset.date, dataset['total_volume'])
plt.title('\nAlgorand Trading Volume', y=-0.60)
plt.gcf().set_size_inches(16,10)
plt.show()

dataset.describe()

# Check datatype of Adj Close price
dataset['price'].dtype

# plotting correlation heatmap
plt.figure(figsize = (10, 6))
dataplot = sns.heatmap(dataset[['price', 'total_volume', 'market_cap']].corr(), cmap="BuPu", annot=True, 
                      fmt=".1f")
plt.show()
# We can't drop any of those features with market_cap and price having correlation at 0.8 , we would lose important informations

## ADF Test 
#ADF test is used to determine the presence of unit root in the series, and hence helps in understand if the series is stationary or not. The null and alternate hypothesis of this test are:
#Null Hypothesis: The series has a unit root.
#Alternate Hypothesis: The series has no unit root.
#If the null hypothesis in failed to be rejected, this test may provide evidence that the series is non-stationary.

result = adfuller(dataset.price.values, autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
for key, value in result[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}')
# ADF Stats value is greater than all critical values, and p-value is also greater than 0.05. So we can strongly reject the null hypothesis, and conclude that, Price value is Non-Stationary.

result = adfuller((log(dataset.price.values)), autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
for key, value in result[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}')
# After applying Log transformation also, ADF Stats value is greater than all critical values, and p-value is also greater than 0.05. It seems, Price value is purely Non-Stationary.

## KPSS test - Kwiatkowski Phillips Schmidt Shin
#KPSS is another test for checking the stationarity of a time series. The null and alternate hypothesis for the KPSS test are opposite that of the ADF test.
#Null Hypothesis: The process is trend stationary.
#Alternate Hypothesis: The series has a unit root (series is not stationary).

result = kpss(dataset['price'].values, regression='c')
print('\nKPSS Statistic: %f' % result[0])
print('p-value: %f' % result[1])
for key, value in result[3].items():
    print('Critial Values:')
    print(f'   {key}, {value}')
#Here we find that, KPSS stats value is too high than critical values. 
#So, we concluded that this time series is Non-Stationary

# Time Series Prediction #
# To perform forecasting, we will need a machine learning model. 
# Most people think of multi-linear regression when they want to predict values. 
# But for Time-series data, this is not a good idea. 
# The main reason to not opt for regression for Time-Series Data is we are 
# interested in predicting the future, which would be extrapolation 
# (predicting outside the range of the data) for linear regression. 
# And as we know that in linear regression any sort of 
# extrapolation is not advisable. 

# What Model to Use?
#1 ARIMA
#2 ARTIFICIAL NEURAL NETWORK
#3 RECURRENT NEURAL NETWORK
#4 LSTM - LONG SHORT TERM MEMORY
#5 CNN - CONVOLUTION NEURAL NETWORK

#1. ARIMA - Univariate Price Forecasting
#ARIMA is a class of models that ‘explains’ a given time series based on its 
# own past values, i.e, its own lags and the lagged forecast errors, 
# so that equation can be used to forecast future values. 
# Any ‘non-seasonal’ time series that exhibits patterns and is not a random 
# white noise can be modeled with ARIMA models. 
# The hypothesis testing (ADF) performed priorly, showed the prices 
# were not stationary, hence we can not use an ARIMA model. 
# We can use Neural networks for price predictions. 

#2. Artificial Neural Network
# Univariate Stock price forecasting
# Using price data, we'll forecast the stock price for the next day. 
data = dataset['price'].values
print('Shape of data: ', data.shape)

# Separate Train and Test data
train_length = int(len(data) * 0.8)
print('Train length: ', train_length)

train_data, test_data = data[:train_length], data[train_length:]
print('Shape of Train and Test data: ', train_data.shape, test_data.shape)

#  Change Shape - Need 2D data
train_data = train_data.reshape(-1, 1)
test_data = test_data.reshape(-1, 1)
print('Shape of Train and Test data: ', train_data.shape, test_data.shape)

# split a univariate sequence into supervised learning [Input and Output]
def create_dataset(dataset, lookback):
    dataX, dataY = [], []
    for i in range(len(dataset) - lookback -1):
        a = dataset[i: (i+lookback), 0]
        dataX.append(a)
        b = dataset[i+lookback, 0]
        dataY.append(b)
    return np.array(dataX), np.array(dataY)

# Automatically select Lag value from PACF graph

plot_pacf(data, lags=10)
plt.show()

# Taking Auto-correlation Lag value Greater than 10% 
pacf_value = pacf(data, nlags=20)
lag = 0
# collect lag values greater than 10% correlation 
for x in pacf_value:
    if x > 0.1:
        lag += 1
    else:
        break
print('Selected look_back (or lag = ): ', lag)

# Separate Input and Output
train_X, train_y = create_dataset(train_data, lag)
test_X, test_y = create_dataset(test_data, lag)

print('Shape of train_X and train_y: ', train_X.shape, train_y.shape)
print('Shape of test_X and test_y: ', test_X.shape, test_y.shape)

#How Data Looks Like - Input and Output
print(train_data[:20])            # original data
for x in range(len(train_X[:20])):
    print(test_X[x], test_y[x], )            
    # trainX and trainY after lookback

# Build an MLP model
# Fix random seed for reproducibility
# Thes seed value helps in initilizing random weights and biases to the neural network.  
np.random.seed(7)

model = Sequential()
model.add(Dense(64, input_dim = lag, activation='relu', name= "1st_hidden"))
# model.add(Dense(64, activation='relu', name = '2nd_hidden'))
model.add(Dense(1, name = 'Output_layer', activation='linear'))
# model.add(Activation("linear", name = 'Linear_activation'))
model.compile(loss="mean_squared_error", optimizer="adam")
model.summary()
# Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.

# Fit data to Model 
epoch_number = 100
batches = 64
history = model.fit(train_X, train_y, 
                    epochs = epoch_number, 
                    batch_size = batches, 
                    verbose = 1, 
                    shuffle=False, 
                    validation_split=0.1)

# Train and Validation Loss 
# plot history
plt.clf
plt.figure(figsize=(10,8))
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.xlabel('Number of Epochs')
plt.ylabel('Train and Test Loss')
plt.title('Train and Test loss per epochs [Univariate]')
plt.legend()
plt.show()

# Make prediction
testPredict = model.predict(test_X)
predicted_value = testPredict[:, 0]

# Evaluation Metrics to Measure Performance
def evaluate_forecast_results(actual, predicted):
    print('R2 Score: ', round(r2_score(actual, predicted), 2))
    print('MAE : ', round(mae(actual, predicted), 2))
    print('MSE: ', round(mean_squared_error(actual,predicted), 2))
    print('RMSE: ', round(math.sqrt(mean_squared_error(actual,predicted)), 2))
    print('NRMSE: ', NRMSE(actual, predicted))
    print('WMAPE: ', WMAPE(actual, predicted))
    
def NRMSE(actual, predicted):
    rmse = math.sqrt(mean_squared_error(actual,predicted))
    nrmse = rmse / np.mean(actual)
    return round(nrmse, 4)

def WMAPE(actual, predicted):
    abs_error = np.sum(actual - predicted)
    wmape = abs_error / np.sum(actual)
    return round(wmape, 4)

evaluate_forecast_results(test_y, predicted_value)

# Here we're plotting Test and Predicted data

plt.figure(figsize=(16, 8))
plt.rcParams.update({'font.size': 12})
plt.plot(test_y[:], '#0077be',label = 'Actual')
plt.plot(predicted_value, '#ff8841',label = 'Predicted')
plt.title('MLP Model for Algorand Price Forecasting')
plt.ylabel('Algorand Price [in Dollar]')
plt.xlabel('Time Steps [in Days] ')
plt.legend()
plt.show()

########################### 3 Recurrent Neural Network ####################
# Load Data - "Adj Close" price
data = dataset['price'].values
print('Shape of data: ', data.shape)

# Separate train and test data
train_length = int(len(data) * 0.8)
print('Train length: ', train_length)

train_data, test_data = data[:train_length], data[train_length:]
print('Shape of Train and Test data: ', len(train_data), len(test_data))

# split a univariate sequence into supervised learning [Input and Output]
from numpy import array
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

# Lag Value already Choosen from PACF Plot
pacf_value = pacf(data, nlags=20)
lag = 0
# collect lag values greater than 10% correlation 
for x in pacf_value:
    if x > 0.1:
        lag += 1
    else:
        break
print('Selected look_back (or lag = ): ', lag)

n_features = 1
train_X, train_y = split_sequence(train_data, lag)
test_X, test_y = split_sequence(test_data, lag)

print('Shape of train_X and train_y: ', train_X.shape, train_y.shape)
print('Shape of test_X and test_y: ', test_X.shape, test_y.shape)

# Reshape train_X and test_X to 3-Dimension
train_X = train_X.reshape((train_X.shape[0], train_X.shape[1], n_features))
test_X = test_X.reshape((test_X.shape[0], test_X.shape[1], n_features))
# New shape of train_X and test_X are :-
print('Shape of train_X and train_y: ', train_X.shape, train_y.shape)
print('Shape of test_X and test_y: ', test_X.shape, test_y.shape)

# Building and Defining the model
# define model
model = Sequential()
model.add(SimpleRNN(64, activation='relu', return_sequences=False, input_shape=(lag, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.summary()

# Fit the model - with training data
# As you are trying to use function decorator in TF 2.0, 
# please enable run function eagerly by using below line after importing TensorFlow:
import tensorflow as tf
tf.config.run_functions_eagerly(True)

# fit model
cb = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)
history = model.fit(train_X, train_y, epochs = 150, batch_size=64, verbose=1, validation_split= 0.1, 
                   callbacks=[cb])

# Summarize model accuracy and Loss
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Make prediction - with Test data
train_predict = model.predict(train_X)
test_predict = model.predict(test_X)

print('Shape of train and test predict: ', train_predict.shape, test_predict.shape)

# Model evaluation
actual_ = test_y
predicted_ = test_predict[:, 0]
len(actual_), len(predicted_)
evaluate_forecast_results(actual_, predicted_)

# Plot test data and Predicted data

plt.rc("figure", figsize=(14,8))
plt.rcParams.update({'font.size': 16})
plt.plot(actual_, label = 'Actual')
plt.plot(predicted_, label = 'Predicted')
plt.xlabel('Time in days')
plt.ylabel('Algorand price')
plt.title('Algorand price prediction using Simple RNN - Test data')
plt.legend()
plt.show()

df_train = pd.DataFrame(columns = ['Train data'])
df_train['Train data'] = train_data

df = pd.DataFrame(columns = ['Test data', 'Predicted data'])
df['Test data'] = actual_
df['Predicted data'] = predicted_

total_len = len(df_train['Train data']) + len(df['Test data'])
range(len(df_train['Train data']), total_len)
x_list = [x for x in range(len(df_train['Train data']), total_len)]
df.index = x_list

plt.rc("figure", figsize=(14,8))
plt.rcParams.update({'font.size': 16})
plt.xlabel('Time in days')
plt.ylabel('Algorand price')
plt.title('Algorand price prediction using Simple RNN')
plt.plot(df_train['Train data'])
plt.plot(df[['Test data', 'Predicted data']])
plt.legend(['Train', 'Test', 'Predictions'], loc='lower right')
plt.show()

################### 4 LSTM Long short term memory ###########################

data = dataset['price'].values
print('Shape of data: ', data.shape)

# Separate train and test data
train_length = int(len(data) * 0.8)
print('Train length: ', train_length)
train_data, test_data = data[:train_length], data[train_length:]
print('Shape of Train and Test data: ', len(train_data), len(test_data))

# Make univariate-series as Supervised Learning
# split a univariate sequence into supervised learning [Input and Output]
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

# Choose Lag value
lag = 2  # Already this is calculated  
n_features = 1

train_X, train_y = split_sequence(train_data, lag)
test_X, test_y = split_sequence(test_data, lag)

print('Shape of train_X and train_y: ', train_X.shape, train_y.shape)
print('Shape of test_X and test_y: ', test_X.shape, test_y.shape)

# Reshape train_X and test_X to 3-Dimension
train_X = train_X.reshape((train_X.shape[0], train_X.shape[1], n_features))
test_X = test_X.reshape((test_X.shape[0], test_X.shape[1], n_features))

# New shape of train_X and test_X are :-
print('Shape of train_X and train_y: ', train_X.shape, train_y.shape)
print('Shape of test_X and test_y: ', test_X.shape, test_y.shape)

# define model
model = Sequential()
model.add(LSTM(64, activation='relu', return_sequences=True, input_shape=(lag, n_features)))
model.add(LSTM(64, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.summary()

# Fit model with data
tf.config.run_functions_eagerly(True)

# fit model
cb = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)
history = model.fit(train_X, train_y, epochs = 150, batch_size = 64, verbose=1, validation_split= 0.1, 
                   callbacks = [cb])



# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Make Prediction
train_predict = model.predict(train_X)
test_predict = model.predict(test_X)

print('Shape of train and test predict: ', train_predict.shape, test_predict.shape)

# Model Evaluation
actual_lstm = test_y
predicted_lstm = test_predict[:, 0]
evaluate_forecast_results(actual_lstm, predicted_lstm)

df_train = pd.DataFrame(columns = ['Train data'])
df_train['Train data'] = train_data

df = pd.DataFrame(columns = ['Test data', 'Predicted data'])
df['Test data'] = actual_lstm
df['Predicted data'] = predicted_lstm

total_len = len(df_train['Train data']) + len(df['Test data'])
range(len(df_train['Train data']), total_len)
x_list = [x for x in range(len(df_train['Train data']), total_len)]
df.index = x_list

plt.rc("figure", figsize=(14,8))
plt.rcParams.update({'font.size': 16})
plt.xlabel('Time in days')
plt.ylabel('Algorand price')
plt.title('Algorand price prediction using LSTM')
plt.plot(df_train['Train data'])
plt.plot(df[['Test data', 'Predicted data']])
plt.legend(['Train', 'Test', 'Predictions'], loc='lower right')
plt.show()


################# 5 CNN Convolution Neural Network #########################

data = dataset['price'].values
print('Shape of data: ', data.shape)

# Separate train and test data
train_length = int(len(data) * 0.8)
print('Train length: ', train_length)
train_data, test_data = data[:train_length], data[train_length:]
print('Shape of Train and Test data: ', len(train_data), len(test_data))

# Make univariate-series as Supervised Learning
# split a univariate sequence into supervised learning [Input and Output]
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

# Choose Lag value
lag = 2  # Already this is calculated  
n_features = 1

train_X, train_y = split_sequence(train_data, lag)
test_X, test_y = split_sequence(test_data, lag)

print('Shape of train_X and train_y: ', train_X.shape, train_y.shape)
print('Shape of test_X and test_y: ', test_X.shape, test_y.shape)

# Reshape train_X and test_X to 3-Dimension
train_X = train_X.reshape((train_X.shape[0], train_X.shape[1], n_features))
test_X = test_X.reshape((test_X.shape[0], test_X.shape[1], n_features))

# New shape of train_X and test_X are :-
print('Shape of train_X and train_y: ', train_X.shape, train_y.shape)
print('Shape of test_X and test_y: ', test_X.shape, test_y.shape)

# define model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(lag, n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mse')
model.summary()

# Fit model with data
tf.config.run_functions_eagerly(True)

# fit model
cb = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)
history = model.fit(train_X, train_y, epochs = 150, batch_size = 64, verbose= 'auto', validation_split= 0.1, callbacks = [cb])

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Make Prediction
train_predict = model.predict(train_X)
test_predict = model.predict(test_X)

print('Shape of train and test predict: ', train_predict.shape, test_predict.shape)

# Model Evaluation
actual_lstm = test_y
predicted_lstm = test_predict[:, 0]
evaluate_forecast_results(actual_lstm, predicted_lstm)

df_train = pd.DataFrame(columns = ['Train data'])
df_train['Train data'] = train_data

df = pd.DataFrame(columns = ['Test data', 'Predicted data'])
df['Test data'] = actual_lstm
df['Predicted data'] = predicted_lstm

total_len = len(df_train['Train data']) + len(df['Test data'])
range(len(df_train['Train data']), total_len)
x_list = [x for x in range(len(df_train['Train data']), total_len)]
df.index = x_list

plt.rc("figure", figsize=(14,8))
plt.rcParams.update({'font.size': 16})
plt.xlabel('Time in days')
plt.ylabel('Algorand price')
plt.title('Algorand price prediction using LSTM')
plt.plot(df_train['Train data'])
plt.plot(df[['Test data', 'Predicted data']])
plt.legend(['Train', 'Test', 'Predictions'], loc='upper right')
plt.show()