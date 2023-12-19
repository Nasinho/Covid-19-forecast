#Index 1: Imported Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

#Index 2: Imported Dataset
data_sh = pd.read_csv('C:\\...\\COVID-19_SH_Dataset.csv')

#Index 3: Defining the top of the dataset
data_sh.head()

#Index 4: Date format and setting the index
data_sh['Date'] = pd.to_datetime(data_sh.Date,format='%Y-%m-%d')
data_sh.index = data_sh['Date']

#Index 5: Transformation of the dataset and sorting
data = data_sh.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(data_sh)),columns=['Date', 'Cases'])
for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Cases'][i] = data['Cases'][i]

#Index 6: Setting the index and removing column titles
new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)

#Index 7: Loading new dataset and defining training and validation parts
dataset = new_data.values
train = dataset[0:500,:]
valid = dataset[500:,:]

#Index 8: Data normalization to percental change
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

#Index 9: Short-term memory definition
x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

#Index 10: Reshaping the dataset to view dimensions
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

#Index 11: Density and unit prediction defined
model = Sequential()
model.add(LSTM(units=500, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=500))
model.add(Dense(1))

#Index 12: Compiling the model with the Adam optimizer
model.compile(loss='mean_squared_error', optimizer='adam')

#Index 13: Fitting the model
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

#Index 14: Predicting cases
inputs = new_data[len(new_data) - len(valid) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)
X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
predicted_cases = model.predict(X_test)
predicted_cases = scaler.inverse_transform(predicted_cases)

#Index 15: Plotting the graph
plt.figure(figsize=(16,8))
train = new_data[:500]
valid = new_data[500:]
valid['Predictions'] = predicted_cases
plt.plot(train['Cases'], label='Case History')
plt.plot(valid['Cases'], label='Actual Cases')
plt.plot(valid['Predictions'], label='Predicted Cases')
plt.title('COVID-19 cases in Schleswig Holstein since 02/2020', fontsize=16)
figure = plt.gcf()
figure.canvas.set_window_title('LSTM: Covid Prediction')
plt.legend()
plt.show()

#Source: Modeled after https://www.analyticsvidhya.com/blog/2018/10/predicting-stock-price-machine-learningnd-deep-learning-techniques-python/?#