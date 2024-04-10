from config import api_key
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Input, Dense, Dropout, Activation
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical as np_utils

stockCode = '603936.SH'
lastDate = '20240322'
import tushare as ts

ts.set_token('94cd405b2c1adff88930d17a3a3d6e5c0c53f4dc4945aeb264b2be3a')  # Set your tushare token
pro = ts.pro_api()

df = pro.daily(ts_code=stockCode, start_date='19900101', end_date=lastDate)
print(df)

# Adjusting the DataFrame to match the target data structure with modified keys
last_refreshed_date = df.iloc[0]['trade_date'] if not df.empty else 'N/A'  # Assuming the first row is the latest
data = {
    "Meta Data": {
        "1. Information": "Daily Prices (open, high, low, close) and Volumes",
        "2. Symbol": stockCode,
        "3. Last Refreshed": last_refreshed_date,
        "4. Output Size": "Full size",
        "5. Time Zone": "Asia/Shanghai"
    },
    "Time Series (Daily)": {date: {"1. open": str(row['open']), "2. high": str(row['high']), "3. low": str(row['low']), "4. close": str(row['close']), "5. volume": str(row['vol'])} for date, row in df.set_index('trade_date').iterrows()}
}

print(data)
sz000001_df_json = pd.DataFrame.from_dict(data, orient='index')
sz000001_df_json
df_tushare = pro.daily(ts_code=stockCode, start_date='19900101', end_date=lastDate)
historical_price_csv_df = pd.DataFrame({
    "timestamp": df_tushare["trade_date"],
    "open": df_tushare["open"],
    "high": df_tushare["high"],
    "low": df_tushare["low"],
    "close": df_tushare["close"],
    "volume": df_tushare["vol"]
})
print(historical_price_csv_df)

len(historical_price_csv_df.close)
df_copy = historical_price_csv_df.copy()
date_close_df = df_copy.filter(['timestamp','open', 'high', 'low', 'close', 'volume'], axis=1).iloc[::-1]
date_close_df
date_close_df.tail(5)
stockprices = date_close_df
#### Train-Test split for time-series ####
test_ratio = 0.2
training_ratio = 1 - test_ratio

train_size = int(training_ratio * len(stockprices))
test_size = int(test_ratio * len(stockprices))
print("train_size: " + str(train_size))
print("test_size: " + str(test_size))

train = stockprices[:train_size]
test = stockprices[train_size:]
## Split the time-series data into training seq X and output value Y
def extract_seqX_outcomeY(data, N, offset):
    """
    Split time-series into training sequence X and outcome value Y
    Args:
        data - dataset 
        N - window size, e.g., 50 for 50 days of historical stock prices
        offset - position to start the split
    """
    X, y = [], []
    
    for i in range(offset, len(data)):
        X.append(data[i-N:i])
        y.append(data[i][-2])  # Assuming the 'close' price is the last column but one
    
    return np.array(X), np.array(y)
#### Calculate the metrics RMSE and MAPE ####
def calculate_rmse(y_true, y_pred):
    """
    Calculate the Root Mean Squared Error (RMSE)  
    """
    rmse = np.sqrt(np.mean((y_true-y_pred)**2))                   
    return rmse

def calculate_mape(y_true, y_pred): 
    """
    Calculate the Mean Absolute Percentage Error (MAPE) %
    """
    y_pred, y_true = np.array(y_pred), np.array(y_true)    
    mape = np.mean(np.abs((y_true-y_pred) / y_true))*100    
    return mape
def plot_stock_trend(var, cur_title, stockprices=stockprices, logNeptune=True, logmodelName='Simple MA'):
    ax = stockprices[['close', var,'200day']].plot(figsize=(20, 10))
    plt.grid(False)
    plt.title(cur_title)
    plt.axis('tight')
    plt.ylabel('Stock Price ($)')
window_size = 50

import neptune

window_var = str(window_size) + 'day'
layer_units, optimizer = 50, 'adam' 
cur_epochs = 50
cur_batch_size = 20
    
cur_LSTM_pars = {'units': layer_units, 
                 'optimizer': optimizer, 
                 'batch_size': cur_batch_size, 
                 'epochs': cur_epochs
                 }
# scale our dataset
scaler = StandardScaler()
scaled_data = scaler.fit_transform(stockprices[['open', 'high', 'low', 'close', 'volume']])
scaled_data_train = scaled_data[:train.shape[0]]
    
# We use past 50 days’ stock prices for our training to predict the 51th day's closing price.
X_train, y_train = extract_seqX_outcomeY(scaled_data_train, window_size, window_size)
### Build a LSTM model and log model summary to Neptune ###    
def Run_LSTM(X_train, layer_units=50):     
    inp = Input(shape=(X_train.shape[1], X_train.shape[2]))
    
    x = LSTM(units=layer_units, return_sequences=True)(inp)
    x = LSTM(units=layer_units)(x)
    out = Dense(1, activation='linear')(x)
    model = tf.keras.Model(inp, out)  # Corrected keras.Model to tf.keras.Model
    
    # Compile the LSTM neural net
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    return model   

model = Run_LSTM(X_train, layer_units=layer_units)

history = model.fit(X_train, y_train, epochs=cur_epochs, batch_size=cur_batch_size, 
                    verbose=1, validation_split=0.1, shuffle=True)
# predict stock prices using past window_size stock prices
def preprocess_testdat(data=stockprices, scaler=scaler, window_size=window_size, test=test):    
    raw = data[['open', 'high', 'low', 'close', 'volume']].iloc[len(data) - len(test) - window_size:].values
    # To avoid the warning, ensure the data passed to scaler.transform() has the same structure as the data used in scaler.fit()
    # Specifically, ensure it's a DataFrame with column names matching those used in fitting.
    raw_df = pd.DataFrame(raw, columns=['open', 'high', 'low', 'close', 'volume'])
    raw_scaled = scaler.transform(raw_df)
    
    X_test = []
    for i in range(window_size, raw_scaled.shape[0]):
        X_test.append(raw_scaled[i-window_size:i])
        
    X_test = np.array(X_test)
    
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))
    return X_test

X_test = preprocess_testdat()

predicted_price_ = model.predict(X_test)
predicted_price = scaler.inverse_transform(predicted_price_)

# Plot predicted price vs actual closing price 
test['Predictions_lstm'] = np.concatenate((np.zeros((predicted_price.shape[0], 4)), predicted_price), axis=1)[:, -1]

# 打印最后1天的日期和最后2天的实际值和预测值
last_day = test.iloc[-1]['timestamp']
last_day_actual_close = test.iloc[-1]['close']
last_day_predicted_close = test.iloc[-1]['Predictions_lstm']
second_last_day_actual_close = test.iloc[-2]['close']
second_last_day_predicted_close = test.iloc[-2]['Predictions_lstm']

print(f"最后1天的日期: {last_day}")
print(f"最后1天的实际收盘价: {last_day_actual_close}, 预测收盘价: {last_day_predicted_close}")
print(f"倒数第二天的实际收盘价: {second_last_day_actual_close}, 预测收盘价: {second_last_day_predicted_close}")

# 预测后面两个交易日的收盘价
# Instead of calculating future dates, use "T+1" and "T+2"
future_dates = ["T+1", "T+2"]
future_raw = np.array([predicted_price[-1]]).reshape(-1,1)  # Assuming the last predicted price for future prediction
# Use a DataFrame for future_raw to match the structure expected by scaler
future_raw_df = pd.DataFrame(future_raw, columns=['close'])
future_raw_scaled = scaler.transform(future_raw_df)

# Corrected future prediction preprocessing to match the expected input shape for the model
future_X_test = np.repeat(future_raw_scaled.T, window_size, axis=0).reshape(1, window_size, 1)
future_predicted_price_ = model.predict(future_X_test)
future_predicted_price = scaler.inverse_transform(future_predicted_price_)

# 预测“T+1”的收盘价
print(f"{future_dates[0]}预测收盘价: {future_predicted_price[0][0]}")

# 使用“T+1”的预测值来预测“T+2”
future_raw_next_day = np.array([future_predicted_price[-1]]).reshape(-1,1)
# Again, use a DataFrame for future_raw_next_day
future_raw_next_day_df = pd.DataFrame(future_raw_next_day, columns=['close'])
future_raw_next_day_scaled = scaler.transform(future_raw_next_day_df)
future_X_test_next_day = np.repeat(future_raw_next_day_scaled.T, window_size, axis=0).reshape(1, window_size, 1)
future_predicted_price_next_day_ = model.predict(future_X_test_next_day)
future_predicted_price_next_day = scaler.inverse_transform(future_predicted_price_next_day_)

# 预测“T+2”的收盘价
print(f"{future_dates[1]}预测收盘价: {future_predicted_price_next_day[0][0]}")

### Plot prediction and true trends and log to Neptune         
def plot_stock_trend_lstm(train, test, logNeptune=True):        
    fig = plt.figure(figsize = (20,10))
    plt.plot(train['timestamp'], train['close'], label = 'Train Closing Price')
    plt.plot(test['timestamp'], test['close'], label = 'Test Closing Price')
    plt.plot(test['timestamp'], test['Predictions_lstm'], label = 'Predicted Closing Price')
    plt.title('LSTM Model')
    plt.xlabel('Date')
    plt.ylabel('Stock Price ($)')
    plt.legend(loc="upper left")
    
plot_stock_trend_lstm(train, test)

