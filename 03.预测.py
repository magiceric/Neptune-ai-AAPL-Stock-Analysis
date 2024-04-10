from config import tushare_api_key
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Input, Dense
import tushare as ts
import os
from datetime import datetime, timedelta
import sqlalchemy
from sqlalchemy import Table, Column, Integer, Float, String, MetaData, Date

# Setting TensorFlow to use GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Database connection and table creation
database_connection_string = 'mysql+pymysql://stock:Abcd1234!!@192.168.3.7:3306/aiStock'
engine = sqlalchemy.create_engine(database_connection_string)
connection = engine.connect()
metadata = MetaData()
predictionHistory = Table('predictionHistory', metadata,
                          Column('id', Integer, primary_key=True),
                          Column('predDate', Date),
                          Column('stockCode', String(10)),
                          Column('lastDate', Date),
                          Column('lastClose', Float),
                          Column('predT1', Float),
                          Column('predT2', Float),
                          )
metadata.create_all(engine)  # Creates the table if it doesn't exist

# Fetching top 50 active stocks and writing to 'stock.list'
ts.set_token(tushare_api_key)
pro = ts.pro_api()

def fetch_and_predict(ts_code):
    ts.set_token(tushare_api_key)  # Use tushare token from config
    pro = ts.pro_api()

    # Fetching data
    df = pro.daily(ts_code=ts_code, start_date='19900101', end_date='20241231')

    # Preparing DataFrame
    df_prepared = pd.DataFrame({
        "timestamp": pd.to_datetime(df["trade_date"], format='%Y%m%d'),
        "close": df["close"]
    })

    # Reverse DataFrame to align dates correctly
    df_prepared = df_prepared.iloc[::-1]

    # Splitting data into training and testing sets with a 9:1 ratio
    train_size = int(0.9 * len(df_prepared))
    train, test = df_prepared[:train_size], df_prepared[train_size:]

    # Scaling the 'close' prices
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train[['close']])
    test_scaled = scaler.transform(test[['close']])

    # Function to create sequences for LSTM
    def create_sequences(data, window_size):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:(i + window_size)])
            y.append(data[i + window_size])
        return np.array(X), np.array(y)

    window_size = 50
    X_train, y_train = create_sequences(train_scaled, window_size)
    X_test, y_test = create_sequences(test_scaled, window_size)

    # Reshaping input to be [samples, time steps, features]
    if len(X_train) > 0 and len(X_test) > 0:  # Check if X_train and X_test are not empty
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    else:
        print("Error: Not enough data to reshape for LSTM input.")
        return {}

    # Building the LSTM model
    def build_lstm_model(input_shape):
        model = tf.keras.Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            LSTM(50),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    model = build_lstm_model((X_train.shape[1], 1))

    # Training the model
    model.fit(X_train, y_train, epochs=15, batch_size=20, verbose=1, validation_split=0.1, shuffle=False)

    # Predicting
    predicted_test = model.predict(X_test)
    predicted_test = scaler.inverse_transform(predicted_test)

    # Adding predictions to the test data
    test['Predictions_lstm'] = np.concatenate((np.full((window_size), np.nan), predicted_test.ravel()))

    # Predicting T+1 and T+2
    last_sequence = test_scaled[-window_size:].reshape((1, window_size, 1))
    future_predictions = []
    future_dates = [test.iloc[-1]['timestamp'] + timedelta(days=i) for i in range(1, 3)]
    for _ in range(2):  # Predict next two values
        last_prediction_scaled = model.predict(last_sequence)

        future_predictions.append(scaler.inverse_transform(last_prediction_scaled)[0][0])
        last_sequence = np.append(last_sequence[:, 1:, :], last_prediction_scaled).reshape((1, window_size, 1))

    # Plotting
    plt.figure(figsize=(20, 10))
    plt.plot(train['timestamp'], train['close'], label='Train Closing Price')
    plt.plot(test['timestamp'], test['close'], label='Test Closing Price')
    plt.plot(test['timestamp'], test['Predictions_lstm'], label='Predicted Closing Price', alpha=0.7)
    legend_labels = ['Train Closing Price', 'Test Closing Price', 'Predicted Closing Price']
    label_offsets = [0.01, 0.09]  # Adjusted offsets for T+1 and T+2 labels for better separation
    for i, (date, price) in enumerate(zip(future_dates, future_predictions)):
        plt.scatter(date, price, color='red')
        # Adjusting label positions for clarity with increased separation
        plt.text(date, price * (1 + label_offsets[i]), f"T+{i+1}: {price:.2f}", fontsize=9)
        legend_labels.append(f"T+{i+1} Predicted Price: {price:.2f}")
    # Adding last close price to legend
    last_close_price = test.iloc[-1]['close'] if 'close' in test.iloc[-1] else None
    legend_labels.append(f"Last Close Price: {last_close_price:.2f}" if last_close_price is not None else "Last Close Price: Not available")  # Fixed to show last close price correctly or indicate not available
    plt.title('LSTM Model Predictions vs Actual Closing Prices')
    plt.xlabel('Date')
    plt.ylabel('Stock Price ($)')
    plt.legend(legend_labels, loc='upper left')
    # Save plot to 'plots' directory under a subdirectory for the current date
    current_date = datetime.now().strftime('%Y%m%d')
    plots_dir = os.path.join('plots', current_date)
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    plot_filename = f"{test.iloc[-1]['timestamp'].strftime('%Y%m%d')}_{ts_code}.png"
    plt.savefig(os.path.join(plots_dir, plot_filename))
    plt.close()

    # Inserting prediction results into the database
    insert_stmt = predictionHistory.insert().values(
        predDate=datetime.now().date(),
        stockCode=ts_code,
        lastDate=test.iloc[-1]['timestamp'].date() if 'timestamp' in test.iloc[-1] else None,
        lastClose=last_close_price,
        predT1=future_predictions[0],
        predT2=future_predictions[1]
    )
    try:
        connection.execute(insert_stmt)
        connection.commit()
        print(f"Successfully inserted prediction results for {ts_code} into the database.")
    except Exception as e:
        print(f"Failed to insert prediction results for {ts_code} into the database. Error: {e}")

    # Ensure the result dictionary always contains the 'last_date' key
    result = {
        "last_date": test.iloc[-1]['timestamp'].strftime('%Y%m%d') if 'timestamp' in test.iloc[-1] else None,
        "last_close_price": last_close_price,
        "T+1_predicted_close_price": future_predictions[0] if len(future_predictions) > 0 else None,
        "T+2_predicted_close_price": future_predictions[1] if len(future_predictions) > 1 else None
    }

    return result

# Reading stock codes from 'stock.list' and running predictions for each
with open('stock.list', 'r') as file:
    stock_codes = [line.strip() for line in file if line.strip()]
for code in stock_codes:
    result = fetch_and_predict(code)
    print(f"Processing {code}")
    # Check if 'last_date' key exists in the result before printing
    if 'last_date' in result:
        print(f"Last Date: {result['last_date']}")
    else:
        print("Last Date: Not available")
    if 'last_close_price' in result and result['last_close_price'] is not None:
        print(f"Last Close Price: {result['last_close_price']}")
    else:
        print("Last Close Price: Not available")
    # Check if 'T+1_predicted_close_price' and 'T+2_predicted_close_price' keys exist in the result before printing
    if 'T+1_predicted_close_price' in result and result['T+1_predicted_close_price'] is not None:
        print(f"T+1 Predicted Close Price: {result['T+1_predicted_close_price']}")
    else:
        print("T+1 Predicted Close Price: Not available")
    if 'T+2_predicted_close_price' in result and result['T+2_predicted_close_price'] is not None:
        print(f"T+2 Predicted Close Price: {result['T+2_predicted_close_price']}")
    else:
        print("T+2 Predicted Close Price: Not available")
