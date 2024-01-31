import streamlit as st
import numpy as np
import pandas as pd
import datetime 
import yfinance as yf
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# set title
st.title('Stock Analysis & Prediction using LSTM')
st.write("Created by Michael Arthur")
st.write("This project aims to educate individuals on the stock market and deep learning, specifically focusing on LSTM networks. By providing insights into financial markets and practical applications of LSTM, the project seeks to enhance understanding at the intersection of data analytics, machine learning, and stock market dynamics.")
st.link_button("Source Code", 'https://github.com/MichaelArthur224/Stock-Prediction-Analysis-App', help=None, type="secondary", disabled=False, use_container_width=False)
# gather user info
ticker_list = pd.read_csv('https://raw.githubusercontent.com/MichaelArthur224/Stock-Prediction-Analysis-App/main/all_tickers.txt')
tickerSymbol = st.selectbox('Select a Ticker', ticker_list)
start_date = st.date_input("Start date", datetime.date(2020, 1, 1))
end_date = st.date_input("End date", datetime.date(2024, 1, 1))

# prediction button
def click_button():
    st.session_state.clicked = True
    # prediction button
def button():
    st.session_state.clicked = True
#st.sidebar.button('LSTM Prediction', on_click=click_button)
# get stock info
ticker_info = yf.Ticker(tickerSymbol)
# history
ticker_history = ticker_info.history(period='max', start=start_date, end=end_date)
# company name
string_name = ticker_info.info['longName']
st.header(string_name)
st.text(tickerSymbol)
# company about
string_summary = ticker_info.info['longBusinessSummary']
st.write(string_summary)
st.write(ticker_history)
# close and open graph
st.header("Close & Open")
st.line_chart(data=ticker_history, x=None, y=['Close', 'Open'], width=0, height=0, use_container_width=True)
st.write("Close and open refer to specific prices associated with a stock during a particular time period")
# volume
st.header("Volume")
st.scatter_chart(data=ticker_history, x=None, y='Volume', color=None, size=None, width=0, height=0, use_container_width=True)
st.write("Volume refers to the total number of shares of a particular security that are traded during a specific period of time, typically within a trading day. It is a measure of market activity and liquidity.")
# high and low graph
st.header("High & Low")
st.line_chart(data=ticker_history, x=None, y=['High', 'Low'], width=0, height=0, use_container_width=True)
st.write("High and low refer to specific price levels associated with a stock during a particular time period:")

# simple Returns Calculation
open_prices = ticker_history['Open']
close_prices = ticker_history['Close']
def calculate_simple_return(open_price, close_price):
    return ((close_price - open_price) / open_price) * 100
ticker_history['simple_return'] = calculate_simple_return(ticker_history['Open'], ticker_history['Close'])
st.header('Simple Returns Over Time')
st.line_chart(ticker_history['simple_return'])
st.write("Simple return is a financial metric that measures the percentage change in the value of an investment over a specific period.")

st.header("What is LSTM?")
st.write("Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN) architecture designed to address the vanishing gradient problem that occurs in traditional RNNs. LSTMs are particularly effective for modeling sequences of data and are widely used in tasks such as time series prediction, natural language processing, speech recognition, and more.")
# LSTM prediction model
def prediction():
  # gather 'close' values
  data = ticker_history.filter(['Close'])
  dataset = data.values
  training_data_len = int(np.ceil( len(dataset) * .8))
  # scale data for lstm
  scaler = MinMaxScaler(feature_range=(0,1))
  scaled_data = scaler.fit_transform(dataset)
  train_data = scaled_data[0:int(training_data_len), :]
  # split x train and y train
  X_train = []
  y_train = []
  # 60 prediction days
  for i in range(60, len(train_data)):
    X_train.append(train_data[i - 60 :i, 0])
    y_train.append(train_data[i, 0])
  # reshape for lstm
  X_train, y_train = np.array(X_train), np.array(y_train)
  X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
  input_shape = (X_train.shape[1], 1)
  # model
  model = Sequential()
  model.add(LSTM(units = 50, return_sequences = True, input_shape = input_shape))
  model.add(LSTM(units = 50, return_sequences = True))
  model.add(LSTM(units = 50, return_sequences = False))
  # Output layer
  model.add(Dense(units = 1, activation = 'linear'))
  # run model
  model.compile(loss = 'mean_squared_error', optimizer = 'adam')
  model.fit(X_train, y_train, epochs=10)
  # testing
  test_data = scaled_data[training_data_len - 60: , :]
  # split x test and y test
  X_test = []
  y_test = dataset[training_data_len:, :]
  # 60 prediction days
  for i in range(60, len(test_data)):
    X_test.append(test_data[i - 60 :i, 0])
  # reshape for results
  X_test = np.array(X_test)
  X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
  # predictions
  preds = model.predict(X_test)
  preds = scaler.inverse_transform(preds)
  train = data[:training_data_len]
  valid = data[training_data_len:]
  valid['Predictions'] = preds
  # prediction graph
  st.line_chart(valid[['Close', 'Predictions']])
  st.write("The  time series prediction model using a Long Short-Term Memory (LSTM) neural network. LSTMs are a type of recurrent neural network (RNN) designed to capture long-term dependencies in sequential data. In this case, the model is trained on historical 'Close' prices of a financial instrument, which are first scaled to a range between 0 and 1 using the Min-Max scaling technique. The LSTM architecture comprises three layers of LSTM units, followed by a Dense layer for the output. The model is compiled with the 'adam' optimizer and uses the mean squared error as the loss function. During training, the LSTM network learns patterns and relationships in the training data, enabling it to make predictions for future 'Close' prices. After training, the model is tested on a separate portion of the dataset, and the predictions are inverse-transformed to the original scale for comparison with the actual closing prices. The resulting predictions are then visualized alongside the true closing prices using Streamlit, providing an interactive and user-friendly platform for exploring the model's forecasting performance.")

# run button
st.header("Predicting Closing Price with LSTM")
if st.button("Click to start LSTM Prediction", on_click=click_button):
    prediction()


























