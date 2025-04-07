import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from datetime import datetime, timedelta
from forex_python.converter import CurrencyRates

# Step 1: Download stock data from Yahoo Finance
def download_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Step 2: Preprocess the data (Scaling and reshaping)
def preprocess_data(data):
    close_prices = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))  # Scale between 0 and 1
    scaled_data = scaler.fit_transform(close_prices)
    return scaled_data, scaler

# Step 3: Create the dataset for training the LSTM model
def create_datasets(scaled_data, time_step=60):
    x_train, y_train = [], []
    for i in range(time_step, len(scaled_data)):
        x_train.append(scaled_data[i-time_step:i, 0])
        y_train.append(scaled_data[i, 0])
    return np.array(x_train), np.array(y_train)

# Step 4: Build the LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Step 5: Predict the stock price for tomorrow and day after tomorrow
def predict_prices(model, scaler, last_60_days):
    last_60_days_scaled = scaler.transform(last_60_days.values.reshape(-1, 1))
    
    # Predict tomorrow's price
    X_test = np.array([last_60_days_scaled])
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_tomorrow = model.predict(X_test)
    predicted_tomorrow_price = scaler.inverse_transform(predicted_tomorrow)[0, 0]

    # Append tomorrow's prediction to last 59 days to predict day after tomorrow
    new_input = np.append(last_60_days_scaled[1:], predicted_tomorrow)
    new_input = np.reshape(new_input, (1, new_input.shape[0], 1))
    
    predicted_day_after_tomorrow = model.predict(new_input)
    predicted_day_after_tomorrow_price = scaler.inverse_transform(predicted_day_after_tomorrow)[0, 0]

    return predicted_tomorrow_price, predicted_day_after_tomorrow_price

# Step 6: Fetch USD to INR conversion rate
def fetch_usd_to_inr():
    currency_converter = CurrencyRates()
    try:
        return currency_converter.get_rate('USD', 'INR')  # Removed timeout
    except Exception as e:
        print(f"Error fetching exchange rate: {e}")
        return 82.0  # Example default value in case of error

# Step 7: Main function
def main():
    ticker = "GOOG"  # Example: Google Inc. (You can change the stock symbol)
    start_date = "2015-01-01"
    end_date = datetime.today().strftime('%Y-%m-%d')
    
    # Download and preprocess stock data
    data = download_stock_data(ticker, start_date, end_date)
    scaled_data, scaler = preprocess_data(data)
    
    # Create training datasets
    time_step = 60  # Use last 60 days to predict the next day
    x_train, y_train = create_datasets(scaled_data, time_step)
    
    # Reshape the data for LSTM
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    # Build and train the LSTM model
    model = build_lstm_model((x_train.shape[1], 1))
    model.fit(x_train, y_train, batch_size=1, epochs=1)  # Train with 1 epoch for demonstration

    # Predict tomorrow and day after tomorrow's stock price
    last_60_days = data['Close'][-60:]  # Get the last 60 days of closing prices
    predicted_tomorrow, predicted_day_after_tomorrow = predict_prices(model, scaler, last_60_days)
    
    # Fetch the USD to INR conversion rate
    usd_to_inr_rate = fetch_usd_to_inr()
    
    # Convert predictions to INR
    predicted_tomorrow_inr = predicted_tomorrow * usd_to_inr_rate
    predicted_day_after_tomorrow_inr = predicted_day_after_tomorrow * usd_to_inr_rate

    # Print the predictions
    current_price = data['Close'].iloc[-1]  # Get the last closing price (scalar)
    tomorrow = datetime.today() + timedelta(days=1)
    day_after_tomorrow = datetime.today() + timedelta(days=2)
    
    print(f"Predicted price for {ticker} on {tomorrow.strftime('%Y-%m-%d')} (INR): ₹{predicted_tomorrow_inr:.2f}")
    print(f"Predicted price for {ticker} on {day_after_tomorrow.strftime('%Y-%m-%d')} (INR): ₹{predicted_day_after_tomorrow_inr:.2f}")
    print(f"Today's actual closing price (USD): ${current_price.iloc[-1]:.2f}")
    print(f"USD to INR Conversion Rate: {usd_to_inr_rate:.2f}")
    
    if predicted_day_after_tomorrow > predicted_tomorrow:
        print("The stock price is predicted to go up the day after tomorrow.")
    else:
        print("The stock price is predicted to go down the day after tomorrow.")

if _name_ == "_main_":
    main()
