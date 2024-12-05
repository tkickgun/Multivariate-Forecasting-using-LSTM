## Stock Price Forecasting Using LSTM
#### Overview
This repository implements a time series forecasting model using Long Short-Term Memory (LSTM) networks to predict the opening price of a stock based on its historical data. The data is fetched using Yahoo Finance, preprocessed, and fed into an LSTM model trained on a 14-day moving window. The prediction is limited to a single future timestamp due to the unavailability of future exogenous variable values.

#### Features
Data Source: Stock price data from Yahoo Finance (yfinance library).
Preprocessing: Scales data using StandardScaler for improved model performance.
Model Architecture:
Two stacked LSTM layers.
Dropout for regularization.
Fully connected Dense layer for final prediction.
Training: Trains on 90% of the data with validation split and batch size for optimization.
Prediction: Predicts the next day's stock opening price based on the previous 14 days' data.

#### Usage
1. Data Preparation
The dataset is fetched from Yahoo Finance for the stock ticker GE from 1970-01-02 to 2020-11-01.
Columns such as Open, High, Low, Close, and Adj Close are standardized.
2. Model Training
The input (scaled_X) consists of 14 rows (days) and 5 columns (features).
The target (scaled_Y) is the next day's Open price.

Train the LSTM model with:
model.fit(scaled_X, scaled_Y, validation_split=0.1, epochs=10, batch_size=16)

#### Forecasting
The model can forecast only the next day's stock opening price:
future_pred = model.predict(test_obs_reshaped)
future_pred_inverse = scaler.inverse_transform(future_pred)
print(f"The predicted opening price is {future_pred_inverse[0]:.2f}")

#### Limitations
Single Timestamp Prediction: The model forecasts only the next day’s stock opening price as future values of exogenous variables (e.g., Close, High, Low, Adj Close) are unavailable.
No Test/Validation Split: For demonstration purposes, all data is used for training.

#### Key Functions
Model Architecture:
Stacked LSTM with 64 and 32 units.
Dropout for regularization.
Dense layer for single-value output.
Performance Metrics:
Root Mean Squared Error (RMSE) for evaluation.

#### Results
Predicted opening price for October 31, 2020: 37.897
