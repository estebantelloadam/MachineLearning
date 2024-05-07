import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Configurable Parameters
START_DATE = "1990-01-01"
HORIZONS = [2, 5, 20, 60, 250, 1000]

sp500 = yf.Ticker("^GSPC")

# Query all data history
sp500 = sp500.history(period="max")

# Visualize the data
sp500.plot.line(y="Close", use_index=True)
plt.show()

# Clean the data
del sp500["Dividends"]
del sp500["Stock Splits"]
sp500 = sp500.loc["1990-01-01":].copy()

# Setup target
sp500["Tomorrow"] = sp500["Close"].shift(-1)
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)

# Add more predictors to the model
new_predictors = []
for horizon in HORIZONS:
    rolling_averages = sp500.rolling(horizon).mean()

    ratio_column = f"Close_Ratio_{horizon}"
    sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"]

    trend_column = f"Trend_{horizon}"
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]

    new_predictors += [ratio_column, trend_column]

sp500 = sp500.dropna()

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
sp500['Close'] = scaler.fit_transform(np.array(sp500['Close']).reshape(-1, 1))

# Transform time series data into supervised learning problem
look_back = 10  # Number of previous time steps to use as input variables to predict the next time step
X, y = [], []
for i in range(len(sp500)-look_back-1):
    X.append(sp500['Close'][i:(i+look_back)])
    y.append(sp500['Close'][i+look_back])
X, y = np.array(X), np.array(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Build the LSTM model
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    tf.keras.layers.LSTM(50, return_sequences=False),
    tf.keras.layers.Dense(25),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, batch_size=64, epochs=100, validation_data=(X_test, y_test))

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f"Model Loss: {loss}")
