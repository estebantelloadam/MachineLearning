import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve


# Configurable Parameters
START_DATE = "1990-01-01"
HORIZONS = [2, 5, 20, 60, 250, 1000]


def fetch_sp500_data():
    """Fetch S&P 500 historical data."""
    try:
        sp500 = yf.Ticker("^GSPC")
        sp500 = sp500.history(period="max")
        return sp500
    except Exception as e:
        print("Error fetching data:", e)
        return None


def compute_rsi(data, window):
    """Compute the RSI (Relative Strength Index) for a given window."""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def compute_macd_and_signal(data, short_window=12, long_window=26, signal_window=9):
    """Compute the MACD and Signal line."""
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()

    return macd, signal


def compute_bollinger_bands(data, window=20, num_std_dev=2):
    """Compute the Bollinger Bands."""
    rolling_mean = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()

    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)

    return upper_band, lower_band


def preprocess_data(data):
    """Preprocess the data by adding target, predictors, and technical indicators."""
    del data["Dividends"]
    del data["Stock Splits"]
    data = data.loc[START_DATE:].copy()

    data["Tomorrow"] = data["Close"].shift(-1)
    data["Tomorrow"] = data["Close"].shift(-1)
    data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)
    data["Close_Next"] = data["Close"].shift(-1)

    predictors = []

    for horizon in HORIZONS:
        rolling_averages = data.rolling(horizon).mean()
        ratio_column = f"Close_Ratio_{horizon}"
        data[ratio_column] = data["Close"] / rolling_averages["Close"]
        trend_column = f"Trend_{horizon}"
        data[trend_column] = data.shift(1).rolling(horizon).sum()["Target"]
        predictors += [ratio_column, trend_column]

    # Adding RSI
    data['RSI'] = compute_rsi(data, window=14)
    predictors.append('RSI')

    # Adding MACD and Signal Line
    data['MACD'], data['Signal_Line'] = compute_macd_and_signal(data)
    predictors.extend(['MACD', 'Signal_Line'])

    # Adding Bollinger Bands
    data['Upper_Bollinger_Band'], data['Lower_Bollinger_Band'] = compute_bollinger_bands(data)
    predictors.extend(['Upper_Bollinger_Band', 'Lower_Bollinger_Band'])

    return data.dropna(), predictors


# Define a function to transform time series data into supervised learning problem
def create_dataset(X, y_class, y_reg, time_steps=1):
    Xs, ys_class, ys_reg = [], [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys_class.append(y_class.iloc[i + time_steps])
        ys_reg.append(y_reg.iloc[i + time_steps])
    return np.array(Xs), np.array(ys_class), np.array(ys_reg)


# Replace or update the main function or wherever the model training is performed
def main():
    # Load and preprocess the data
    data = fetch_sp500_data()
    data, predictors = preprocess_data(data)

    # Normalize the feature columns
    scaler = MinMaxScaler(feature_range=(0, 1))
    data[predictors] = scaler.fit_transform(data[predictors])

    # Create LSTM dataset
    time_steps = 10  # Number of time steps to look back
    X, y_class, y_reg = create_dataset(data[predictors], data['Target'], data['Close_Next'], time_steps)

    # Split the data into training and testing sets
    X_train, X_test, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42)
    X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)

    # Build the LSTM model
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        tf.keras.layers.LSTM(50, return_sequences=False),
        tf.keras.layers.Dense(25),
        tf.keras.layers.Dense(1)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    history = model.fit(X_train, y_train, batch_size=64, epochs=100, validation_data=(X_test, y_test))

    # Evaluate the model
    loss = model.evaluate(X_test, y_test)
    print(f"Model Loss: {loss}")

    # Plotting training & validation loss values
    plt.figure(figsize=(14, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Progression')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

    # Compute metrics for regression model
    rmse = np.sqrt(mean_squared_error(y_test_reg, predictions_reg))
    mape = mean_absolute_percentage_error(y_test_reg, predictions_reg)
    r2 = r2_score(y_test_reg, predictions_reg)

    print(f"RMSE for Regression Model: {rmse}")
    print(f"MAPE for Regression Model: {mape}%")
    print(f"R^2 for Regression Model: {r2}")

    # Plotting real vs predicted values for regression model
    plt.figure(figsize=(14, 5))
    plt.plot(y_test_reg, label='True values')
    plt.plot(predictions_reg, label='Predicted values')
    plt.title('Stock Closing Price Prediction (Regression Model)')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

    # Making predictions
    predictions = model.predict(X_test)

    # Plotting real vs predicted values
    plt.figure(figsize=(14, 5))
    plt.plot(y_test, label='True values')
    plt.plot(predictions, label='Predicted values')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Normalized Stock Price')
    plt.legend()
    plt.show()


# If the script is being run, call the main function
if __name__ == "__main__":
    main()
