import traceback
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve


# Configurable Parameters
START_DATE = "1990-01-01"
HORIZONS = [2, 5, 20, 60, 250, 1000]


class DataHandler:
    def __init__(self, start_date):
        self.START_DATE = start_date
        self.data = None
        self.predictors = None

    def fetch_sp500_data(self):
        """Fetch S&P 500 historical data."""
        try:
            sp500 = yf.Ticker("^GSPC")
            sp500 = sp500.history(period="max")
            return sp500
        except Exception as e:
            print("Error fetching data:", e)
            traceback.print_exc()  # To get detailed traceback info
            return None

    def preprocess_data(self):
        """Preprocess the data by adding target, predictors, and technical indicators."""
        try:
            self.data = self.fetch_sp500_data()
            del self.data["Dividends"]
            del self.data["Stock Splits"]
            self.data = self.data.loc[self.START_DATE:].copy()
            self.data["Tomorrow"] = self.data["Close"].shift(-1)
            self.data["Target"] = (self.data["Tomorrow"] > self.data["Close"]).astype(int)
            self.data["Close_Next"] = self.data["Close"].shift(-1)

            predictors = []

            for horizon in HORIZONS:
                rolling_averages = self.data.rolling(horizon).mean()
                ratio_column = f"Close_Ratio_{horizon}"
                self.data[ratio_column] = self.data["Close"] / rolling_averages["Close"]
                trend_column = f"Trend_{horizon}"
                self.data[trend_column] = self.data.shift(1).rolling(horizon).sum()["Target"]
                predictors += [ratio_column, trend_column]

            # Adding RSI
            feature_engineer = FeatureEngineering()
            self.data['RSI'] = feature_engineer.compute_rsi(self.data, window=14)
            predictors.append('RSI')

            # Adding MACD and Signal Line
            self.data['MACD'], self.data['Signal_Line'] = feature_engineer.compute_macd_and_signal(self.data)
            predictors.extend(['MACD', 'Signal_Line'])

            # Adding Bollinger Bands
            self.data['Upper_Bollinger_Band'], self.data['Lower_Bollinger_Band'] = feature_engineer.compute_bollinger_bands(self.data)
            predictors.extend(['Upper_Bollinger_Band', 'Lower_Bollinger_Band'])

            self.predictors = predictors
        except Exception as e:
            print("Error during preprocessing:", e)
            traceback.print_exc()

        return self.data.dropna(), self.predictors


class FeatureEngineering:
    def compute_rsi(self, data, window):
        """Compute the RSI (Relative Strength Index) for a given window."""
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)

        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def compute_macd_and_signal(self, data, short_window=12, long_window=26, signal_window=9):
        """Compute the MACD and Signal line."""
        short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
        long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
        macd = short_ema - long_ema
        signal = macd.ewm(span=signal_window, adjust=False).mean()

        return macd, signal

    def compute_bollinger_bands(self, data, window=20, num_std_dev=2):
        """Compute the Bollinger Bands."""
        rolling_mean = data['Close'].rolling(window=window).mean()
        rolling_std = data['Close'].rolling(window=window).std()

        upper_band = rolling_mean + (rolling_std * num_std_dev)
        lower_band = rolling_mean - (rolling_std * num_std_dev)

        return upper_band, lower_band

    def create_dataset(self, X, y_class, y_reg, time_steps=1):
        """Transform time series data into a supervised learning problem."""
        Xs, ys_class, ys_reg = [], [], []
        for i in range(len(X) - time_steps):
            v = X.iloc[i:(i + time_steps)].values
            Xs.append(v)
            ys_class.append(y_class.iloc[i + time_steps])
            ys_reg.append(y_reg.iloc[i + time_steps])
        return np.array(Xs), np.array(ys_class), np.array(ys_reg)


class ClassificationModel:
    def __init__(self, input_shape):
        self.model = self.build_model(input_shape)

    def build_model(self, input_shape):
        model = Sequential()
        model.add(LSTM(50, input_shape=input_shape, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
        model.add(LSTM(50, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model

    def train(self, X_train, y_train_class, X_test, y_test_class):
        self.model.fit(X_train, y_train_class, batch_size=64, epochs=100, validation_data=(X_test, y_test_class))

    def evaluate(self, X_test, y_test_class):
        y_pred = self.model.predict(X_test)
        y_pred_binary = np.round(y_pred)

        precision = precision_score(y_test_class, y_pred_binary)
        recall = recall_score(y_test_class, y_pred_binary)
        f1 = f1_score(y_test_class, y_pred_binary)
        roc_auc = roc_auc_score(y_test_class, y_pred)

        metrics = {
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'ROC AUC': roc_auc
        }

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(metrics.keys(), metrics.values())
        ax.set_ylabel('Score')
        ax.set_title('Classification Metrics')
        plt.ylim(0, 1)
        plt.show()

        # Plotting the binary predictions vs. the real binary values
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(y_test_class, label='Real Binary Values', linestyle='--')
        ax.plot(y_pred_binary, label='Predicted Binary Values', alpha=0.7)
        ax.set_title('Real vs Predicted Binary Values')
        ax.legend()
        plt.show()

        return metrics


class RegressionModel:
    def __init__(self, input_shape):
        self.model = self.build_model(input_shape)

    def build_model(self, input_shape):
        model = Sequential()
        model.add(LSTM(50, input_shape=input_shape, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
        model.add(LSTM(50, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def train(self, X_train, y_train_reg, X_test, y_test_reg):
        self.model.fit(X_train, y_train_reg, batch_size=64, epochs=100, validation_data=(X_test, y_test_reg))

    def compute_rmse(y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    def compute_mape(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def evaluate(self, X_test, y_test_reg):
        y_pred = self.model.predict(X_test)

        rmse = self.compute_rmse(y_test_reg, y_pred)
        mape = self.compute_mape(y_test_reg, y_pred)
        r2 = r2_score(y_test_reg, y_pred)

        metrics = {
            'RMSE': rmse,
            'MAPE (%)': mape,
            'R^2': r2
        }

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(metrics.keys(), metrics.values())
        ax.set_ylabel('Score')
        ax.set_title('Regression Metrics')
        plt.show()

        # Plotting the predicted vs real Closing Price
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(y_test_reg, label='Real Closing Price', linestyle='--')
        ax.plot(y_pred, label='Predicted Closing Price', alpha=0.7)
        ax.set_title('Real vs Predicted Closing Price')
        ax.legend()
        plt.show()

        return metrics


def main():
    # Initialize DataHandler and fetch, preprocess data
    data_handler = DataHandler(START_DATE)
    data = data_handler.fetch_sp500_data()
    data, predictors = data_handler.preprocess_data()

    # Feature Engineering
    fe = FeatureEngineering()
    time_steps = 10
    X, y_class, y_reg = fe.create_dataset(data[predictors], data['Target'], data['Close_Next'], time_steps)

    # Split dataset into training and testing sets
    X_train, X_test, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42)
    _, _, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)

    # Classification model
    print("Training and evaluating classification model...")
    classification_model = ClassificationModel(input_shape=(X_train.shape[1], X_train.shape[2]))
    classification_model.train(X_train, y_train_class, X_test, y_test_class)
    classification_model.evaluate(X_test, y_test_class)

    # Regression model
    print("Training and evaluating regression model...")
    regression_model = RegressionModel(input_shape=(X_train.shape[1], X_train.shape[2]))
    regression_model.train(X_train, y_train_reg, X_test, y_test_reg)
    regression_model.evaluate(X_test, y_test_reg)

if __name__ == '__main__':
    main()
