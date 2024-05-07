import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, learning_curve


# --- Data Handling Section ---
def fetch_sp500_data():
    """Fetch S&P 500 historical data."""
    try:
        sp500 = yf.Ticker("^GSPC")
        sp500 = sp500.history(period="max")
        return sp500
    except Exception as e:
        print("Error fetching data:", e)
        return None


# --- Feature Engineering Section ---
def compute_rsi(data, window=14):
    """Compute the RSI for a dataset."""
    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_macd(data, short_window=12, long_window=26, signal_window=9):
    """Compute the MACD and Signal Line for a dataset."""
    short_ema = data["Close"].ewm(span=short_window, adjust=False).mean()
    long_ema = data["Close"].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal


def compute_bollinger_bands(data, window=20, num_std=2):
    """Compute the Bollinger Bands for a dataset."""
    rolling_mean = data["Close"].rolling(window=window).mean()
    rolling_std = data["Close"].rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band


def add_technical_indicators(data):
    """Add RSI, MACD, Signal Line, and Bollinger Bands to a dataset."""
    data["RSI"] = compute_rsi(data)
    data["MACD"], data["Signal_Line"] = compute_macd(data)
    data["Upper_Bollinger"], data["Lower_Bollinger"] = compute_bollinger_bands(data)
    return data


# --- Model Evaluation Section ---
def plot_learning_curve(model, X, y):
    """Plot learning curve for a given model and dataset."""
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, n_jobs=-1,
                                                            train_sizes=np.linspace(0.1, 1.0, 10))
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.xlabel("Number of Training Samples")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.show()


# --- Main Execution ---
def main():
    sp500 = fetch_sp500_data()
    if sp500 is not None:
        sp500 = add_technical_indicators(sp500).dropna()
        X = sp500[["RSI", "MACD", "Signal_Line", "Upper_Bollinger", "Lower_Bollinger", "Volume"]]
        y = (sp500["Close"].pct_change().shift(-1) > 0).astype(int)

        # Hyperparameter Tuning using Random Search
        param_dist = {
            "n_estimators": [50, 100, 200],
            "max_features": ["sqrt", "sqrt"],
            "max_depth": [None, 10, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "bootstrap": [True, False]
        }
        model = RandomForestClassifier(random_state=1)
        random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=30,
                                           cv=5, verbose=1, n_jobs=-1, random_state=42)
        random_search.fit(X, y)
        print("Best hyperparameters:", random_search.best_params_)

        # Plot learning curve for the best model
        best_model = random_search.best_estimator_
        plot_learning_curve(best_model, X, y)


if __name__ == "__main__":
    main()
