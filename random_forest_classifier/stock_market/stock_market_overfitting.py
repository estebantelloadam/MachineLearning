import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score

# Configurable Parameters
START_DATE = "1990-01-01"
HORIZONS = [2, 5, 60, 250, 1000]
BACKTEST_START = 250*10
BACKTEST_STEP = 250

def fetch_sp500_data():
    try:
        sp500 = yf.Ticker("^GSPC")
        sp500 = sp500.history(period="max")
        return sp500
    except Exception as e:
        print("Error fetching data:", e)
        return None

sp500 = fetch_sp500_data()

if sp500 is not None:
    try:
        # Initial visualization of closing prices
        sp500.plot.line(y="Close", use_index=True, title="S&P 500 Closing Prices")
        plt.show()

        # Data preparation
        del sp500["Dividends"]
        del sp500["Stock Splits"]
        sp500 = sp500.loc[START_DATE:].copy()

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

        # Splitting data into training and test set
        train_data, test_data = train_test_split(sp500, test_size=0.2, random_state=1)

        # Improved model with regularization parameters
        model = RandomForestClassifier(n_estimators=200, min_samples_split=50, max_depth=10, min_samples_leaf=5,
                                       max_features='sqrt', random_state=1)

        # Train and evaluate the model using cross-validation
        cross_val_scores = cross_val_score(model, train_data[new_predictors], train_data["Target"], cv=5)
        print("Cross-validation scores:", cross_val_scores)
        print("Average CV score:", cross_val_scores.mean())

        # Training the model on the entire training set and evaluating on the test set
        model.fit(train_data[new_predictors], train_data["Target"])
        predictions = model.predict(test_data[new_predictors])

        # Expanded Model Evaluation
        print("Model Evaluation on Test Set:")
        print(f"Precision: {precision_score(test_data['Target'], predictions)}")
        print(f"Recall: {recall_score(test_data['Target'], predictions)}")
        print(f"F1 Score: {f1_score(test_data['Target'], predictions)}")
        print(f"Confusion Matrix: \n{confusion_matrix(test_data['Target'], predictions)}")
        print(f"ROC-AUC Score: {roc_auc_score(test_data['Target'], predictions)}")

    except Exception as e:
        print("Error processing data:", e)
