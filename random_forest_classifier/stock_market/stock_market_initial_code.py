import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

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
horizons = [2, 5, 60, 250, 1000]
new_predictors = []
for horizon in horizons:
    rolling_averages = sp500.rolling(horizon).mean()

    ratio_column = f"Close_Ratio_{horizon}"
    sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"]

    trend_column = f"Trend_{horizon}"
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]

    new_predictors += [ratio_column, trend_column]

sp500 = sp500.dropna()

# Improved model
model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)
predictors = ["Close", "Volume", "Open", "High", "Low"]


def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    predictions = model.predict_proba(test[predictors])[:, 1]
    predictions[predictions >= 0.6] = 1
    predictions[predictions < 0.6] = 0
    predictions = pd.Series(predictions, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], predictions], axis=1)
    return combined


def backtest(data, predictors, model, start=250*10, step=250):
    all_predictions = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
        return pd.concat(all_predictions)


predictions = backtest(sp500, new_predictors, model)
print(predictions["Predictions"].value_counts())
print(predictions["Predictions"].value_counts() / predictions.shape[0])
print(precision_score(predictions["Target"], predictions["Predictions"]))
