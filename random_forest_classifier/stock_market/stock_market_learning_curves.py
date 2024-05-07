import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve

# Configurable Parameters
START_DATE = "1990-01-01"
HORIZONS = [2, 5, 60, 250, 1000]
BACKTEST_START = 250 * 10
BACKTEST_STEP = 250


def fetch_sp500_data():
    """Fetch S&P 500 historical data."""
    try:
        sp500 = yf.Ticker("^GSPC")
        sp500 = sp500.history(period="max")
        return sp500
    except Exception as e:
        print("Error fetching data:", e)
        return None


def preprocess_data(data):
    """Preprocess the data by adding target and predictors."""
    data["Tomorrow"] = data["Close"].shift(-1)
    data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)

    predictors = []
    for horizon in HORIZONS:
        rolling_averages = data.rolling(horizon).mean()
        ratio_column = f"Close_Ratio_{horizon}"
        data[ratio_column] = data["Close"] / rolling_averages["Close"]
        trend_column = f"Trend_{horizon}"
        data[trend_column] = data.shift(1).rolling(horizon).sum()["Target"]
        predictors += [ratio_column, trend_column]

    return data.dropna(), predictors


def train_and_evaluate(data, predictors):
    """Train the model and evaluate its performance using cross-validation."""
    train_data, test_data = train_test_split(data, test_size=0.25, random_state=1)

    # Model with regularization parameters
    model = RandomForestClassifier(n_estimators=200, min_samples_split=5, max_depth=10,
                                   min_samples_leaf=5, max_features='sqrt', random_state=1)

    # Evaluate using cross-validation
    plot_learning_curves(model, train_data[predictors], train_data["Target"])
    cross_val_scores = cross_val_score(model, train_data[predictors], train_data["Target"], cv=5)
    print("Cross-validation scores:", cross_val_scores)
    print("Average CV score:", cross_val_scores.mean())

    # Train on entire training set and evaluate on test set
    model.fit(train_data[predictors], train_data["Target"])
    predictions = model.predict(test_data[predictors])

    evaluate_model(test_data["Target"], predictions)


def plot_learning_curves(model, X, y):
    """Plot learning curves for a given model."""
    train_sizes, train_scores, validation_scores = learning_curve(
        estimator=model,
        X=X,
        y=y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    # Calculate the mean and standard deviation for training and validation scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    validation_scores_mean = np.mean(validation_scores, axis=1)
    validation_scores_std = np.std(validation_scores, axis=1)

    # Plot the learning curves
    plt.figure(figsize=(10, 6))
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, validation_scores_mean - validation_scores_std,
                     validation_scores_mean + validation_scores_std, alpha=0.1, color='g')
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training Score')
    plt.plot(train_sizes, validation_scores_mean, 'o-', color='g', label='Validation Score')
    plt.title('Learning Curves')
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


def evaluate_model(true_values, predictions):
    """Evaluate and print model performance metrics."""
    print("Model Evaluation on Test Set:")
    print(f"Precision: {precision_score(true_values, predictions)}")
    print(f"Recall: {recall_score(true_values, predictions)}")
    print(f"F1 Score: {f1_score(true_values, predictions)}")
    print(f"Confusion Matrix: \n{confusion_matrix(true_values, predictions)}")
    print(f"ROC-AUC Score: {roc_auc_score(true_values, predictions)}")


def main():
    sp500 = fetch_sp500_data()
    if sp500 is not None:
        # sp500.plot.line(y="Close", use_index=True, title="S&P 500 Closing Prices")
        # plt.show()

        sp500, predictors = preprocess_data(sp500)
        train_and_evaluate(sp500, predictors)

if __name__ == "__main__":
    main()
