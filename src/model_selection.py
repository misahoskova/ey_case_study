import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_data():
    X = pd.read_csv("../data/X_features.csv")
    y = pd.read_csv("../data/y_target.csv").squeeze()

    return X, y

def split_data(X, y, test_size = 0.2, random_state = 42):
    return train_test_split(X, y, test_size = test_size, random_state = random_state)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    return {
    "mae": mean_absolute_error(y_test, y_pred),
    "mse": mean_squared_error(y_test, y_pred),
    "r2": r2_score(y_test, y_pred),
    "model": model
}

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        "Random Forest": RandomForestRegressor(random_state = 42),
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state = 42),
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        result = evaluate_model(model, X_test, y_test)
        result["name"] = name
        results.append(result)

    return results

def save_best_model(results, output_dir = "../models"):
    best = max(results, key = lambda x: x["r2"])
    os.makedirs(output_dir, exist_ok = True)
    path = os.path.join(output_dir, f"{best['name'].lower().replace(' ', '_')}.joblib")
    joblib.dump(best["model"], path)

    return best["name"], path


def run_model_selection():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    best_name, model_path = save_best_model(results)
    print(f"Best model: {best_name} saved to {model_path}")

    return results