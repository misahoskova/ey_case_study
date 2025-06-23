import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X = pd.read_csv("../data/X_features.csv")
y = pd.read_csv("../data/y_target.csv").squeeze()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 42
)

param_dist = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 20, None],
    "min_samples_split": [2, 5, 10],
    "max_features": ["sqrt", "log2, None"]
}

rf = RandomForestRegressor(random_state = 42, n_jobs = -1)

random_search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter = 5,
    cv = 2,
    verbose = 1,
    n_jobs = -1,
    scoring = "r2"
)

random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_
best_params = random_search.best_params_

y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Best model parameters:")
print("Parameters:", best_params)
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R²:  {r2:.4f}")

os.makedirs("../models", exist_ok = True)
model_path = "../models/random_forest_optimized.joblib"
joblib.dump(best_model, model_path)
print(f"Optimalized mode saved to: {model_path}")