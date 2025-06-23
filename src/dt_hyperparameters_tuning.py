import os
import joblib
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X = pd.read_csv("../data/X_features.csv")
y = pd.read_csv("../data/y_target.csv").squeeze()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 42
)

param_dist = {
    "max_depth": [5, 10, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

dt = DecisionTreeRegressor(random_state=42)

search = RandomizedSearchCV(
    dt,
    param_distributions=param_dist,
    n_iter = 5,
    cv = 2,
    scoring = "r2",
    verbose = 1,
    n_jobs = -1
)

search.fit(X_train, y_train)
best_model = search.best_estimator_

y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nThe Best Decision Tree model:")
print("Parameters:", search.best_params_)
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R²:  {r2:.4f}")

os.makedirs("../models", exist_ok = True)
joblib.dump(best_model, "../models/decision_tree_optimized.joblib")
print("Model saved to ../models/decision_tree_optimized.joblib")