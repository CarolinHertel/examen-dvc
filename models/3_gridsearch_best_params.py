# Decide on the regression model to implement and the parameters to test. 
# At the end of this script, we will have the best parameters saved as a .pkl file in the models directory.

import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Paths
processed_path = os.path.join("data", "processed_data")
model_path = os.path.join("src", "models")
os.makedirs(model_path, exist_ok=True)

# Load training data
X_train = pd.read_csv(os.path.join(processed_path, "X_train_scaled.csv"))
y_train = pd.read_csv(os.path.join(processed_path, "y_train.csv"))

# Flatten y if needed
if isinstance(y_train, pd.DataFrame):
    y_train = y_train.values.ravel()

# Model and parameter grid
model = RandomForestRegressor(random_state=42)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# GridSearchCV
grid_search = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=2,
    return_train_score=True
)
grid_search.fit(X_train, y_train)

# Save best params and full results
joblib.dump(grid_search.best_params_, os.path.join(model_path, "best_params.pkl"))
pd.DataFrame(grid_search.cv_results_).to_csv(os.path.join(model_path, "grid_search_results.csv"), index=False)

print("✅ Best parameters saved to best_params.pkl")
print(f"🏆 Best Params: {grid_search.best_params_}")
