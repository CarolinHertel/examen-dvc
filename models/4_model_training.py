# Using the parameters found through GridSearch, we will train the model and save the trained model in the models directory.

import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

# Paths
processed_path = os.path.join("data", "processed_data")
model_path = os.path.join("src", "models")
os.makedirs(model_path, exist_ok=True)

# Load scaled training data and best parameters
X_train = pd.read_csv(os.path.join(processed_path, "X_train_scaled.csv"))
y_train = pd.read_csv(os.path.join(processed_path, "y_train.csv"))
best_params = joblib.load(os.path.join(model_path, "best_params.pkl"))

# Flatten y if needed
if isinstance(y_train, pd.DataFrame):
    y_train = y_train.values.ravel()

# Train model with best parameters
model = RandomForestRegressor(random_state=42, **best_params)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, os.path.join(model_path, "final_model.pkl"))

print("✅ Model trained and saved to final_model.pkl")
