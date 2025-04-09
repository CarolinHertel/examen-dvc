# Finally, using the trained model, we will evaluate its performance and make predictions. 
# At the end of this script, we will have a new dataset in data containing the predictions, 
# along with a scores.json file in the metrics directory that will capture evaluation metrics of our model (e.g., MSE, R2).

import os
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import mean_squared_error, r2_score

# Define paths
processed_path = os.path.join("data", "processed_data")
model_path = os.path.join("src", "models")
metrics_path = os.path.join("metrics")
os.makedirs(metrics_path, exist_ok=True)

# Load data and model
X_test = pd.read_csv(os.path.join(processed_path, "X_test_scaled.csv"))
y_test = pd.read_csv(os.path.join(processed_path, "y_test.csv"))
model = joblib.load(os.path.join(model_path, "final_model.pkl"))

# Flatten y if it's a dataframe
if isinstance(y_test, pd.DataFrame):
    y_test = y_test.values.ravel()

# Predict
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Save predictions
pred_df = pd.DataFrame({
    'actual': y_test,
    'predicted': y_pred
})
pred_df.to_csv(os.path.join("data", "predictions.csv"), index=False)

# Save scores
scores = {
    "mse": mse,
    "rmse": rmse,
    "r2_score": r2
}
with open(os.path.join(metrics_path, "scores.json"), "w") as f:
    json.dump(scores, f, indent=4)

# Output summary
print("✅ Evaluation complete.")
print(f"📉 MSE: {mse:.4f}")
print(f"📏 RMSE: {rmse:.4f}")
print(f"📈 R²: {r2:.4f}")
print("📁 Predictions saved to data/predictions.csv")
print("📁 Metrics saved to metrics/scores.json")