# As you may notice, the data varies widely in scale, so normalization is necessary. 
# You can use existing functions to construct this script. 
# As output, this script will create two new datasets (X_train_scaled, X_test_scaled) which you will also save in data/processed.

# scripts/normalize_data.py

import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Define paths
processed_path = os.path.join("data", "processed_data")
X_train_path = os.path.join(processed_path, "X_train.csv")
X_test_path = os.path.join(processed_path, "X_test.csv")

# Load datasets
X_train = pd.read_csv(X_train_path)
X_test = pd.read_csv(X_test_path)

# Drop datetime if present (optional, based on your modeling needs)
if 'date' in X_train.columns:
    X_train = X_train.drop(columns=['date'])
    X_test = X_test.drop(columns=['date'])

# Fit scaler on training data only
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame with original column names
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Save the scaled datasets
X_train_scaled_df.to_csv(os.path.join(processed_path, "X_train_scaled.csv"), index=False)
X_test_scaled_df.to_csv(os.path.join(processed_path, "X_test_scaled.csv"), index=False)

print("✅ Data successfully normalized and saved.")
