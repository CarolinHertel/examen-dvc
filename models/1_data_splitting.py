# Split the data into training and testing sets. Our target variable is silica_concentrate, 
# located in the last column of the dataset. This script will produce 4 datasets (X_test, X_train, y_test, y_train) that you can store in data/processed.

# scripts/split_data.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Define paths
raw_data_path = os.path.join("data", "raw_data", "raw.csv")  
processed_path = os.path.join("data", "processed_data")
os.makedirs(processed_path, exist_ok=True)

# Load data
df = pd.read_csv(raw_data_path, parse_dates=["date"])

# Split features and target
X = df.drop(columns=["silica_concentrate"])
y = df["silica_concentrate"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save the datasets
X_train.to_csv(os.path.join(processed_path, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(processed_path, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(processed_path, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(processed_path, "y_test.csv"), index=False)

print("✅ Data successfully split and saved.")
