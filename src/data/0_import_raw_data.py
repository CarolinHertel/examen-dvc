import os
import requests

# URL of the CSV file
url = "https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv"

# Local path where the file should be saved
output_path = "data/raw_data/raw.csv"

# Create the directory if it doesn't exist
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Download the file
response = requests.get(url)

# Check if the download was successful
if response.status_code == 200:
    with open(output_path, 'wb') as f:
        f.write(response.content)
    print(f"File downloaded successfully and saved to {output_path}")
else:
    print(f"Failed to download file. Status code: {response.status_code}")
