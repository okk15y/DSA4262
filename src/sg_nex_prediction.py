import os
import subprocess
from prediction import make_predictions

# Get the names of the folders
command = ['aws', 's3', '--no-sign-request', 'ls', 's3://sg-nex-data/data/processed_data/m6Anet/']
result = subprocess.run(command, capture_output=True, text=True)

# Extract folder names from the result
folders = [line.split()[1].strip('/') for line in result.stdout.splitlines() if line]

# Model to be used
MODEL_PATH = '../artifacts/best_model_normal_smote.keras'
print(f'Model used: {MODEL_PATH}')

# Loop through the folders to perform predictions
for folder in folders:
    # Define the paths
    file_path = f's3://sg-nex-data/data/processed_data/m6Anet/{folder}/data.json'  # Path to the data.json file in S3
    data_path = f'../data/{folder}_data.json'  # Path to save the data.json file locally
    pred_path = f'../data/predictions/{folder}_pred.csv'  # Path to save predictions as CSV file

    # Download the data.json file if it does not exist
    print(f'Downloading {folder}/data.json to {data_path}')
    if not os.path.exists(data_path):
        download_command = ['aws', 's3', '--no-sign-request', 'cp', file_path, data_path]
        subprocess.run(download_command, check=True)

    # Perform the prediction using the make_predictions function
    make_predictions(data_path, pred_path, MODEL_PATH)