import os
import subprocess
import argparse
from prediction import make_predictions

ARTIFACTS_FOLDER = "../artifacts"
DEFAULT_MODEL_PATH = f"{ARTIFACTS_FOLDER}/trained_model.keras"

def main(model_path):
    # Get the names of the folders
    command = ['aws', 's3', '--no-sign-request', 'ls', 's3://sg-nex-data/data/processed_data/m6Anet/']
    result = subprocess.run(command, capture_output=True, text=True)

    # Extract folder names from the result
    folders = [line.split()[1].strip('/') for line in result.stdout.splitlines() if line]

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
        make_predictions(data_path, pred_path, model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make predictions on all SG-NEx data using the trained model.')
    parser.add_argument('--model', type=str, default=None, help='Path to the trained model file (Keras).')
    args = parser.parse_args()

    # if no model path input, default to specified path
    if args.model is None:
        args.model = DEFAULT_MODEL_PATH
        print(f"Using default model path: {DEFAULT_MODEL_PATH}")
    
    main(args.model)