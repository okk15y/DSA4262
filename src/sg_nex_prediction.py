import os
from prediction import make_predictions

# Get the names of the folders
folders = !aws s3 --no-sign-request ls s3://sg-nex-data/data/processed_data/m6Anet/
folders = [f.split()[1].strip('/') for f in folders]

# Model to be used
MODEL_PATH = '../artifacts/best_model_normal_smote.keras'
print(f'Model used: {MODEL_PATH}')

# Loop through the folders to perform predictions
for folder in folders:
    # Define the paths
    file_path = f's3://sg-nex-data/data/processed_data/m6Anet/{folder}/data.json' # Path to the data.json file in S3
    data_path = f'../data/{folder}_data.json' # Path to save the data.json file
    pred_path = f'../data/{folder}_pred.csv' # Saved as CSV file

    # Download the data.json file if it does not exist
    print(f'Downloading {folder}/data.json to {data_path}')
    if not os.path.exists(data_path):
        !aws s3 --no-sign-request cp $file_path $data_path

    # Perform the prediction using the prediction.py script
    make_predictions(data_path, pred_path, MODEL_PATH)