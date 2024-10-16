import gzip
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import load_model


def load_json_gz_to_dataframe(file_path, num_lines=0):
    '''
    If num_lines <= 0, read all lines.
    Else, read until specified number of lines.
    '''
    data = []
    with gzip.open(file_path) as f:
        for i, line in enumerate(f, start=1):
            if num_lines > 0 and i > num_lines:
                break
            json_data = json.loads(line)
            for transcript, positions in json_data.items():
                for position, sequences in positions.items():
                    position = int(position)
                    for sequence, reads in sequences.items():
                        data.append({
                            'transcript_id': transcript,
                            'position': position,
                            'sequence': sequence,
                            "data" : reads
                            })
    return pd.DataFrame(data)


def data_processing(dataset):
    transcriptID_position = dataset[['transcript_id', 'position']]


    # Get mean data
    dataset['data'] = dataset['data'].apply(lambda x: np.mean(x, axis=0))

    
    # Standardize scale

    # Extract the lists from the 'data' column
    data_lists = np.vstack(dataset['data'].values)

    # Initialize the StandardScaler
    scaler = joblib.load('base_training_model_scaler.pkl')

    # Fit and transform the data
    data_scaled = scaler.transform(data_lists)

    # Replace the original 'data' column with the scaled data
    dataset['data'] = list(data_scaled)

    dataset = np.vstack(dataset["data"].values)

    return dataset, transcriptID_position


def predict(dataset, transcriptID_position, model):
    predictions = model.predict(dataset)
    predictions_df = pd.DataFrame(predictions, columns=['score'])
    predictions_df = pd.concat([transcriptID_position, predictions_df], axis=1)
    predictions_df.rename(columns={'position': 'transcript_position'}, inplace=True)
    predictions_df.to_csv('predictions.csv', index=False)


def main(file_path):
    # change this to name of model
    model = load_model('base_model.keras')
    
    input_data = load_json_gz_to_dataframe(file_path)
    input_data, transcriptID_position = data_processing(input_data)
    print("Processed Dataset:\n", input_data[:5])
    print("TranscriptID:\n", transcriptID_position[:5])
    
    predict(input_data, transcriptID_position, model)


if __name__ == "__main__":
    input_file = input("File Path of json.gz dataset: ")
    main(input_file)
