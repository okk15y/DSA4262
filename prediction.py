import gzip
import json
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
    scaler = StandardScaler()

    # Fit and transform the data
    data_scaled = scaler.fit_transform(data_lists)

    # Replace the original 'data' column with the scaled data
    dataset['data'] = list(data_scaled)

    
    # Extract the middle 5-mers sequence
    dataset['sequence'] = dataset['sequence'].apply(lambda x: x[1:-1])

    
    # One-hot encoding
    
    # Define the possible nucleotides for each position
    D = ['A', 'G', 'T']
    R = ['A', 'G']
    A = ['A']
    C = ['C']
    H = ['A', 'C', 'T']
    
    # Initialize an empty list to store the DRACH motifs
    drach_motifs = []
    
    # Generate all combinations using nested loops
    for d in D:
        for r in R: 
            for a in A:
                for c in C:
                    for h in H:
                        motif = d + r + a + c + h
                        drach_motifs.append(motif)

    # Initialize the OneHotEncoder with DRACH motifs
    encoder = OneHotEncoder(categories=[drach_motifs])

    # Fit the encoder to 'sequence' and transform it into a one-hot encoded matrix
    one_hot_matrix = encoder.fit_transform(dataset[['sequence']])

    # Convert the one-hot encoded matrix into a column
    one_hot_column = pd.Series([list(row) for row in one_hot_matrix.toarray()])

    # Concatenate the one-hot encoded column to dataset
    df_encoded = pd.concat([dataset, one_hot_column.rename('one_hot_encoded')], axis=1)

    # Combine 'data' and 'one_hot_encoded' into a single list for each row
    df_encoded['combined'] = df_encoded.apply(lambda x: x['data'].tolist() + x['one_hot_encoded'], axis=1)

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
    model = load_model('best_model.keras')
    
    input_data = load_json_gz_to_dataframe(file_path)
    input_data, transcriptID_position = data_processing(input_data)
    
    predict(input_data, transcriptID_position, model)


if __name__ == "__main__":
    input_file = input("File Path of json.gz dataset: ")
    main(input_file)