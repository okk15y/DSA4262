import gzip
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_json_gz_to_dataframe(file_path):
    """
    Load gzipped JSON data and return it as a DataFrame.
    """
    data = []
    with gzip.open(file_path, 'rt') as f:
        for line in f:
            json_data = json.loads(line)
            for transcript, positions in json_data.items():
                for position, sequences in positions.items():
                    position = int(position)
                    for sequence, reads in sequences.items():
                        data.append({
                            'transcript_id': transcript,
                            'position': position,
                            'sequence': sequence,
                            'data': reads
                        })
    return pd.DataFrame(data)

def extract_mean_data(dataset):
    """
    Calculate the mean of 'data' column values for each row.
    """
    dataset['mean_data'] = dataset['data'].apply(lambda x: np.mean(x, axis=0))
    return dataset.dropna(subset=['mean_data'])

def scale_mean_data(dataset, scaler=None, scaler_path='mean_data_scaler.pkl'):
    """
    Scale the 'mean_data' column using StandardScaler and save the scaler.
    """
    if scaler is None:
        scaler = StandardScaler()
        mean_data_scaled = scaler.fit_transform(np.vstack(dataset['mean_data'].values))
        dataset['mean_data_scaled'] = list(mean_data_scaled)
        joblib.dump(scaler, scaler_path)
        print('Scaler saved to', scaler_path)
    else:
        dataset['mean_data_scaled'] = list(scaler.transform(np.vstack(dataset['mean_data'].values)))
    return dataset

def load_scaler(scaler_path='mean_data_scaler.pkl'):
    """
    Load the saved scaler object from the given path.
    """
    return joblib.load(scaler_path)

def DRACH_encoder():
    """
    Return a OneHotEncoder object with predefined DRACH motifs.
    """
    # Define DRACH motifs to be used for one-hot encoding
    D, R, A, C, H = ['A', 'G', 'T'], ['A', 'G'], ['A'], ['C'], ['A', 'C', 'T']
    drach_motifs = [d + r + a + c + h for d in D for r in R for a in A for c in C for h in H]
    encoder = OneHotEncoder(categories=[drach_motifs], handle_unknown='ignore')
    return encoder

def one_hot_encode_DRACH(dataset, encoder=None, encoder_path='drach_encoder.pkl'):
    """
    Apply one-hot encoding to the middle 'sequence' column based on predefined motifs.
    """
    # Extract middle sequence for one-hot encoding
    dataset['middle_sequence'] = dataset['sequence'].apply(lambda x: x[1:-1] if len(x) > 2 else '')

    # One-hot encode the middle sequence
    if encoder is None:
        encoder = DRACH_encoder()
        one_hot_matrix = encoder.fit_transform(dataset[['middle_sequence']])
        joblib.dump(encoder, encoder_path)
        print('Encoder saved to', encoder_path)
    else:
        one_hot_matrix = encoder.transform(dataset[['middle_sequence']])

    dataset['middle_sequence_OHE'] = list(one_hot_matrix.toarray())
    return dataset

def load_DRACH_encoder(encoder_path='drach_encoder.pkl'):
    """
    Load the saved DRACH encoder object from the given path.
    """
    return joblib.load(encoder_path)

def prepare_for_model(dataset):
    """
    Combine scaled 'mean_data' and one-hot encoded 'middle_sequence' for model input.
    """
    combined_features = np.hstack([np.vstack(dataset['mean_data_scaled']), np.vstack(dataset['middle_sequence_OHE'])])
    return combined_features