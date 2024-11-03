import os
import gzip
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# If no existing mean reads scaler/ drach encoder, new ones will be created and saved to these paths
ARTIFACTS_FOLDER = "../artifacts"
MEAN_READS_SCALER_PATH = f"{ARTIFACTS_FOLDER}/mean_reads_scaler.pkl"
DRACH_ENCODER_PATH = f"{ARTIFACTS_FOLDER}/drach_encoder.pkl"

def load_data_to_dataframe(file_path):
    """
    Load JSON data or gzipped JSON data into a DataFrame.
    
    Parameters:
    - file_path (str): Path to the JSON file or gzipped JSON file.
    
    Returns:
    - pd.DataFrame: DataFrame containing the loaded data.
    """
    data = []
    
    # Open the file using gzip.open if it ends with .gz
    if file_path.endswith('.gz'):
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
                                'reads': reads
                            })
    else:
        with open(file_path, 'r') as f:
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
                                'reads': reads
                            })
    return pd.DataFrame(data)


def combine_data(df, labels):
    """
    Combine df with labels
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing the unlabelled.
    - labels (pd.DataFrame): DataFrame containing the labels.
    
    Returns:
    - pd.DataFrame: Merged DataFrame with combined data and labels.
    """
    # Left join df with labels on 'transcript_id' and 'position'
    merged_df = pd.merge(df, labels,
                         left_on=['transcript_id', 'position'],
                         right_on=['transcript_id', 'transcript_position'],
                         how='left')
    # Reorder gene_id to the first column and drop duplicate columns
    gene_id = merged_df['gene_id']
    merged_df = merged_df.drop(columns=['transcript_position', 'gene_id'])
    merged_df.insert(0, 'gene_id', gene_id)
    return merged_df


def extract_mean_reads(df):
    """
    Compute the mean of 'reads' for each row.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing the 'reads' column.
    
    Returns:
    - pd.DataFrame: DataFrame with an additional 'mean_reads' column.
    """
    df['mean_reads'] = df['reads'].apply(lambda x: np.mean(x, axis=0))
    return df


def extract_middle_sequence(df):
    """
    Extract the middle 5-mers sequence from the 'sequence' column.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing the 'sequence' column.
    
    Returns:
    - pd.DataFrame: DataFrame with an additional 'middle_sequence' column.
    """
    df['middle_sequence'] = df['sequence'].apply(lambda x: x[1:-1])
    return df


###################
# Scaling
###################
def scale_mean_reads(df, scaler=None, scaler_path=MEAN_READS_SCALER_PATH):
    """
    Scale the 'mean_reads' column using StandardScaler.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing the 'mean_reads' column.
    - scaler (StandardScaler, optional): Pre-fitted scaler. If None, a new scaler will be fitted.
    - scaler_path (str): Path to save the fitted scaler.
    
    Returns:
    - scaler: The provided or newly fitted scaler.
    - df (pd.DataFrame): DataFrame with an additional 'scaled_mean_reads' column.
    """
    df = df.copy() # to avoid warning
    if scaler is None: # If no scaler is provided, fit a new one and save it
        scaler = StandardScaler()
        scaled_mean_reads = scaler.fit_transform(np.vstack(df['mean_reads'].values))
        joblib.dump(scaler, scaler_path)
        print('Scaler saved to', scaler_path)
    else: # Use the provided scaler to transform the data
        scaled_mean_reads = scaler.transform(np.vstack(df['mean_reads'].values))
    df['scaled_mean_reads'] = list(scaled_mean_reads)
    return scaler, df

def load_scaler(scaler_path=MEAN_READS_SCALER_PATH):
    """
    Load the saved scaler from the given path.
    
    Parameters:
    - scaler_path (str): Path to the saved scaler file.
    
    Returns:
    - StandardScaler: Loaded scaler.
    """
    return joblib.load(scaler_path)


########################
# DRACH encoder
########################
def drach_encoder():
    """
    Return a OneHotEncoder object with predefined DRACH motifs.
    
    Returns:
    - OneHotEncoder: Encoder for DRACH motifs.
    """
    # Define DRACH motifs to be used for one-hot encoding
    D, R, A, C, H = ['A', 'G', 'T'], ['A', 'G'], ['A'], ['C'], ['A', 'C', 'T']
    drach_motifs = [d + r + a + c + h for d in D for r in R for a in A for c in C for h in H]
    encoder = OneHotEncoder(categories=[drach_motifs], handle_unknown='ignore')
    return encoder

def load_DRACH_encoder(encoder_path=DRACH_ENCODER_PATH):
    """
    Load the saved DRACH encoder from the given path.
    
    Parameters:
    - encoder_path (str): Path to the saved encoder file.
    
    Returns:
    - Encoder object: Loaded encoder.
    """
    return joblib.load(encoder_path)

def one_hot_encode_DRACH(df, encoder=None, encoder_path=DRACH_ENCODER_PATH):
    """
    Apply one-hot encoding to the middle 5-mers sequence.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing the 'middle_sequence' column.
    - encoder (OneHotEncoder, optional): Pre-fitted encoder. If None, a new encoder will be fitted.
    - encoder_path (str): Path to save the fitted encoder.
    
    Returns:
    - encoder: The provided or newly fitted encoder.
    - df (pd.DataFrame): DataFrame with an additional 'middle_sequence_OHE' column.
    """
    # One-hot encode the middle sequence
    if encoder is None: # If no encoder is provided, fit a new one and save it
        encoder = drach_encoder()
        one_hot_matrix = encoder.fit_transform(df[['middle_sequence']])
        joblib.dump(encoder, encoder_path)
        print('DRACH Encoder saved to', encoder_path)
    else:
        one_hot_matrix = encoder.transform(df[['middle_sequence']])
    df['middle_sequence_OHE'] = list(one_hot_matrix.toarray())
    return encoder, df


##############################
# Prepare data for model input
##############################
def prepare_for_model(df):
    """
    Combine 'scaled_mean_reads' and `middle_sequence_OHE` for model input.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing 'scaled_mean_reads' and 'middle_sequence_OHE' columns.
    
    Returns:
    - np.ndarray: Combined features for model input.
    """
    combined_features = np.hstack([np.vstack(df['scaled_mean_reads']), np.vstack(df['middle_sequence_OHE'])])
    return combined_features
