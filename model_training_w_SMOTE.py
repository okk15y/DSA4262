import argparse
import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import AUC
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GroupKFold
from helper_functions import (
    load_json_gz_to_dataframe,
    extract_mean_reads,
    scale_mean_reads,
    load_scaler,
    extract_middle_sequence,
    load_DRACH_encoder,
    one_hot_encode_DRACH,
    combine_data,
    prepare_for_model
)

def build_model(input_shape):
    """
    Define and compile neural network model.
    """
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(150, activation='relu'),
        Dropout(0.2),  # Dropout layer for regularization
        Dense(32, activation='relu'),
        Dropout(0.2),  # Another dropout layer
        Dense(1, activation='sigmoid')
    ])
    # Set AUC with Precision-Recall (PR) curve
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[AUC(curve='PR', name='auc_pr')])
    return model

def train_model(input_path, label_path):
    # Load and combine labels with the dataset
    labels = pd.read_csv(label_path)
    dataset = load_json_gz_to_dataframe(input_path)
    dataset = combine_data(dataset, labels)
    dataset = extract_mean_reads(dataset)
    dataset = extract_middle_sequence(dataset)

    # Prepare labels and group identifiers (gene_id)
    y = dataset['label'].values
    gene_ids = dataset['gene_id'].values

    # Split data into train and test sets using GroupKFold on 'gene_id'
    train_idx, test_idx = next(GroupKFold(n_splits=5).split(dataset, y, groups=gene_ids))
    train_data, test_data = dataset.iloc[train_idx], dataset.iloc[test_idx]

    # Scale train data and apply the same scaling to test data
    train_data = scale_mean_reads(train_data)
    scaler = load_scaler('mean_reads_scaler.pkl')
    test_data = scale_mean_reads(test_data, scaler)

    # Prepare model input for the training dataset
    test_data = one_hot_encode_DRACH(test_data)
    X_test = prepare_for_model(test_data)
    y_test = test_data['label'].values

    # Perform SMOTE for each DRACH motif in the training set
    encoder = load_DRACH_encoder('drach_encoder.pkl')
    DRACH_motifs = encoder.categories_

    X_resampled, y_resampled = [], []

    for motif in DRACH_motifs:
        # Filter training data for the current motif
        motif_data = train_data[train_data['middle_sequence'] == motif] 
        y_motif = motif_data['label'].values
        X_motif = np.vstack(motif_data['scaled_mean_reads'].values)

        # Apply SMOTE on the current motif's training data
        smote = SMOTE(random_state=4262)
        X_motif_resampled, y_motif_resampled = smote.fit_resample(X_motif, y_motif)

        # One-hot encode the resampled features for the current motif
        X_motif_resampled_encoded = one_hot_encode_DRACH(X_motif_resampled, encoder)

        # Append the resampled and encoded data to the overall dataset
        X_resampled.append(X_motif_resampled_encoded)
        y_resampled.append(y_motif_resampled)


    # Combine the resampled data from all motifs
    X_resampled = np.vstack(X_resampled)
    y_resampled = np.concatenate(y_resampled)

    # Build and train the model on the combined resampled training data
    model = build_model(X_resampled.shape[1])
    checkpoint = ModelCheckpoint(
        'best_model_with_SMOTE_and_DRACH.keras',
        save_best_only=True,
        monitor='val_auc_pr',
        mode='max'
    )

    # Train the model with a validation split from the resampled training data
    model.fit(
        X_resampled, y_resampled,
        epochs=5,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint]
    )

    print('Model training complete!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model with specified input and labels.')
    parser.add_argument('--input', type=str, required=True, help='Path to the training data file (JSON gzipped).')
    parser.add_argument('--labels', type=str, required=True, help='Path to the labels file (CSV).')
    args = parser.parse_args()
    train_model(args.input, args.labels)