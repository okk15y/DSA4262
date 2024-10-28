import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import AUC
from sklearn.model_selection import GroupKFold
from helper_functions import (
    load_json_gz_to_dataframe,
    extract_mean_data,
    scale_mean_data,
    one_hot_encode_DRACH,
    prepare_for_model
)

def build_model(input_shape):
    """
    Define and compile neural network model
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

def combine_data(dataset, labels):
    """
    Combine dataset with labels
    """
    # Left join dataset with labels on 'transcript_id' and 'position'
    merged_df = pd.merge(dataset, labels,
                         left_on=['transcript_id', 'position'],
                         right_on=['transcript_id', 'transcript_position'],
                         how='left')
    # Reorder gene_id to the first column and drop duplicate columns
    gene_id = merged_df['gene_id']
    merged_df = merged_df.drop(columns=['transcript_position', 'gene_id'])
    merged_df.insert(0, 'gene_id', gene_id)
    return merged_df

def train_model(input_path, label_path):
    # Load and preprocess data
    labels = pd.read_csv(label_path)
    dataset = load_json_gz_to_dataframe(input_path)
    dataset = extract_mean_data(dataset)
    dataset = scale_mean_data(dataset)  # Scales and saves the scaler

    # Fit and save the DRACH encoder
    dataset = one_hot_encode_DRACH(dataset)

    # Combine labels with the dataset
    dataset = combine_data(dataset, labels)

    # Prepare model input
    X = prepare_for_model(dataset)
    y = dataset['label'].values 
    gene_ids = dataset['gene_id'].values  # Extract gene_id for splitting

    # Split data into train and test sets, grouping by `gene_id`
    train_idx, test_idx = next(GroupKFold(n_splits=5).split(X, y, groups=gene_ids))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    gene_ids_train = gene_ids[train_idx]

    # Set up GroupKFold for cross-validation
    gkf = GroupKFold(n_splits=5) 
    best_fold_auc = 0.0

    for fold, (train_fold_idx, val_fold_idx) in enumerate(gkf.split(X_train, y_train, groups=gene_ids_train)):
        print(f'Training fold {fold + 1}...')

        X_fold_train, X_fold_val = X_train[train_fold_idx], X_train[val_fold_idx]
        y_fold_train, y_fold_val = y_train[train_fold_idx], y_train[val_fold_idx]

        model = build_model(X_fold_train.shape[1])
        checkpoint = ModelCheckpoint(
            f'best_model_fold_{fold + 1}.keras',
            save_best_only=True,
            monitor='val_auc_pr',
            mode='max'
        )
        
        model.fit(
            X_fold_train, y_fold_train,
            epochs=20,
            batch_size=32,
            validation_data=(X_fold_val, y_fold_val),
            callbacks=[checkpoint]
        )

        # Load the best model for this fold and evaluate on the test set
        model.load_weights(f'best_model_fold_{fold + 1}.keras')
        test_auc = model.evaluate(X_test, y_test, verbose=0)[1]  # AUC-PR score

        print(f'Fold {fold + 1} Test AUC-PR: {test_auc:.4f}')

        # Save model if it has the highest AUC-PR so far
        if test_auc > best_fold_auc:
            best_fold_auc = test_auc
            best_fold_model = f'best_model_fold_{fold + 1}.keras'

    print(f'Best model from cross-validation: {best_fold_model} with Test AUC-PR: {best_fold_auc:.4f}')

    # Save the best model
    model.load_weights(best_fold_model)
    model.save('trained_model.keras')

if __name__ == "__main__":
    input_path = input("Enter path to training data: ")
    labels_path = input("Enter path to training labels: ")
    train_model(input_path, labels_path)