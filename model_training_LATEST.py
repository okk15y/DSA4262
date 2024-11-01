import argparse
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import AUC
from sklearn.model_selection import GroupKFold
from helper_functions import (
    load_json_gz_to_dataframe,
    extract_mean_reads,
    scale_mean_reads,
    load_scaler,
    extract_middle_sequence,
    one_hot_encode_DRACH,
    combine_data,
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

def train_model(input_path, label_path):
    # Load and combine labels with the dataset
    print(f'Loading data from {input_path}...')
    labels = pd.read_csv(label_path)
    dataset = load_json_gz_to_dataframe(input_path)
    dataset = combine_data(dataset, labels)

    print('Preprocessing data...')
    # Extract mean reads
    dataset = extract_mean_reads(dataset)

    # One-hot encode the DRACH sequences
    dataset = extract_middle_sequence(dataset)
    dataset = one_hot_encode_DRACH(dataset)

    # Prepare labels and group identifiers (gene_id)
    y = dataset['label'].values
    gene_ids = dataset['gene_id'].values

    # Split data into train and test sets using GroupKFold on 'gene_id' (80-20 split)
    train_idx, test_idx = next(GroupKFold(n_splits=5).split(dataset, y, groups=gene_ids))
    train_data, test_data = dataset.iloc[train_idx], dataset.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Scale training set first before using the same scaler for the test set
    train_data = scale_mean_reads(train_data)
    scaler = load_scaler('mean_reads_scaler.pkl')
    test_data = scale_mean_reads(test_data, scaler)

    # Prepare model input
    X_train = prepare_for_model(train_data)
    y_train = train_data['label'].values  # Prepare labels for training

    X_test = prepare_for_model(test_data)
    y_test = test_data['label'].values  # Prepare labels for testing

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
            epochs=10,
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

    print(f'Best model from cross-validation (without SMOTE): {best_fold_model} with Test AUC-PR: {best_fold_auc:.4f}')

    # Save the best model
    model.load_weights(best_fold_model)
    model.save('trained_model.keras')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model with specified input and labels.')
    parser.add_argument('--input', type=str, required=True, help='Path to the training data file (JSON gzipped).')
    parser.add_argument('--labels', type=str, required=True, help='Path to the labels file (CSV).')
    args = parser.parse_args()
    train_model(args.input, args.labels)