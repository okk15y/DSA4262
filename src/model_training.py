import argparse
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import AUC
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold
from helper_functions import (
    load_data_to_dataframe,
    extract_mean_reads,
    scale_mean_reads,
    extract_middle_sequence,
    one_hot_encode_DRACH,
    combine_data,
    prepare_for_model
)

# Path to save the trained model to
ARTIFACTS_FOLDER = "../artifacts/"
OUTPUT_MODEL_PATH = f"{ARTIFACTS_FOLDER}/trained_model.keras"

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
    # Keep track of AUC-PR and AUC-ROC during training
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[
            AUC(curve='PR', name='auc_pr'),  # AUC-PR
            AUC(curve='ROC', name='auc_roc')  # AUC-ROC
        ]
    )
    return model

def train_model(input_path, label_path):
    # Load and combine labels with the unlabelled dataset
    print(f'Loading data from {input_path}...')
    labels = pd.read_csv(label_path)
    df = load_data_to_dataframe(input_path)
    df = combine_data(df, labels)

    print('Preprocessing data...')
    df = extract_mean_reads(df)
    df = scale_mean_reads(df)
    df = extract_middle_sequence(df)
    encoder, df = one_hot_encode_DRACH(df)

    X_train = prepare_for_model(df)
    y_train = df['label'].values

    # SMOTE
    smote = SMOTE(random_state=4262)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # Set up 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=4262)
    best_val_auc_pr = 0

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_resampled)):
        print(f"Training fold {fold + 1}")

        # Split the data for this fold
        X_train, X_val = X_resampled[train_idx].copy(), X_resampled[val_idx].copy()
        y_train, y_val = y_resampled[train_idx].copy(), y_resampled[val_idx].copy()

        # Initialize and compile the model
        model = build_model(X_train.shape[1])

        # Define a temporary callback to track best model in this fold
        temp_checkpoint = ModelCheckpoint(
            'temp_best_model_fold.keras', monitor='val_auc_pr', mode='max', save_best_only=True
        )

        # Train the model on this fold
        history = model.fit(
            X_train, y_train,
            epochs=5,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[temp_checkpoint]
        )

        # Track the best validation AUC-PR for this fold
        fold_best_auc_pr = max(history.history['val_auc_pr'])
        print(f"Fold {fold + 1} best AUC-PR: {fold_best_auc_pr}")

        # Update the best model if this fold's AUC-PR is the highest
        if fold_best_auc_pr > best_val_auc_pr:
            best_val_auc_pr = fold_best_auc_pr
            best_fold_model = load_model('temp_best_model_fold.keras')
            # Save the best model across all folds
            best_fold_model.save('trained_model.keras')
            print(f"New best model saved with AUC-PR: {best_val_auc_pr}")

    print("Training complete. Best model across all folds saved as 'trained_model.keras'.")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model with specified input and labels.')
    parser.add_argument('--input', type=str, required=True, help='Path to the training data file (JSON gzipped).')
    parser.add_argument('--labels', type=str, required=True, help='Path to the labels file (CSV).')
    args = parser.parse_args()
    train_model(args.input, args.labels)