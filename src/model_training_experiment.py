import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, auc, average_precision_score
from tensorflow.keras.metrics import AUC
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE
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

def flatten_reads(row):
    gene_id = row['gene_id']
    transcript_id = row['transcript_id']
    position = row['position']
    sequence = row['sequence']
    reads = row['reads']
    label = row['label']
    return pd.DataFrame({'gene_id': [gene_id] * len(reads),
                         'transcript_id': [transcript_id] * len(reads),
                         'position': [position] * len(reads),
                         'sequence': [sequence] * len(reads),
                         'reads': reads,
                         'mean_reads': reads, 
                         'label': [label] * len(reads)})

def train_model(input_path, label_path):
    # Load and combine labels with the unlabelled dataset
    print(f'Loading data from {input_path}...')
    labels = pd.read_csv(label_path)
    df = load_data_to_dataframe(input_path)
    df = combine_data(df, labels)

    print('Train-test split...')
    train_gene_ids, test_gene_ids = train_test_split(df['gene_id'].unique(), test_size=0.2, random_state=4262)
    train_df = df[df['gene_id'].isin(train_gene_ids)].copy()
    test_df = df[df['gene_id'].isin(test_gene_ids)].copy()
    
    print('Flattening reads...')
    lab1_df = train_df[train_df['label'] == 1].copy()
    lab1_df = lab1_df.apply(flatten_reads, axis=1)
    lab1_df = pd.concat(lab1_df.values, ignore_index=True)

    lab0_df = train_df[train_df['label'] == 0].copy()
    # Extract mean reads
    # lab0_df = extract_mean_reads(lab0_df)

    # Flatten lab0_df
    lab0_df = lab0_df.apply(flatten_reads, axis=1)
    lab0_df = pd.concat(lab0_df.values, ignore_index=True)

    print('Preprocessing data...')
    train_df = pd.concat([lab0_df, lab1_df], ignore_index=True)
    test_df = extract_mean_reads(test_df)

    scaler, train_df = scale_mean_reads(train_df)
    scaler, test_df = scale_mean_reads(test_df, scaler=scaler)

    train_df = extract_middle_sequence(train_df)
    test_df = extract_middle_sequence(test_df)
                                      
    encoder, train_df = one_hot_encode_DRACH(train_df)
    encoder, test_df = one_hot_encode_DRACH(test_df, encoder=encoder)

    # Smote
    print('Applying SMOTE by DRACH...')

    # Get the scaled mean reads and one-hot encoded middle sequence columns as dataframe
    X_resampled_list = [] 
    y_resampled_list = []
    middle_sequence_list = []

    for middle_seq, group in train_df.groupby('middle_sequence'):
        X = np.vstack(group['scaled_mean_reads'].values)
        y = group['label'].values
        
        if sum(y) < 10: # Skip if the number of positive labels is less than 10
            X_resampled_list.append(X)
            y_resampled_list.append(y)
            middle_sequence_list.extend([middle_seq] * len(y))
            continue

        # Apply SMOTE to the scaled mean reads and labels
        smote = SMOTE(random_state=4262)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Append resampled data and one-hot encoding to lists
        X_resampled_list.append(X_resampled)
        y_resampled_list.append(y_resampled)
        middle_sequence_list.extend([middle_seq] * len(y_resampled))

    # Combine all resampled data into final arrays
    X_resampled = np.vstack(X_resampled_list)
    y_resampled = np.concatenate(y_resampled_list)

    # Combine into a final DataFrame if needed
    resampled_drach_df = pd.DataFrame({
        'scaled_mean_reads': list(X_resampled),
        'middle_sequence': middle_sequence_list,
        'label': y_resampled
    })

    # Encode resampled data
    encoder, resampled_drach_df = one_hot_encode_DRACH(resampled_drach_df, encoder)


    # Prepare data for model
    X_resampled_drach = prepare_for_model(resampled_drach_df)
    y_resampled_drach = resampled_drach_df['label'].values

    # Without SMOTE
    X_train = prepare_for_model(train_df)
    y_train = train_df['label'].values

    # Prepare test data
    X_test = prepare_for_model(test_df)
    y_test = test_df['label'].values

    # Set up 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=4262)
    best_val_auc_pr_smote = 0
    best_val_auc_pr_no_smote = 0
    BEST_MODEL_PATH_SMOTE = "../artifacts/best_model_with_smote_experiment.keras"
    BEST_MODEL_PATH_NO_SMOTE = "../artifacts/best_model_without_smote_experiment.keras"

    # Set checkpoint to save the best model (based on validation AUC-PR)
    checkpoint_smote = ModelCheckpoint(
            BEST_MODEL_PATH_SMOTE,
            save_best_only=True,
            monitor='val_auc_pr',
            mode='max'
        )

    checkpoint_no_smote = ModelCheckpoint(
            BEST_MODEL_PATH_NO_SMOTE,
            save_best_only=True,
            monitor='val_auc_pr',
            mode='max'
        )
    # Cross-validation with SMOTE
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_resampled_drach)):
        print(f"Training fold {fold + 1} with SMOTE")

        # Split the data with SMOTE for this fold
        X_train_smote, X_val_smote = X_resampled_drach[train_idx].copy(), X_resampled_drach[val_idx].copy()
        y_train_smote, y_val_smote = y_resampled_drach[train_idx].copy(), y_resampled_drach[val_idx].copy()

        # Initialize and compile the model
        model_smote = build_model(X_train_smote.shape[1])

        # Train the model on this fold with SMOTE data
        history_smote = model_smote.fit(
            X_train_smote, y_train_smote,
            epochs=5,
            batch_size=32,
            validation_data=(X_val_smote, y_val_smote),
            callbacks=[checkpoint_smote]
        )

        # Track the best validation AUC-PR across folds with SMOTE
        fold_best_auc_pr_smote = max(history_smote.history['val_auc_pr'])
        if fold_best_auc_pr_smote > best_val_auc_pr_smote:
            best_val_auc_pr_smote = fold_best_auc_pr_smote


    # Cross-validation without SMOTE
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"Training fold {fold + 1} without SMOTE")

        # Split the data without SMOTE for this fold
        X_train_no_smote, X_val_no_smote = X_train[train_idx].copy(), X_train[val_idx].copy()
        y_train_no_smote, y_val_no_smote = y_train[train_idx].copy(), y_train[val_idx].copy()

        # Initialize and compile the model
        model_no_smote = build_model(X_train_no_smote.shape[1])

        # Train the model on this fold without SMOTE data
        history_no_smote = model_no_smote.fit(
            X_train_no_smote, y_train_no_smote,
            epochs=5,
            batch_size=32,
            validation_data=(X_val_no_smote, y_val_no_smote),
            callbacks=[checkpoint_no_smote]
        )

        # Track the best validation AUC-PR across folds without SMOTE
        fold_best_auc_pr_no_smote = max(history_no_smote.history['val_auc_pr'])
        if fold_best_auc_pr_no_smote > best_val_auc_pr_no_smote:
            best_val_auc_pr_no_smote = fold_best_auc_pr_no_smote

    # Load the best model for both SMOTE and no-SMOTE versions
    best_model_smote = load_model(BEST_MODEL_PATH_SMOTE)
    best_model_no_smote = load_model(BEST_MODEL_PATH_NO_SMOTE)

    # Generate predictions and probabilities for both models on the test data
    threshold = 0.9 # Threshold for binary predictions

    y_pred_smote = (best_model_smote.predict(X_test) > threshold).astype("int32").flatten()  # Binary predictions
    y_proba_smote = best_model_smote.predict(X_test).flatten()  # Probabilities

    y_pred_no_smote = (best_model_no_smote.predict(X_test) > threshold).astype("int32").flatten()  # Binary predictions
    y_proba_no_smote = best_model_no_smote.predict(X_test).flatten()  # Probabilities

    # Calculate metrics for SMOTE model
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba_smote)
    roc_auc_smote = roc_auc_score(y_test, y_proba_smote)
    pr_auc_smote = auc(recall, precision)
    accuracy_smote = accuracy_score(y_test, y_pred_smote)
    average_precision_smote = average_precision_score(y_test, y_proba_smote)

    # Calculate metrics for No-SMOTE model
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba_no_smote)
    roc_auc_no_smote = roc_auc_score(y_test, y_proba_no_smote)
    pr_auc_no_smote = auc(recall, precision)
    accuracy_no_smote = accuracy_score(y_test, y_pred_no_smote)
    average_precision_no_smote = average_precision_score(y_test, y_proba_no_smote)

    # Print comparison results
    print("Metrics Comparison on Test Data:")
    print("\nWith SMOTE:")
    print(f"AUC-ROC: {roc_auc_smote:.4f}")
    print(f"AUC-PR: {pr_auc_smote:.4f}")
    print(f"Accuracy: {accuracy_smote:.4f}")
    print(f"Average Precision Score: {average_precision_smote:.4f}")

    print("\nWithout SMOTE:")
    print(f"AUC-ROC: {roc_auc_no_smote:.4f}")
    print(f"AUC-PR: {pr_auc_no_smote:.4f}")
    print(f"Accuracy: {accuracy_no_smote:.4f}")
    print(f"Average Precision Score: {average_precision_no_smote:.4f}")
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model with specified input and labels.')
    parser.add_argument('--input', type=str, required=True, help='Path to the training data file (JSON gzipped).')
    parser.add_argument('--labels', type=str, required=True, help='Path to the labels file (CSV).')
    args = parser.parse_args()
    train_model(args.input, args.labels)
