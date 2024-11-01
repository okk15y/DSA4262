import gzip
import json
import joblib
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint


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


def load_labels(file_path):
    return pd.read_csv(file_path)


def data_processing(dataset):
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

    # Save standardize scale
    joblib.dump(scaler, 'base_training_model_scaler.pkl')

    return dataset


def combine_data(dataset, labels):
    merged_df = pd.merge(dataset, labels,
                     left_on  = ['transcript_id', 'position'],
                     right_on = ['transcript_id', 'transcript_position'],
                     how = 'left')
    tmp = merged_df['gene_id']
    merged_df = merged_df.drop(columns=['transcript_position', 'gene_id'])
    merged_df.insert(0, 'gene_id', tmp)
    
    return merged_df


def X_y_train(merged_df):
    X_train = merged_df.drop(columns=['gene_id', 'transcript_id', 'position', 'label'])
    X_train = np.vstack(X_train["data"].values)

    y_train = merged_df[['label']]

    return X_train, y_train


def model_building(X_train, y_train):
    # Create a sequential model
    model = Sequential()

    # Input layer
    model.add(Input(shape=(X_train.shape[1],)))

    # First hidden layer
    model.add(Dense(150, activation='relu'))  # First hidden layer
    model.add(Dropout(0.2))  # Optional: dropout layer for regularization

    # Second hidden layer
    model.add(Dense(32, activation='relu'))

    # Output layer (binary classification)
    model.add(Dense(1, activation='sigmoid'))  # Sigmoid activation for binary output

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    checkpoint = ModelCheckpoint('base_model.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

    # Fit the model to the training data
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[checkpoint])  # Adjust epochs and batch_size as needed

    # After fitting, you can save the model if required
    model.save('base_model.keras')  


def main(data_file, labels_file):
    dataset = load_json_gz_to_dataframe(data_file)
    labels = load_labels(labels_file)

    dataset = data_processing(dataset)
    print("Processed Dataset:\n", dataset.head(5))
    merged_df = combine_data(dataset, labels)
    print("Full Dataset:\n", merged_df.head(5))

    X_test, y_test = X_y_train(merged_df)
    print("X Data:\n", X_test[:5])
    print("y Data:\n", y_test[:5])
    
    model_building(X_test, y_test)


if __name__ == "__main__":
    data_file = input("File Path of json.gz dataset: ")
    labels_file = input("File Path of m6A labels: ")
    main(data_file, labels_file)
