import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from helper_functions import (
    load_json_gz_to_dataframe,
    extract_mean_data,
    load_scaler,
    scale_mean_data,
    one_hot_encode_DRACH,
    prepare_for_model,
    load_DRACH_encoder
)

def make_predictions(input_path, output_path, model_path='trained_model.keras'):
    # Load the trained model, scaler, and encoder
    model = load_model(model_path)
    scaler = load_scaler('mean_data_scaler.pkl')
    encoder = load_DRACH_encoder('drach_encoder.pkl')

    # Load and preprocess new data
    dataset = load_json_gz_to_dataframe(input_path)
    dataset = extract_mean_data(dataset)
    dataset = scale_mean_data(dataset, scaler)
    dataset = one_hot_encode_DRACH(dataset, encoder)

    # Prepare data for prediction
    X = prepare_for_model(dataset)
    predictions = model.predict(X)

    # Create a DataFrame with results
    results_df = dataset[['transcript_id', 'position']].copy()
    results_df['prediction'] = predictions
    results_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    input_path = input("Enter path to prediction data: ")
    output_path = input("Enter path to save predictions: ")
    model_path = input("Enter path to trained model: ")
    make_predictions(input_path, output_path)
