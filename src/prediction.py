import os
import argparse
from tensorflow.keras.models import load_model
from helper_functions import (
    load_data_to_dataframe,
    extract_mean_reads,
    load_scaler,
    scale_mean_reads,
    load_DRACH_encoder,
    extract_middle_sequence,
    one_hot_encode_DRACH,
    prepare_for_model
)

# Now that model training is done, fetch mean reads scaler, drach encoder and trained model from these paths
ARTIFACTS_FOLDER = "../artifacts"
MEAN_READS_SCALER_PATH = f"{ARTIFACTS_FOLDER}/mean_reads_scaler.pkl"
DRACH_ENCODER_PATH = f"{ARTIFACTS_FOLDER}/drach_encoder.pkl"
DEFAULT_MODEL_PATH = f"{ARTIFACTS_FOLDER}/trained_model.keras"


def make_predictions(input_path, output_path, model_path):
    # Load the trained model, scaler, and encoder
    model = load_model(model_path)
    scaler = load_scaler(MEAN_READS_SCALER_PATH)
    encoder = load_DRACH_encoder(DRACH_ENCODER_PATH)

    # Load and preprocess new data
    print(f"Loading data from {input_path}...")
    dataset = load_data_to_dataframe(input_path)
    print("Preprocessing data...")
    dataset = extract_mean_reads(dataset)
    scaler, dataset = scale_mean_reads(dataset, scaler)
    dataset = extract_middle_sequence(dataset)
    encoder, dataset = one_hot_encode_DRACH(dataset, encoder)

    # Prepare data for prediction
    X = prepare_for_model(dataset)
    print("Making predictions...")
    predictions = model.predict(X)

    # Create a DataFrame with results
    results_df = dataset[['transcript_id', 'position']].copy()
    results_df['score'] = predictions

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the results to a CSV file
    results_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    
    return results_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make predictions using the trained model.')
    parser.add_argument('--input', type=str, required=True, help='Path to the prediction data file (JSON gzipped).')
    parser.add_argument('--output', type=str, required=True, help='Path to save the predictions (CSV).')
    parser.add_argument('--model', type=str, default=None, help='Path to the trained model file (Keras).')
    args = parser.parse_args()

    # if no model path input, default to specified path
    if args.model is None:
        args.model = DEFAULT_MODEL_PATH
        print(f"Using default model path: {DEFAULT_MODEL_PATH}")
    
    make_predictions(args.input, args.output, args.model)
