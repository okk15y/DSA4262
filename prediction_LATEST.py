import argparse
from tensorflow.keras.models import load_model
from helper_functions import (
    load_json_gz_to_dataframe,
    extract_mean_reads,
    load_scaler,
    scale_mean_reads,
    load_DRACH_encoder,
    one_hot_encode_DRACH,
    prepare_for_model
)


def make_predictions(input_path, output_path, model_path='trained_model.keras'):
    # Load the trained model, scaler, and encoder
    model = load_model(model_path)
    scaler = load_scaler('mean_reads_scaler.pkl')
    encoder = load_DRACH_encoder('drach_encoder.pkl')

    # Load and preprocess new data
    print(f"Loading data from {input_path}...")
    dataset = load_json_gz_to_dataframe(input_path)
    print("Preprocessing data...")
    dataset = extract_mean_reads(dataset)
    dataset = scale_mean_reads(dataset, scaler)
    dataset = one_hot_encode_DRACH(dataset, encoder)

    # Prepare data for prediction
    X = prepare_for_model(dataset)
    print("Making predictions...")
    predictions = model.predict(X)

    # Create a DataFrame with results
    results_df = dataset[['transcript_id', 'position']].copy()
    results_df['prediction'] = predictions
    results_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make predictions using the trained model.')
    parser.add_argument('--input', type=str, required=True, help='Path to the prediction data file (JSON gzipped).')
    parser.add_argument('--output', type=str, required=True, help='Path to save the predictions (CSV).')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model file (Keras).')
    args = parser.parse_args()
    make_predictions(args.input, args.output, args.model)