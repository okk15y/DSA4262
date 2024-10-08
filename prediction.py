import gzip
import json
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

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

def data_processing(dataset):
    transcriptID_position = dataset[['transcript_id', 'position']]
    dataset['data'] = dataset['data'].apply(lambda x: np.mean(x, axis=0))
    dataset = np.vstack(dataset["data"].values)
    return dataset, transcriptID_position

def predict(dataset, transcriptID_position, model):
    predictions = model.predict(dataset)
    predictions_df = pd.DataFrame(predictions, columns=['score'])
    predictions_df = pd.concat([transcriptID_position, predictions_df], axis=1)
    predictions_df.rename(columns={'position': 'transcript_position'}, inplace=True)
    predictions_df.to_csv('predictions.csv', index=False)

def main(file_path):
    # change this to name of model
    model = load_model('final_model.keras')
    
    input_data = load_json_gz_to_dataframe(file_path)
    input_data, transcriptID_position = data_processing(input_data)
    
    predict(input_data, transcriptID_position, model)

if __name__ == "__main__":
    input_file = input("File Path of json.gz dataset: ")
    main(input_file)
