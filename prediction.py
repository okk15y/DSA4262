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

def main(file_path):
    # change this to name of model
    model = load_model('final_model.keras')
    
    input_data = load_json_gz_to_dataframe(file_path)
    input_data['data'] = input_data['data'].apply(lambda x: np.mean(x, axis=0))
    
    predictions = model.predict(input_data)
    predictions.to_csv('predictions.csv', index=False)

if __name__ == "__main__":
    input_file = input("File Path of json.gz dataset: ")
    main(input_file)
