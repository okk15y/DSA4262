# DSA4262 Project
This repository contains the code and data for the DSA4262 project from FlexiProtein. The project involves training and evaluating machine learning models to predict m6A sites.

### Key Files and Directories

- `best_model_fold_*.keras`: Saved models for different folds.
- `data.info.labelled`: Labelled data information.
- `drach_encoder.pkl`: Encoder used for data preprocessing.
- `helper_functions.py`: Contains helper functions used across the project.
- `mean_data_scaler.pkl`: Scaler used for normalizing data.
- `model_training_LATEST.py`: Script for model training.
- `prediction_LATEST.py`: Script for generating predictions.
- `trained_model.keras`: Final trained model.
- `dataset0.json.gz`: Data used for training

## Getting Started

### Prerequisites (actually might not need this as installation will be part of their setup)

- Python 3.10 (i anyhow one this)
- Required Python packages (listed in `requirements.txt`)

### Installation

1. Download the Specific Python Version (3.10) and pip
     ```sh
    sudo apt-get update
    sudo apt-get install python3.10
    sudo apt install python3-pip gzip 
    ```
2. Create and activate the virtual environment
    ```sh
    sudo pip3 install virtualenv
    virtualenv --python="/usr/bin/python3.10" test
    source test/bin/activate
    ```

3. Clone the repository:
    ```sh
    git clone https://github.com/okk15y/DSA4262.git
    ```

4. Change into the DSA4262 Folder
    ```sh
    cd DSA4262
    ```

5. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

### Usage

Before training the model and generating the predictions, ensure that you have imported the appropriate data in `.gz` file format.

1. Train the model:
    ```sh
    python model_training.py --input train_data.json.gz --labels train_labels.csv
    ```
2. Generate predictions:
    ```sh
    python prediction.py --input prediction_data.json.gz --output predictions --model trained_model.keras
    ```

### Authors

- 

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.