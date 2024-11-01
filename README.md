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

## Getting Started

### Prerequisites

- Python 3.9.9 (i anyhow one this)
- Required Python packages (listed in `requirements.txt`)

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/okk15y/DSA4262.git
    ```
2. Install the required packages:
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