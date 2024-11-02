# DSA4262 Project
This repository contains the code and data for the DSA4262 project from FlexiProtein. The project involves training and evaluating machine learning models to predict m6A sites.

![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)

## Project Overview

The DSA4262 project aims to develop and evaluate machine learning models for predicting m6A sites in RNA sequences. m6A is a common and important RNA modification that plays a crucial role in various biological processes. Accurate prediction of m6A sites can provide valuable insights into RNA biology and disease mechanisms.


### Key Files and Directories

- `best_model_fold_*.keras`: Saved models for different folds.
- `data.info.labelled`: Labelled data information.
- `drach_encoder.pkl`: Encoder used for data preprocessing.
- `helper_functions.py`: Contains helper functions used across the project.
- `mean_data_scaler.pkl`: Scaler used for normalizing data.
- `model_training_LATEST.py`: Script for model training.
- `prediction_LATEST.py`: Script for generating predictions.
- `dataset0.json.gz`: Data used for training
- `dataset1.json.gz`: Sample data used for predictions

## Getting Started

### Prerequisites

- Python 3.10
- `gzip` and `json` modules (part of the Python Standard Library)

### Installation

1. Download the Specific Python Version (3.10) and pip
     ```sh
    sudo apt update
    sudo apt install software-properties-common -y
    ```
    
    ```sh
    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt-get update
    ```

    ```sh
    sudo apt-get install python3.10 python3.10-venv python3.10-dev
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

## Usage

Before training the model and generating the predictions, ensure that you have imported the appropriate data in `.gz` file format. / or we just provide sample data in the repo

1. Train the model:
    ```sh
    python model_training.py --input train_data.json.gz --labels train_labels.info.labelled
    ```
2. Generate predictions:
    ```sh
    python prediction.py --input prediction_data.json.gz --output predictions --model trained_model.keras
    ```

## Authors

- 

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.