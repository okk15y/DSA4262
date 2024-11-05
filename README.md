# DSA4262 Project
This repository contains the code and data for the DSA4262 project from FlexiProtein. The project involves training and evaluating machine learning models to predict m6A sites.

![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)

## Project Overview

The DSA4262 project aims to develop and evaluate machine learning models for predicting m6A sites in RNA sequences. m6A is a common and important RNA modification that plays a crucial role in various biological processes. Accurate prediction of m6A sites can provide valuable insights into RNA biology and disease mechanisms.


### Key Files and Directories
```
DSA4262/
├── artifacts/
│   ├── drach_encoder.pkl
│   ├── mean_reads_scaler.pkl
│   ├── trained_model.keras
│   └── temp_best_model_fold.keras
├── data/
│   ├── train_data.json.gz
│   ├── train_labels.csv
|   └── prediction_data.json
├── src/
│   ├── helper_functions.py
│   ├── model_training.py
│   ├── prediction.py
│   ├── Task1.ipynb
│   └── Task2.ipynb
├── .gitignore
├── README.md
└── requirements.txt
```

- [`artifacts/`](./artifacts/): Contains saved models, encoders and scalers.
  - `temp_best_model_fold.keras`: Log the best model for each fold during training.
  - `drach_encoder.pkl`: Encoder used for data preprocessing.
  - `mean_reads_scaler.pkl`: Scaler used for data preprocessing.
  - `trained_model.keras`: Pre-trained model.
  
- [`data/`](./data/):
  - `train_data.json.gz`: Sample training data
  - `train_labels.csv`: Sample training labels
  - `prediction_data.json`: Sample prediction data

- [`src/`](.src/): Contains source code files.
  - `helper_functions.py`: Contains helper functions used across the project.
  - `model_training.py`: Script for model training.
  - `prediction.py`: Script for generating predictions.


## Getting Started

### Prerequisites

- Python 3.10
- `gzip` and `json` modules (part of the Python Standard Library)

### Installation

1. Set up an AWS Ubuntu Instance (recommended t3.medium)

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
    ```

    ```sh
    sudo apt install python3-pip gzip 
    ```
2. Create and activate the virtual environment
    ```sh
    sudo pip3 install virtualenv
    virtualenv --python="/usr/bin/python3.10" test
    source test/bin/activate
    ```

3. Clone the repository
    ```sh
    git clone https://github.com/okk15y/DSA4262.git
    ```

4. Navigate into the src folder
    ```sh
    cd DSA4262/src
    ```

5. Install the required packages
    ```sh
    pip install -r ../requirements.txt
    ```

## Usage

Before training the model and generating the predictions, ensure that you have imported the appropriate data in `.gz` or `.json` file format. Sample data is also provided in the repo in the [`data`](./data/) folder. Starting in the src folder, 

1. (Optional) Train the model:
    
    A pre-trained model ([`trained_model.keras`](./artifacts/trained_model.keras)) is provided in the [`artifacts`](./artifacts/) directory. You can skip this step if you wish to use the provided model.
    ```sh
    python model_training.py --input ../data/train_data.json.gz --labels ../data/train_labels.csv
    ```
2. Generate predictions:
    ```sh
    python prediction.py --input ../data/prediction_data.json --output ../data/predictions/predictions.csv --model ../artifacts/trained_model.keras
    ```

## Authors

- Chong Jing Ren
- Liew Jia Xuan, Celine
- Ong Kok Kiong
- Quek Hong Lin

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
