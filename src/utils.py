import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.exceptions import CustomException
from src.logger import logging


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        logging.error(f"Error occurred while saving object to {file_path}: {e}")
        raise CustomException(e, sys)


def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
    except Exception as e:
        logging.error("Exception occurred during model evaluation")
        raise CustomException(e, sys)


import os
import pickle
import gdown
import sys
from src.exceptions import CustomException
from src.logger import logging


def download_file_from_drive(url, output_path):
    """
    Downloads a file from Google Drive and saves it locally.
    """
    try:
        logging.info(f"Downloading file from Google Drive URL: {url}")
        gdown.download(url, output_path, quiet=False)
        logging.info(f"File downloaded successfully and saved to: {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"Error downloading file from Google Drive: {e}")
        raise CustomException(e, sys)


def load_object(file_path_or_url):
    """
    Loads a pickled object either from a local file or a Google Drive URL.
    """
    try:
        if file_path_or_url.startswith("http"):
            # Assuming the URL is a Google Drive link
            output_path = "temp_file.pkl"  # Temporary local file to store the downloaded model
            download_file_from_drive(file_path_or_url, output_path)
            file_path = output_path
        else:
            file_path = file_path_or_url

        # Ensure the file exists before loading
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Load the pickled object
        with open(file_path, "rb") as file_obj:
            logging.info(f"Loading object from {file_path}")
            return pickle.load(file_obj)
    except Exception as e:
        logging.error(f"Error occurred while loading object: {e}")
        raise CustomException(e, sys)
