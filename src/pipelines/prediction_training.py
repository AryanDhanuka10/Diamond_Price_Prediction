import sys
import os
import pandas as pd
import pickle
import gdown
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
            # If it's a URL, download the file
            output_path = "temp_file.pkl"  # Temporary file to store the downloaded content
            download_file_from_drive(file_path_or_url, output_path)
            file_path = output_path
        else:
            file_path = file_path_or_url

        # Check if the file exists before loading
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Load the pickled object
        with open(file_path, "rb") as file_obj:
            logging.info(f"Loading object from {file_path}")
            return pickle.load(file_obj)
    except Exception as e:
        logging.error(f"Error occurred while loading object: {e}")
        raise CustomException(e, sys)


class PredictPipeline:
    def __init__(self):
        # URLs for the preprocessor and model files
        self.preprocessor_url = "https://drive.google.com/uc?id=1GtVUx45rpiHgtTzFKRjuJY-IY02GGRqC"
        self.model_url = "https://drive.google.com/uc?id=11OujRjqt161MqpWwS8Q8mU9oFHHbqLp2"

    def predict(self, features):
        try:
            # Load preprocessor and model from URLs
            preprocessor = load_object(self.preprocessor_url)
            model = load_object(self.model_url)

            # Preprocess features and make predictions
            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred

        except Exception as e:
            logging.info("Exception occurred in prediction")
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        carat: float,
        depth: float,
        table: float,
        x: float,
        y: float,
        z: float,
        cut: str,
        color: str,
        clarity: str,
    ):
        self.carat = carat
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z
        self.cut = cut
        self.color = color
        self.clarity = clarity

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "carat": [self.carat],
                "depth": [self.depth],
                "table": [self.table],
                "x": [self.x],
                "y": [self.y],
                "z": [self.z],
                "cut": [self.cut],
                "color": [self.color],
                "clarity": [self.clarity],
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info("Dataframe Gathered")
            return df
        except Exception as e:
            logging.info("Exception Occurred in prediction pipeline")
            raise CustomException(e, sys)
