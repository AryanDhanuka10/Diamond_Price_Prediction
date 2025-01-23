import os  # for creating path
import sys  # for logging and exceptions
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exceptions import CustomException
from src.logger import logging


## Initialize the data ingestion configuration
@dataclass
class DataIngestionconfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw.csv")


# create the data ingestion class
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionconfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion method starts")

        try:
            df = pd.read_csv(os.path.join("notebooks/data", "gemstone.csv"))
            logging.info("Dataset read as pandas Dataframe")

            # saving the dataframe dataset into raw data
            os.makedirs(
                os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True
            )
            df.to_csv(self.ingestion_config.raw_data_path)

            logging.info("Raw Data Is Created")

            train_set, test_set = train_test_split(df, test_size=0.30, random_state=42)
            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )
            test_set.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )
            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            logging.info("Exception Occured at Data Ingestion Stage")
            raise CustomException(e, sys)
