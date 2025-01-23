import os
import sys

import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.exceptions import CustomException
from src.logger import logging

if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    print(train_data_path, test_data_path)
