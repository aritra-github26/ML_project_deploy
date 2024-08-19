import os
import sys
import numpy as np
import pandas as pd

from exception import CustomException
from logger import logging

from dataclasses import dataclass
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', 'data.csv')
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered Data Ingestion Component")
        try:
            pass # Continue from here, stay tuned


        except Exception as e:
            logging.error("Error in Data Ingestion Component")
            raise CustomException(e, sys)