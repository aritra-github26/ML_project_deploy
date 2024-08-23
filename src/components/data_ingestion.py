import os
import sys
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging

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
            df = pd.read_csv(os.path.join('notebooks', 'data', 'cinemaTicket_Ref.csv')) # Dataset Loading from the source
            logging.info("Read Dataset")
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Making some necessary changes for this dataset only :)
            df[['date']] = df[['date']].apply(pd.to_datetime)
            df = df.dropna()
            df.drop_duplicates(inplace=True)
            df = df.drop(columns= ['total_sales','tickets_out','month','quarter','day','ticket_use'])
            df[['tickets_sold','show_time']] = df[['tickets_sold','show_time']].astype(int)
            df['day'] = df['date'].dt.day.astype(int)
            df['month'] = df['date'].dt.month.astype(int)
            df['year'] = df['date'].dt.year.astype(int)
            df = df.drop(columns= ['date'])
            
            
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Split Dataset into Train and Test")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info("Ingestion Complete")
            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path


        except Exception as e:
            logging.error("Error in Data Ingestion Component")
            raise CustomException(e, sys)
        
