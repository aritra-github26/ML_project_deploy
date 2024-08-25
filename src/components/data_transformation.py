import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

@dataclass
class DataTransformationConfig:
    preprocessor_obj_path = os.path.join('artifacts', 'preprocessor.joblib')

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_transformation_obj(self):
        try:
            num_cols = ['film_code', 'cinema_code', 'show_time', 'occu_perc',
       'ticket_price', 'capacity', 'day', 'month', 'year']
            num_pipeline = Pipeline(steps=[
                ('std_scaler', StandardScaler())
            ])

            # do this for categorical columns too, if any

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, num_cols)
            ])
            logging.info('Pipeline created')
            return preprocessor
        
        except Exception as e:
            logging.error("Error in getting transformation object")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            logging.info("Read Train and Test Data")

            # Data Transformation Logic
    #         num_cols = ['film_code', 'cinema_code', 'show_time', 'occu_perc',
    #    'ticket_price', 'capacity', 'day', 'month', 'year']
            preprocessor = self.get_transformation_obj()
            target_col_name = 'tickets_sold'

            input_features_train = train_df.drop(columns=target_col_name, axis=1)
            target_feature_train = train_df[target_col_name]
            
            input_features_test = test_df.drop(columns=target_col_name)
            target_feature_test = test_df[target_col_name]

            logging.info("Applying Preprocessing")
            input_features_train_arr = preprocessor.fit_transform(input_features_train)
            input_features_test_arr = preprocessor.transform(input_features_test)

            train_arr = np.c_[input_features_train_arr, np.array(target_feature_train)]
            test_arr = np.c_[input_features_test_arr, np.array(target_feature_test)]

            save_object(
                file_path=self.transformation_config.preprocessor_obj_path,
                obj=preprocessor
            )
            logging.info("Transformation Complete")

            return (
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_obj_path
            )

        except Exception as e:
            logging.error("Error in Data Transformation Component")
            raise CustomException(e, sys)