import sys
import os
from src.exception import CustomException
from src.utils import load_object

import pandas as pd

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self, features):
        try:
            model_path = os.path.join('artifacts', 'model.joblib')
            preprocessor_path = os.path.join('artifacts', 'preprocessor.joblib')
            print('Artifacts Loading..........')
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            print('After Loading..........')
            features_preprocessed = preprocessor.transform(features)
            prediction = model.predict(features_preprocessed)
            print('Ticket Sales Predicted: ', prediction)
            return prediction
        except Exception as e:
            raise CustomException(e, sys)
        
class CustomData:
    def __init__(self,
                 film_code: int,
                 cinema_code: int,
                 show_time: int,
                 occu_perc: float,
                 ticket_price: float,
                 capacity: float,
                 day: int,
                 month: int,
                 year: int):
        
        self.film_code = film_code
        self.cinema_code = cinema_code
        self.show_time = show_time
        self.occu_perc = occu_perc
        self.ticket_price = ticket_price
        self.capacity = capacity
        self.day = day
        self.month = month
        self.year = year

    def get_data_as_df(self):
        try:
            input_dict = {
                'film_code': [self.film_code],
                'cinema_code': [self.cinema_code],
                'show_time': [self.show_time],
                'occu_perc': [self.occu_perc],
                'ticket_price': [self.ticket_price],
                'capacity': [self.capacity],
                'day': [self.day],
                'month': [self.month],
                'year': [self.year]
            }
            return pd.DataFrame(input_dict)
        except Exception as e:
            raise CustomException(e, sys)