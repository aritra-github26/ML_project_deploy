import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        logging.info("Saving object")
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as f:
            joblib.dump(obj, f)
        logging.info("Object saved")
    except Exception as e:
        logging.error("Error in saving object")
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            report[list(models.keys())[i]] = r2

        return report

    except Exception as e:
        logging.error("Error in model evaluation")
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return joblib.load(file_obj)
    except Exception as e:
        logging.error("Error in loading object")
        raise CustomException(e, sys)