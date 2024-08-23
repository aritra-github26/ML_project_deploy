import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
import numpy as np
import dill

def save_object(file_path, obj):
    try:
        logging.info("Saving object")
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as f:
            dill.dump(obj, f)
        logging.info("Object saved")
    except Exception as e:
        logging.error("Error in saving object")
        raise CustomException(e, sys)