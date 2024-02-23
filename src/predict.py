import pickle
import datetime
import logging
import pandas as pd
import numpy as np


def make_predictions(X_test):

    # Set up the logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Add a handler to log messages to a file
    try: 
        file_handler = logging.FileHandler('../logs/predict.txt')
    except FileNotFoundError:
        file_handler = logging.FileHandler('logs/predict.txt')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Load the binary classifiers
    try:
        with open("./models/lgbm_classifier.pickle", "rb") as f:
            clf = pickle.load(f)
    except FileNotFoundError:
        with open("models/lgbm_classifier.pickle", "rb") as f:
            clf = pickle.load(f)

    # Use the trained classifier to predict the class for each example
    final_predictions = clf.predict(X_test) 

    # format predictions to be more user friendly
    final_predictions = [(int(i) + 1)*20  for i in final_predictions]

    for i, pred in enumerate(final_predictions):
        logger.info(f"Prediction for sample {i}: {pred}")

    return final_predictions


if __name__ == "__main__":
    df = pd.read_csv('../data/filtered_data.csv' , index_col=None, header=0, lineterminator='\n')
    X = df.drop(['price_label', 'average_price', 'avg_selling_price'],axis=1)
    make_predictions(X.sample(frac=0.1))
