import logging, pickle 
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from imblearn.over_sampling import ADASYN


def train():

    # Set up the logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Add a handler to log messages to a file
    file_handler = logging.FileHandler('../logs/train.txt')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Read data
    df = pd.read_csv('../data/filtered_data.csv' , index_col=None, header=0, lineterminator='\n')

    # Define the target
    X = df.drop(['price_label', 'average_price', 'avg_selling_price'],axis=1)
    y = df.price_label

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)  

    # Oversampling of minority class
    adasyn = ADASYN(sampling_strategy='minority', random_state=0)
    X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)
    logger.info("New target distribution after resampling of class 4: {}".format(y_resampled.value_counts()))
    

    clf = lgb.LGBMClassifier()
        
    # Random Search for Hyperparameters
    param_grid = {
        'boosting_type': ['gbdt'],
        'objective': ['binary'],
        'num_leaves': [32, 64, 128, 256],
        'learning_rate': [0.05, 0.1, 0.15, 0.2, 0.5],
        'min_child_samples': [10, 20, 30, 40],
        'n_estimators': [50, 100, 200, 500, 750],
        'verbose': [-1]
        }

    clf_random = RandomizedSearchCV(estimator = clf, param_distributions = param_grid, n_iter = 30, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    clf_random.fit(X_resampled, y_resampled)
    logger.info(f'Best params for lgbm classifier: {clf_random.best_params_}')

 
    # Get the best hyperparameters
    best_clf = clf_random.best_estimator_

    # Predict the target using the best classifier
    y_pred = best_clf.predict(X_test)

    # Calculate the evaluation metrics 
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    # Log the evaluation metrics
    logger.info("Accuracy: %f", accuracy)
    logger.info("Precision: %f", precision)
    logger.info("Recall: %f", recall)
    logger.info("F1 Score: %f", f1)

    # Log classification report
    logger.info("Classification Report:\n\n %s", classification_report(y_test , y_pred, digits=3))


    # Save the binary classifiers
    with open("../models/lgbm_classifier.pickle", "wb") as f:
        pickle.dump(best_clf, f)


if __name__ == "__main__":
    train()
