from classifier import Classifier
import pandas as pd
from classification import Classification
from sklearn.feature_selection import VarianceThreshold
import json
import pickle
import lightgbm
import numpy as np

class EnsembleFusion:
    
    def __init__(self):
        try:
            with open('data/hyper_parameters.json') as file:
                hyper_parameters = json.load(file)
        except:
            print('counldnt find hyper parameters')
    
        x_train = pd.read_pickle('data/ml_projects/x_train.df')
        y_train = pd.read_pickle('data/ml_projects/y_train.df')
        
        x_test = pd.read_pickle('data/ml_projects/x_test.df')
        x_test = x_test.drop(columns=['repository', 'sha'])
        y_test = pd.read_pickle('data/ml_projects/y_test.df')
        # Fetch Resutls of the PyRef
        self.pyRef = pd.read_pickle('data/ml_projects/test_data.df')['has_refactoring'].to_numpy()
        # Fetch the labels
        self.y_labels = pd.read_pickle('data/ml_projects/y_test.df')['y_label'].to_numpy()

        # np.save('data/voting/PyRef.npy', x_test_pyref)
        # np.save('data/voting/y_label.npy', x_test_labels)



        # Prepare and train the model
        correlated = ['files', 'CountClassBase', 'CountDeclExecutableUnit', 'CountLine', 'CountLineCode', 'CountLineCodeExe', 'CountStmt', 'CountStmtDecl', 'CountStmtExe']
        # Drop the columns
        x_test = x_test.drop(columns=correlated)
        x_train = x_train.drop(columns=correlated)

        selector = VarianceThreshold(0.001)
        selector.fit(x_train)
        new_x_train = x_train[x_train.columns[selector.get_support(indices=True)]]
        new_x_test = x_test[x_test.columns[selector.get_support(indices=True)]]
        
        x_train = new_x_train
        y_train = y_train
        x_test = new_x_test
        
        for column in x_train.columns.values[-25:]:
            normalized = Classifier.minmax_normalize(x_train[column], x_test[column])
            x_train[column] = normalized[0]
            x_test[column] = normalized[1]
        
        
        x_train = x_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        x_test = x_test.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        
        y_train = y_train.values.ravel()
        y_test = y_test.values.ravel()
        
        train_data = lightgbm.Dataset(x_train, label=y_train)
        test_data = lightgbm.Dataset(x_test, label=y_test)
        model = lightgbm.train(hyper_parameters['lightgbm'], train_data)
        y_pred = model.predict(x_test)
        # Convert probabilities to binary predictions
        self.PRefScanner = np.where(y_pred > 0.5, 1, 0)
        
        print(Classification.static_scores(y_test, self.PRefScanner))
        
    def compare(self):

        # Perform ensemble voting using majority vote
        majority_predictions = []
        unanimous_predictions = []

        for i in range(len(self.y_labels)):
            # Get the predictions from each model for the current sample
            prediction_PRefScanner = self.PRefScanner[i]
            prediction_pyRef = self.pyRef[i]

            # Perform majority voting to get the ensemble prediction
            majority_vote = int((prediction_PRefScanner + prediction_pyRef) >= 1)
            unanimous_vote = int((prediction_PRefScanner + prediction_pyRef) >= 2)
            majority_predictions.append(majority_vote)
            unanimous_predictions.append(unanimous_vote)

        # Convert the ensemble_predictions list to a NumPy array
        majority_predictions = np.array(majority_predictions)
        unanimous_predictions = np.array(unanimous_predictions)

        print('PyRef', Classification.static_scores(self.y_labels, self.pyRef))
        print('PRefScanner', Classification.static_scores(self.y_labels, self.PRefScanner))
        print('Unanimous Voting', Classification.static_scores(self.y_labels, unanimous_predictions))
        print('Majority Voting', Classification.static_scores(self.y_labels, majority_predictions))
