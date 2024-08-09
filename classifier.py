import pandas as pd
from classification import Classification
from sklearn.feature_selection import VarianceThreshold
import json
import pickle

class Classifier:
    
    testing = 'testing'
    ground_truth = 'ground truth'
    general = 'general'
    
    def __init__(self, validation_method):
        try:
            with open('data/hyper_parameters.json') as file:
                self.hyper_parameters = json.load(file)
        except:
            print('counldnt find hyper parameters')
    
    
        x_train = pd.read_pickle('data/ml_projects/x_train.df')
        y_train = pd.read_pickle('data/ml_projects/y_train.df')
        
        # Select the testing set based on the validation method.
        if validation_method == Classifier.testing:
            x_test = pd.read_pickle('data/ml_projects/x_test.df')
            y_test = pd.read_pickle('data/ml_projects/y_test.df')
        if validation_method == Classifier.ground_truth:
            x_test = pd.read_pickle('data/ground_truth/x_test.df')
            y_test = pd.read_pickle('data/ground_truth/y_test.df')
        if validation_method == 'general':
            x_test = pd.read_pickle('data/general_projects/x_test.df')
            y_test = pd.read_pickle('data/general_projects/y_test.df')
        
        
        correlated = ['files', 'CountClassBase', 'CountDeclExecutableUnit', 'CountLine', 'CountLineCode', 'CountLineCodeExe', 'CountStmt', 'CountStmtDecl', 'CountStmtExe']
        # Drop the columns
        x_test = x_test.drop(columns=correlated)
        x_train = x_train.drop(columns=correlated)

        # Apply vaiance threshold, we have reduced the features using a rough variance threshold in pre_processing, we apply the final cut-off
        selector = VarianceThreshold(0.001)
        selector.fit(x_train)
        x_train = x_train[x_train.columns[selector.get_support(indices=True)]]
        x_test = x_test[x_test.columns[selector.get_support(indices=True)]]
              
        # Normalization of process and code metrics
        for column in x_train.columns.values[-25:]:
            normalized = Classifier.minmax_normalize(x_train[column], x_test[column])
            x_train[column] = normalized[0]
            x_test[column] = normalized[1]
        
        self.x_train = x_train.reset_index(drop=True)
        self.y_train = y_train.reset_index(drop=True)
        self.x_test = x_test.reset_index(drop=True)
        self.y_test = y_test.reset_index(drop=True)


    # define the min-max normalization function
    @staticmethod
    def minmax_normalize(x_train, x_test):
        col_min = x_train.min()  # get minimum value
        col_max = x_train.max()  # get maximum value
        x_train_normalized = (x_train - col_min) / (col_max - col_min)  # compute normalized values
        x_test_normalized = (x_test - col_min) / (col_max - col_min)  # compute normalized values

        return x_train_normalized, x_test_normalized

    def run(self, classifier_name=False):
        """
            This method runs the classifier based on the given input, the names of the classifiers can be accesed at Classification.names
        """
        if classifier_name and (classifier_name in self.hyper_parameters):
            print('running for: ', classifier_name)
            classification = Classification(self.x_train, self.y_train, self.x_test, self.y_test, True)
            function = getattr(classification, classifier_name)
            model = function(False, self.hyper_parameters[classifier_name])
            print(model)
        else:
            print("Provide Classifier Name or hyper parameters")
            return False

