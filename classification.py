import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import lightgbm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from os import path
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from sklearn.svm import LinearSVC

class Classification:

    names = ['lightgbm', 'random_forest', 'AdaBoost', 'xgboost', 'catboost', 'complement_nb', 'k_neighbors', 'MLPClassifier', 'support_vector_machine', 'C4_5', 'GBMClassifier']

    def __init__(self, x_train, y_train, x_test, y_test, reval = True):
        self.x_train = x_train
        self.x_test = x_test
        if reval:
            self.y_train = y_train.values.ravel()
            self.y_test = y_test.values.ravel()
        else:
            self.y_train = y_train
            self.y_test = y_test

    def scores(self, predictions, average='binary'):
        results = {
            'Accuracy': accuracy_score(self.y_test, predictions),
            'Precision': precision_score(self.y_test, predictions, average=average),
            'Recall': recall_score(self.y_test, predictions, average=average),
            'AUC': roc_auc_score(self.y_test, predictions),
        }
        results['F1'] = (2 * (results['Precision'] * results['Recall']) / (results['Precision'] + results['Recall']))
        return results
    
    @staticmethod
    def static_scores(y_test, predictions, average='binary'):
        results = {
            'Accuracy': accuracy_score(y_test, predictions),
            'Precision': precision_score(y_test, predictions, average=average),
            'Recall': recall_score(y_test, predictions, average=average),
            'AUC': roc_auc_score(y_test, predictions),
        }
        results['F1'] = (2 * (results['Precision'] * results['Recall']) / (results['Precision'] + results['Recall']))
        return results

    def random_forest(self, tuning=False, params=False):
        if not tuning:
            param_dist = {}
            if params:
                print(params)
                param_dist = params
            print('Random Forest')
            model = RandomForestClassifier(**param_dist, n_jobs=-1)
            model.fit(self.x_train, self.y_train)
            y_pred = model.predict(self.x_test)
            predictions = [round(value) for value in y_pred]
            return model, self.scores(predictions)
        else:
            print('RandomizedSearchCV Hyper Parameter Tuning')
            param_distributions = {
                'n_estimators': randint(50, 200),
                'max_depth': randint(3, 7),
                'min_samples_split': randint(2, 10),
                'criterion': ["gini", "entropy", "log_loss"],
                'min_samples_leaf': randint(1, 5),
                'max_features': ['sqrt', 'log2'],
                'bootstrap': [True, False]
            }
            rf_model = RandomForestClassifier(n_jobs=-1)
            search = RandomizedSearchCV(estimator=rf_model, param_distributions=param_distributions, cv=3, n_jobs=-1)
            search.fit(self.x_train, self.y_train)
            return dict(search.best_params_)
    
    def AdaBoost(self, tuning=False, params=False):
        if not tuning:
            print('adaboost')
            param_dist = {}
            if params:
                print(params)
                param_dist = params
            model = AdaBoostClassifier(**param_dist)
            model.fit(self.x_train, self.y_train)
            y_pred = model.predict(self.x_test)
            predictions = [round(value) for value in y_pred]
            return model, self.scores(predictions)
        else: 
            print('AdaBoostClassifier Hyper Parameter Tuning')
            param_distributions = {
                'n_estimators': randint(50, 200),
                'learning_rate': uniform(0.01, 2),
                'algorithm': ['SAMME', 'SAMME.R'],
                'random_state': randint(0, 100)
            }
            ada_model = AdaBoostClassifier()
            search = RandomizedSearchCV(estimator=ada_model, param_distributions=param_distributions, cv=3, n_jobs=-1)
            search.fit(self.x_train, self.y_train)
            best_params = search.best_params_
            return dict(search.best_params_)

    def xgboost(self, tuning=False, params=False):
        if not tuning:
            print("Xgboost")
            param_dist = {}
            if params:
                print(params)
                param_dist = params
            model = XGBClassifier(**param_dist, n_jobs=-1)
            model.fit(self.x_train, self.y_train)
            y_pred = model.predict(self.x_test)
            predictions = [round(value) for value in y_pred]
            return model, self.scores(predictions)
        else:
            print('XGBoost Hyper Parameter Tuning')
            param_distributions = {
                'n_estimators': randint(50, 200),
                'learning_rate': uniform(0.01, 1),
                'max_depth': randint(3, 7),
                'subsample': uniform(0.5, 0.9),
                'min_child_weight': randint(1, 10),
                'gamma': uniform(0, 1),
                'colsample_bytree': uniform(0.5, 0.9),
                'reg_alpha': uniform(0, 1)
            }
            xgb_model = XGBClassifier()
            search = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_distributions, cv=3, n_jobs=-1)
            search.fit(self.x_train, self.y_train)
            best_params = search.best_params_
            print('XGBoost model trained.')
            return dict(search.best_params_)
        
    def lightgbm(self, tuning=False, params=False):

        if not tuning:
            param_dist = {}
            if params:
                print('params')
                param_dist = params
                
            print("lightGBM")
            train_data = lightgbm.Dataset(self.x_train, label=self.y_train)
            test_data = lightgbm.Dataset(self.x_test, label=self.y_test)
            model = lightgbm.train(param_dist, train_data)
            y_pred = model.predict(self.x_test)
            # Convert probabilities to binary predictions
            y_pred_binary = np.where(y_pred > 0.5, 1, 0)
            print(y_pred_binary)
            predictions = [round(value) for value in y_pred]
            return model, self.scores(y_pred_binary)
        else:
            print('LightGBM Hyper Parameter Tuning')
            param_distributions = {
                'learning_rate': uniform(0.01, 0.99),  # Updated range: [0.01, 1 - 0.01]
                'n_estimators': randint(50, 200),
                'max_depth': randint(3, 7),
                'num_leaves': randint(20, 100),
                'bagging_fraction': uniform(0.1, 0.9),  # Updated range: [0.1, 1 - 0.1]
                'min_child_samples': randint(10, 50),
                # 'subsample': uniform(0.6, 0.3),  # Updated range: [0.6, 0.6 + 0.3]
                'colsample_bytree': uniform(0.6, 0.3),  # Updated range: [0.6, 0.6 + 0.3]
                'reg_lambda': uniform(0, 1),
                'reg_alpha': uniform(0, 1)
            }
            lgb_model = lightgbm.LGBMClassifier()
            search = RandomizedSearchCV(estimator=lgb_model, param_distributions=param_distributions, cv=3, n_jobs=-1)
                
            search.fit(self.x_train, self.y_train)
            return dict(search.best_params_)
    
    def catboost(self, tuning=False, params=False):
        if not tuning:
            param_dist = {}
            if params:
                print(params)
                param_dist = params
            print("CatBoost")
            model = CatBoostClassifier(**param_dist)
            model.fit(self.x_train, self.y_train)
            y_pred = model.predict(self.x_test)
            predictions = [round(value) for value in y_pred]
            return model, self.scores(predictions)
        else:
            print('CatBoost Hyper Parameter Tuning')
            param_distributions = {
                'learning_rate': uniform(0.01, 1),
                'n_estimators': randint(50, 200),
                'max_depth': randint(3, 7),
                'l2_leaf_reg': uniform(0, 10),
                'colsample_bylevel': uniform(0.6, 0.9),
                'random_strength': uniform(0, 1),
                'bootstrap_type': ['Bayesian', 'Bernoulli', 'MVS'],
                'max_ctr_complexity': randint(1, 5),
                'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide']
            }
            cat_model = CatBoostClassifier()
            search = RandomizedSearchCV(estimator=cat_model, param_distributions=param_distributions, cv=3, n_jobs=-1)
            search.fit(self.x_train, self.y_train)
            best_params = search.best_params_
            return dict(search.best_params_)

    def complement_nb(self, tuning=False, params=False):
        if not tuning:
            param_dist = {}
            if params:
                print(params)
                param_dist = params
            print("Complement Naive Bayes")
            model = ComplementNB(**param_dist)
            model.fit(self.x_train, self.y_train)
            y_pred = model.predict(self.x_test)
            predictions = [round(value) for value in y_pred]
            return model, self.scores(predictions)
        else:
            print('Complement Naive Bayes Hyperparameter Tuning')
            param_distributions = {
                'alpha': uniform(0.0, 5.0),
                'fit_prior': [True, False],
                'norm': [True, False]
            }
            nb_model = ComplementNB()
            search = RandomizedSearchCV(estimator=nb_model, param_distributions=param_distributions, cv=3, n_jobs=-1)
            search.fit(self.x_train, self.y_train)
            best_params = search.best_params_
            return dict(search.best_params_)

    def k_neighbors(self, tuning=False, params=False):
        if not tuning:
            print("k_neighbors")
            param_dist = {}
            if params:
                print(params)
                param_dist = params
            print("K-Nearest Neighbors (KNN)")
            model = KNeighborsClassifier(**param_dist)
            model.fit(self.x_train, self.y_train)
            y_pred = model.predict(self.x_test)
            predictions = [round(value) for value in y_pred]
            return model, self.scores(predictions)
        else:
            print("K-Nearest Neighbors (KNN) Hyperparameter Tuning")
            param_distributions = {
                'n_neighbors': randint(1, 10),
                'leaf_size': randint(1, 50),
                'p': [1, 2],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'metric': ['euclidean', 'manhattan', 'minkowski'],
                'n_jobs': [-1]
            }
            knn_model = KNeighborsClassifier()
            search = RandomizedSearchCV(estimator=knn_model, param_distributions=param_distributions, cv=3, n_jobs=-1)
            search.fit(self.x_train, self.y_train)
            best_params = search.best_params_
            return dict(search.best_params_)

    def MLPClassifier(self, tuning=False, params=False):
        if not tuning:
            print("Multi-Layer Perceptron (MLP) ")
            param_dist = {}
            if params:
                print(params)
                param_dist = params
                
            model = MLPClassifier(**param_dist)
            model.fit(self.x_train, self.y_train)
            y_pred = model.predict(self.x_test)
            predictions = [round(value) for value in y_pred]
            return model, self.scores(predictions)
        else:
            print("Multi-Layer Perceptron (MLP) Hyperparameter Tuning")
            param_distributions = {
                'hidden_layer_sizes': [(50,), (100,), (200,)],
                'activation': ['identity', 'logistic', 'tanh', 'relu'],
                'solver': ['lbfgs', 'sgd', 'adam'],
                'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
                'learning_rate': ['constant', 'invscaling', 'adaptive'],
                'learning_rate_init': [0.001, 0.01, 0.1, 1.0],
                'power_t': [0.1, 0.5, 1.0],
                'max_iter': randint(100, 1000),
                'shuffle': [True, False],
                'random_state': [None],
                'tol': [1e-4, 1e-3, 1e-2],
                'momentum': [0.1, 0.5, 0.9],
                'nesterovs_momentum': [True, False],
                'early_stopping': [True, False],
                'validation_fraction': [0.1, 0.2, 0.3],
                'beta_1': [0.9, 0.99, 0.999],
                'beta_2': [0.9, 0.99, 0.999],
                'epsilon': [1e-8, 1e-7, 1e-6]
            }
            mlp_model = MLPClassifier()
            search = RandomizedSearchCV(estimator=mlp_model, param_distributions=param_distributions, cv=3, n_jobs=-1)
            search.fit(self.x_train, self.y_train)
            best_params = search.best_params_
            return dict(search.best_params_)

    def support_vector_machine(self, tuning=False, params=False):
            if not tuning:
                print("Support Vector Machine (SVM)")
                param_dist = {}
                if params:
                    print(params)
                    param_dist = params
                model = LinearSVC()
                model.fit(self.x_train, self.y_train)
                y_pred = model.predict(self.x_test)
                predictions = [round(value) for value in y_pred]
                return model, self.scores(predictions)
            else:
                print("Support Vector Machine (SVM) hyper parameter tuning")
                param_distributions = {
                    'C': uniform(0.1, 10.0),
                    'tol': uniform(1e-4, 1e-2),
                    'class_weight': [None, 'balanced'],
                    'random_state': [None]
                }
                svm_model = LinearSVC()
                search = RandomizedSearchCV(estimator=svm_model, param_distributions=param_distributions, cv=3, n_jobs=-1)
                search.fit(self.x_train, self.y_train)
                best_params = search.best_params_
                return best_params
            
    def C4_5(self, tuning=False, params=False):
        if not tuning:
            print("C4.5 Decision Tree")
            param_dist = {}
            if params:
                print(params)
                param_dist = params
                
            model = DecisionTreeClassifier(**param_dist)
            model.fit(self.x_train, self.y_train)
            y_pred = model.predict(self.x_test)
            predictions = [round(value) for value in y_pred]
            return model, self.scores(predictions)
        else:
            print("C4.5 Decision Tree Hyperparameter Tuning")
            param_distributions = {
                'max_depth': [None, 3, 5, 7],
                'criterion': ['gini', 'entropy'],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 3, 5],
                'max_features': ['auto', 'sqrt', 'log2'],
                'random_state': [None],
                'splitter': ['best', 'random']
            }
            dt_model = DecisionTreeClassifier()
            search = RandomizedSearchCV(estimator=dt_model, param_distributions=param_distributions, cv=3, n_jobs=-1)
            search.fit(self.x_train, self.y_train)
            best_params = search.best_params_
            return dict(search.best_params_)

    def GBMClassifier(self, tuning=False, params=False):
        if not tuning:
            print("Gradient Boosting Machine (GBM)")
            param_dist = {}
            if params:
                print(params)
                param_dist = params
                
            model = GradientBoostingClassifier(**param_dist)
            model.fit(self.x_train, self.y_train)
            y_pred = model.predict(self.x_test)
            predictions = [round(value) for value in y_pred]
            return model, self.scores(predictions)
        else:
            print("Gradient Boosting Machine (GBM) Hyperparameter Tuning")
            param_distributions = {
                'learning_rate': uniform(0.01, 1),
                'n_estimators': randint(50, 200),
                'max_depth': randint(3, 7),
                'min_samples_split': randint(2, 10),
                'min_samples_leaf': randint(1, 5),
                'subsample': uniform(0.6, 0.9),
                'max_features': ['sqrt', 'log2'],
                'loss': ['deviance', 'exponential']
            }
            gbm_model = GradientBoostingClassifier()
            search = RandomizedSearchCV(estimator=gbm_model, param_distributions=param_distributions, cv=3, n_jobs=-1)
            search.fit(self.x_train, self.y_train)
            best_params = search.best_params_
            return dict(search.best_params_)
