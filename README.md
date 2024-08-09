
# Supplemental Materials

This repository contains the replication package for the paper "Detecting Refactoring Commits in Python Projects: A Machine Learning-based Approach".


## Introduction



We have organized the replication package into two folders and five Python files:

1. data: This folder contains all the data required to run the experiments.
2. appendix: This folder includes additional information about the paper, such as the complete list of important features, results of all classifiers, results of the correlation analysis, keywords used to identify refactoring commits in the state-of-the-art approaches, and sensitivity analysis results.
3. classification.py: This helper model is responsible for performing classification using different approaches.
4. classifier.py: classifier.py: This model is responsible for training classifiers and obtaining classifier scores under different evaluation criteria.
5. ensembleFusion.py: This model is responsible for generating the results of PRefScanner and PyRef and evaluating the ensembled model results.
6. interpretation.py: This model is responsible for collecting and interpreting the trained model's features.
7. main.py: This file acts as the main script, calling different models to generate our research question results.

Our code is based on the following packages and versions:
- scikit-learn: 1.1.1
- pandas: 1.4.3
- numpy: 1.22.4
- xgboost: 1.6.1
- lightgbm: 3.3.5
- scipy: 1.8.1
- catboost: 1.2
- lime: 0.2.0.1

The following code can be used to install all packages in the environment.
```bash
  pip install -r requirements.txt
```
To load the dataset unzip the data.zip file in the root directory of the project. You can use the command below:
```bash
  unzip data.zip
```

We recommend using Python version Python 3.10.12 and every Python requirement should be met.

    
## Usage/Examples

We have the following codes and fucntions available on main.py:
- **Research Question 1**:
    - Define one:
        - final_classifier = Classifier(Classifier.testing)
        - final_classifier = Classifier(Classifier.ground_truth)
    - Run the experiment:
        - final_classifier.run('lightgbm') (different classifier names could be passed)
    
- **Research Question 2:**
    - ensemble = EnsembleFusion()
    - ensemble.compare()

- **Research Question 3:**
    - Interpretation.interpret_features()

Below is an example of ho to run the functions in the root of the project.
```javascript
from classifier import Classifier
from classification import Classification
from ensembleFusion import EnsembleFusion
from interpretation import Interpretation


### RQ 1 RESULTS ##
"""
    To assess the model results, you can call the function below:
    Testing Set -> final_classifier = Classifier(Classifier.testing)
    Ground Truth -> final_classifier = Classifier(Classifier.ground_truth)
    General Projects -> final_classifier = Classifier(Classifier.general)
"""
final_classifier = Classifier(Classifier.testing)

"""
    The performance of different classification methods can be assessed by passing
    the name of the classifier to the run function. The classifier names can be accessed from Classification.names.
"""
final_classifier.run('lightgbm')

### RQ 2 RESULTS ###
"""
    This part is responsible for obtaining the ensemble learning results by combining PRefScanner and PyRer.
"""
ensemble = EnsembleFusion()
ensemble.compare()

### RQ 3 RESULTS ###
"""
    This part is responsible for identifying the most important features contributing to PRefScanner classifications.
"""
Interpretation.interpret_features()



```
## Data Information
In the data folder, the data provided include:
- **general projects**: The dataset of Python general projects that we used to measure the generalizability of our approach.
- **ground truth**: The ground truth dataset of our study.
- **interpretation**: The MLRefScanner model (lightgbm.pickle) and the output of the LIME interpreter.
- **ml_projects**: The dataset on which MLRefScanner was built.
- **hyperparameters.json**: The hyperparameters needed to train the model.

To get and extract information, the pipeline data is explained in the preprocessing folder.

## System Requirements
MLScanRef performs the best using the LightGBM algorithm. LightGBM is a highly efficient model that is easy to use and integrated into the machine-learning pipeline. For optimal performance when building MLRefScanner, we recommend having at least 5 GB of memory available and a modern multi-core processor (e.g., Intel i5/i7 or AMD Ryzen).





