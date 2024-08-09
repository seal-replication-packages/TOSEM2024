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
final_classifier = Classifier(Classifier.ground_truth)

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


