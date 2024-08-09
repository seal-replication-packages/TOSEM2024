from dataCleaner import DataCleaner
from vocabulary import Vocabulary
from dataPreparation import DataPreparation
import sar
import numpy as np
import pandas as pd


# Clean the data extracted from Understand, pyRef, and bash scripts
DataCleaner.run()

# Preprocess the dataset with stemming and spelling correction
vocabulary = Vocabulary()
vocabulary.download()
vocabulary.pre_process('data/project.json', 'data/project.json')

# Extract author contribution ratios for refactoring activities
DataPreparation.authors_refactoring_contribution("data/project.json", sar.both)

# Label the refactoring data and save the results
DataPreparation.label_data('data/project.json', 'data/labeled', sar.both)

# Load the vocabulary from file
with open('data/vocabulary.npy', 'rb') as f:
    vocabulary = np.load(f, allow_pickle=True)

# Vectorize the dataset into machine-readable format
project = pd.read_pickle('data/labeled/labeled.df')
DataPreparation.vectorize(project, vocabulary)
