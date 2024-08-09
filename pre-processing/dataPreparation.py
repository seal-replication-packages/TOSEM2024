import json
import os
from helper import Merged
import re
import textstat
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import statics
import helper

class DataPreparation:

    @staticmethod
    def authors_refactoring_contribution(dataset, refactoring_keywords):
        
        # Read in JSON file
        with open(dataset, 'r') as f:
            data = json.load(f)

        # Initialize counters
        total_commits = 0
        results = {}

        # Iterate over commits and count number of commits per author
        for project, commits in data.items():
            # if project != 'pymc':
            #     continue
            print(project)
            results[project] = {}
            for commit_hash, commit_data in commits.items():
                refactorings_count = len(commit_data['refactorings'])
                author = commit_data['author']
                if author not in results[project]:
                    results[project][author] = {}
                label = False
                for key in refactoring_keywords:
                    if (refactorings_count >= 1) or (key in commit_data['message']):
                        label = True

                if label:
                    results[project][author][commit_data['date']] = 1
                else:
                    results[project][author][commit_data['date']] = 0

            summary = {}
            for repo, details in results.items():
                summary[repo] = {}
                for author, dates in details.items():
                    dev = 0
                    ref = 0
                    summary[repo][author] = {}

                    import datetime

                    # Convert the string keys to datetime objects and store them in a list
                    datetime_list = [(datetime.datetime.strptime(key, '%a %b %d %H:%M:%S %Y %z'), value) for key, value in dates.items()]
                    # Sort the list based on the datetime objects
                    sorted_list = sorted(datetime_list, key=lambda x: x[0])
                    # Convert the sorted list back to a dictionary with the original string keys
                    d = {key: value for (key, value) in [(dt.strftime('%a %b %-d %H:%M:%S %Y %z'), v) for (dt, v) in sorted_list]}

                    for date, value in d.items():
                        if value == 0:
                            dev += 1
                        elif value == 1:
                            ref += 1
                        summary[repo][author][date] = {'dev': dev, 'ref': ref}

        with open('data/authors.json', 'w') as f:
            json.dump(summary, f, indent=4)

    @staticmethod
    def label_data(preprocessed, directory, refactoring_keywords):

        with open("data/authors.json") as json_file:
            authors = json.load(json_file)
        with open('data/code-metrics/mars.json', 'r') as file:
            # Load the JSON data
            metrics = json.load(file)
        print('finished loading non ml metrics')

        with open(preprocessed) as json_file:
            preprocessed = json.load(json_file)

        metric_labels = list(metrics['mars']['0041ed9c36afce59b6adf652c64a2bd9f1fb7acd']['current'].keys())
        
        zero_metric_list = {}
        for metric_label in metric_labels:
            zero_metric_list[metric_label] = 0

        if not os.path.exists(directory):
            os.mkdir(directory)
        
        results = []
        for repository, commits in preprocessed.items():
            # print(repository)
            for commit, details in commits.items():
                if details['changes']:

                    # get code metrics
                    has_py = False
                    for row in details['changes']:
                        if row['path'].split('.')[-1] == 'py':
                            has_py = True

                    try:
                        code_metrics = metrics[repository][commit]['difference']       
                    # For the ones that do not exist in metrics (not python files don't have metrics)              
                    except KeyError:
                        if not has_py:
                            code_metrics = zero_metric_list
                        else:
                            code_metrics = False

                    refactorings_count = len(details['refactorings'])
                    loc = Merged.codes_added_deleted(details)

                    lines_added = loc[0]
                    lines_deleted = loc[1]
                    lines_changed = lines_added - lines_deleted
                    files = Merged.files(details)
                    
                    # Shannon's entropy
                    code_entropy = Merged.code_entropy(details)
                    
                    words_count = len(details['pre_processed'].split())
                    sentences_count = len(re.split('[.!?]+', details['pre_processed']))
                    readability = textstat.flesch_reading_ease(details['message'])

                    refactoring_contribution_ratio = authors[repository][details['author']][details['date']]['ref']/(authors[repository][details['author']][details['date']]['ref']+authors[repository][details['author']][details['date']]['dev'])

                    # (refactorings_count >= threshold) or
                    label = 0
                    
                    # Check for existing python code
                    has_py_file = False
                    for row in details['changes']:
                        file_extension = os.path.splitext(row['path'])[1]
                        if not (file_extension == ".md" or file_extension == ".rst"):
                            has_py_file = True


                    for key in refactoring_keywords:
                        if ((refactorings_count >= 1) or (key in details['message'])) and has_py_file:
                            # print(key, details['message'])
                            label = 1
                    has_refactoring = 0
                    if (refactorings_count >= 1):
                        has_refactoring = 1
                    has_keyword = 0
                    for key in refactoring_keywords:
                        if (key in details['message']):
                            has_keyword = 1
                        
                    if has_py_file:
                        has_py_file = 1
                    else:
                        has_py_file = 0
                    if code_metrics:
                        metric_values = list(code_metrics.values())
                        results.append([repository, commit, details['pre_processed'], has_refactoring, has_keyword, has_py_file,
                                        lines_added, lines_deleted, lines_changed, files, code_entropy, 
                                        words_count, sentences_count, readability, 
                                        refactoring_contribution_ratio, label]+metric_values)
                    
        results = pd.DataFrame(results, columns=['repository', 'sha', 'message', 'has_refactoring', 'has_keyword', 'has_py_file',
                                    'lines_added', 'lines_deleted', 'lines_changed', 'files', 'code_entropy',
                                    'words_count', 'sentences_count', 'readability',
                                    'refactoring_contribution_ratio', 'y_label']+metric_labels) 
        results = results.drop(columns=['repository', 'sha'])      
        results.to_pickle(directory+"/labeled.df", compression='infer')


    @staticmethod
    def vectorize(dataset, vocabulary):
        
        # Split the initlal dataframe
        messages = dataset['message']
        process_metrics = dataset.loc[:, statics.process_metrics].reset_index(drop=True)
        y_test = dataset.loc[:, ['y_label']].reset_index(drop=True)
        
        # Concat Vectorized commit messages and process metrics into x_test
        model = CountVectorizer(vocabulary=vocabulary, ngram_range=(1, 6), analyzer='word', max_features=None)
        model.fit(messages)
        matrix = model.transform(messages).toarray()
        messages_vectorized = pd.DataFrame(matrix, columns=model.get_feature_names_out())
        x_test = pd.concat([messages_vectorized, process_metrics], axis=1)
        
        x_train = pd.read_pickle('../data/ml_projects/x_train.df')
        x_test = helper.minmax_normalize(x_train, x_test)
        
        x_test.to_pickle('data/labeled/x_labeled.df')
        y_test.to_pickle('data/labeled/y_labeled.df')

        