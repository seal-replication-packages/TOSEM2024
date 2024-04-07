import numpy as np
import pandas as pd
import pickle
from lime import lime_tabular
import multiprocessing
import re
import json
import statistics

class Interpretation:
    
    @staticmethod
    def collect_weights():
        """
            Collectes the feature weights for each testing set record (total of 124824) and store the medians at median_weights.json
        """
        num_entities = 124824
        # number of commits each core
        range_size = 2000
        num_tuples = (num_entities + range_size - 1) // range_size
        start_ends = [(start, min(start + range_size, num_entities)) for start in range(0, num_tuples * range_size, range_size)]

        # Define the function that will run in each process (Multiprocesssing)
        def process_iteration(start, end):
            print('working on'+str(start)+str(end))
            with open('data/interpretation/x_test.df', 'rb') as file:
                x_test = pickle.load(file)
            with open('data/interpretation/y_test.df', 'rb') as file:
                y_test = pickle.load(file)
            with open('data/interpretation/x_train.df', 'rb') as file:
                x_train = pickle.load(file)
            with open('data/interpretation/y_train.df', 'rb') as file:
                y_train = pickle.load(file)
            with open('data/interpretation/lightgbm.pickle', 'rb') as file:
                model = pickle.load(file)

            x_train = x_train
            y_train = y_train
            x_test = x_test[start:end]
            y_test = y_test[start:end]
            
            def extract_feature_name(feature):
                pattern = r"(?<![<>=])\b([a-zA-Z_]\w*(?:\s+[a-zA-Z_]\w*)*)\b(?![<>=])"
                match = re.search(pattern, feature)
                if match:
                    return match.group(1)
                else:
                    return feature

            def predict_fn(data):
                return np.array(list(zip(1-model.predict(data),model.predict(data))))

            print('training lime explainer')
            explainer = lime_tabular.LimeTabularExplainer(
                training_data=np.array(x_train),
                feature_names=x_train.columns,
                class_names=[1,0],
                mode='classification'
            )

            results = {}
            count = 0
            for index, row in x_test.iterrows():
                print(count)
                count = count+1
                exp = explainer.explain_instance(
                    data_row=row, 
                    predict_fn=predict_fn,
                    num_features=len(x_test.columns)
                )

                print(exp.as_list())
                for feature, weight in exp.as_list():
                    feature_name = extract_feature_name(feature)
                    if feature_name not in results:
                        results[feature_name] = []
                    results[feature_name].append(weight)

                out_file = open('data/interpretation/batches/'+str(start)+'_'+str(end)+'.json', "w")
                json.dump(results, out_file, indent=4)
                out_file.close()

        # alter start and end based on the cores available
        # print(start_ends[0:10])
        # print(start_ends[10:20])
        # exit()
        start_ends = start_ends[0:60]
        
        processes = []

        # Loop through the start_end
        for start_end in start_ends:
            print(start_end)
            # Create a process for each start_end tuple
            process = multiprocessing.Process(target=process_iteration, args=start_end)
            processes.append(process)
        # Start each process
        for process in processes:
            process.start()
        # Wait for all processes
        for process in processes:
            process.join()
        
        # Merge results into a singular file and calculate medians
        merged_dict = {}
        for start_end in start_ends:
            file_name = 'data/interpretation/batches/'+str(start_end[0])+'_'+str(start_end[1])+'.json'
            print(file_name)
            with open(file_name, 'rb') as file:
                weights = json.load(file)
            # Iterate over all the keys in the dictionaries
            for key in weights.keys():
                if key not in merged_dict:
                    merged_dict[key] = []
                # Merge the lists from all three dictionaries while preserving the order
                merged_dict[key] = merged_dict[key]+weights[key]
                
        # Replace lists with their median values
        for key, value in merged_dict.items():
            median_value = statistics.median(value)
            merged_dict[key] = median_value

        out_file = open('data/interpretation/median_weights.json', "w")
        json.dump(merged_dict, out_file, indent=4)
        out_file.close()
    
    @staticmethod
    def interpret_features():
        """
            Interpret the features obtained from running Lime using Interpretation.collect_weights()
        """
        with open('data/interpretation/lightgbm.pickle', 'rb') as file:
            model = pickle.load(file)
        with open('data/interpretation/x_train.df', 'rb') as file:
            x_train = pickle.load(file)
        with open('data/interpretation/median_weights.json', 'rb') as file:
            weights = json.load(file)
        weights_df = pd.DataFrame(list(weights.items()), columns=['Feature', 'Weight'])

        feat_importances = pd.Series(model.feature_importance(), index=x_train.columns)
        feat_importances = feat_importances.reset_index()
        feat_importances.columns = ['Feature', 'Importance']
        feat_importances = feat_importances[feat_importances['Importance'] != 0].reset_index(drop=True)
        
        merged_df = pd.merge(feat_importances, weights_df, on='Feature').sort_values(by='Importance', ascending=False)

        # Save the top features to a CSV file
        merged_df.to_csv('data/interpretation/top_features.csv', index=False)
