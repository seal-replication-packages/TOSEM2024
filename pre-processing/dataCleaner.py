import os
import json
import helper
import pandas as pd

class DataCleaner:
    """
        This class is responsible to process the mined data
    """

    @staticmethod
    def pyref(location):
        """
        This method cleans up the Pyref results from the location and groups them based on sha and renames files
        :param str location: the location folder
        :return: Non
        """
        print("Location: " + location)

        for filename in os.listdir(location):
            old_file_name = os.path.join(location, filename)
            if os.path.isfile(old_file_name):
                # Check if the file is not already converted
                if old_file_name.endswith("_data.json"):
                    new_file_name = old_file_name[:-10] + '.json'
                    os.rename(old_file_name, new_file_name)
                    print(old_file_name + " -> Renamed to -> " + new_file_name)

                    # Grouping refactorings based on sha
                    results = {}
                    with open(new_file_name) as json_file:
                        refactorings = json.load(json_file)
                    for refactoring in refactorings:
                        sha = refactoring['Commit']
                        refactoring.pop('Commit', None)
                        refactoring = {k.lower(): v for k, v in refactoring.items()}
                        if sha not in results:
                            results[sha] = []
                        results[sha].append(refactoring)

                    # Replace with good structure
                    out_file = open(new_file_name, "w")
                    json.dump(results, out_file, indent=4)
                    out_file.close()
                    print(new_file_name + " -> Organized into json format")

    @staticmethod
    def commit_logs(location):
        """
        This method cleans up the Commit logs results from the location and groups them based on sha
        :param str location: the location folder
        :return: Non
        """
        print("Location: " + location)
        count = 0
        for filename in os.listdir(location):
            file_name = os.path.join(location, filename)
            if os.path.isfile(file_name):
                count= count+1
                print(count)
                # Check if the file is not already converted
                if file_name.endswith(".json"):
                    print("working on: ", file_name)
                    if file_name != 'data/repositories/non-ml/details/commit-logs/adwaita-icon-theme.json':
                        helper.authors_fix(file_name)
                    results = {}
                    # print("working on: ", file_name)
                    with open(file_name) as json_file:
                        commits = json.load(json_file)
                        
                    if not isinstance(commits, list):
                        continue
                    # # Check if the first value is an array (list) and keep only the first element
                    # for key in commits:
                    #     if isinstance(commits[key], list):
                    #         commits[key] = commits[key][0]
                    
                    for commit in commits:
                        # print(commit)
                        sha = commit['commit']
                        commit.pop('commit', None)
                        if sha not in results:
                            results[sha] = []
                        results[sha].append(commit)

                    # Replace with good structure
                    out_file = open(file_name, "w")
                    json.dump(results, out_file, indent=4)
                    out_file.close()
                    print(file_name + " -> Organized into json format")

    @staticmethod
    def stats(location):
        """
        This method cleans up the Pyref results from the location folder into target folder
        Important Notes: We set False to the ones that don't have lines changes (image or they dont exist)
        (May cause error in future)
        :param str location: the location folder
        :return: Non
        """
        print("Location: " + location)

        for filename in os.listdir(location):
            file_name = os.path.join(location, filename)
            if os.path.isfile(file_name):
                if file_name.endswith("json"):

                    # Clean not utf8 encoded formats
                    with open(file_name, 'r') as file:
                        file_data = file.read()
                    file_data = file_data.replace('""', '"')
                    file_data = file_data.replace('\\', '-')
                    with open(file_name, 'w') as file:
                        file.write(file_data)

                    # Convert Strings to Int on insertions and deletions
                    with open(file_name) as json_file:
                        json_file = json.load(json_file)

                    # Clean the data and set them to proper types
                    results = {}
                    for sha, file_changes in json_file.items():
                        results[sha] = []
                        for file_change in file_changes:
                            # We set False to the ones that don't have lines changes (image or they dont exist)
                            insertions = file_change["insertions"]
                            if file_change["insertions"] == '-':
                                insertions = False
                            deletions = file_change["deletions"]
                            if file_change["deletions"] == '-':
                                deletions = False
                            # Append the changes to results as temp variable that holds changes
                            results[sha].append({
                                "insertions": insertions,
                                "deletions": deletions,
                                "path": file_change["path"]
                            })

                    # Write in pretty json format
                    out_file = open(file_name, "w")
                    json.dump(results, out_file, indent=4)
                    out_file.close()
                    print(file_name + " -> Pretty formatted")

    @staticmethod
    def merge_commits_and_refactorings():
        """
        This method combines all refactoring and gitlog commit information into a singular file
        :param str location: the starting location folder
        :return: Non
        """
        
        refactorings_location = 'data/refactorings'
        stats_location = 'data/commit-stats'
        commit_logs_location = 'data/commit-logs'
        print("Refactorings location: " + refactorings_location)
        print("Stats location: " + stats_location)
        print("Commit logs location: " + commit_logs_location)
        print("-----------")
        repos = helper.repos(stats_location)

        results = {}
        count = 0

        repository_list = ['mars']
        # till here
        
        for repo in repos:
            if repo not in repository_list:
                print('skipping ', repo)
                continue
            # till here
            print("Working on: " + repo)

            # Load refactorings
            with open(refactorings_location + "/" + repo + ".json") as json_file:
                refactorings = json.load(json_file)
            refactorings_count = len(refactorings.keys())
            # print("Number of refactoring shas: "+ str(refactorings_count))

            # Load stats
            with open(stats_location + "/" + repo + ".json") as json_file:
                stats = json.load(json_file)

            # Load Commit logs
            with open(commit_logs_location + "/" + repo + ".json") as json_file:
                commmit_logs = json.load(json_file)

            # Merge results
            results[repo] = {}
            refactoring_finds_count = 0
            for sha, changes in stats.items():
                results[repo][sha] = {}
                for key, value in commmit_logs[sha][0].items():
                    results[repo][sha][key] = value
                results[repo][sha]['changes'] = changes
                results[repo][sha]['refactorings'] = []
                if sha in refactorings:
                    refactoring_finds_count = refactoring_finds_count + 1
                    results[repo][sha]['refactorings'] = refactorings[sha]
                    
            # Check if we could target all refactoring shas
            if refactoring_finds_count != refactorings_count:
                print("Results: *****Error (Can't find all refactorings in stats)*****")
            else:
                print("Results: Success")

        # Save as json file
        out_file = open("data/project.json", "w")
        json.dump(results, out_file, indent=4)
        out_file.close()


    @staticmethod
    def run():
        # DataCleaner.commit_logs('data/repositories/non-ml/details/commit-logs')
        # DataCleaner.pyref('data/repositories/non-ml/details/py-ref')
        # DataCleaner.stats('data/repositories/non-ml/details/commit-stats')
        # DataCleaner.releases('Data/fixed_batch/Releases')
        DataCleaner.merge_commits_and_refactorings()

