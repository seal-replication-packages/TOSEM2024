import os
import json
import pickle
import math
class FileManager:
    @staticmethod
    def read_json(directory):
        with open(directory) as json_file:
            return json.load(json_file)

    @staticmethod
    def dump_json(data, directory):
        out_file = open(directory, "w")
        json.dump(data, out_file, indent=4)
        out_file.close()

    @staticmethod
    def read_data(directory):
        with open(directory, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def dump_data(data, directory):
        with open(directory, 'wb') as f:
            pickle.dump(data, f)


class Merged:
    """
    This class is responsible to clear cut the information in the combined dataset (merged.json)
    """

    @staticmethod
    def code_churn(commit_detail):
        """
        This method returns development lines for each commit detail insertions + deletions
        :param commit_detail: sha -> details
        :return: lines of codes
        """
        lines = 0
        for change in commit_detail['changes']:
            lines = lines + int(change['insertions'])
            lines = lines + int(change['deletions'])
        return lines

    @staticmethod
    def code_difference(commit_detail):
        """
        This method returns development lines for each commit detail insertions + deletions
        :param commit_detail: sha -> details
        :return: lines of codes
        """
        lines = 0
        for change in commit_detail['changes']:
            lines = lines + (int(change['insertions'])-int(change['deletions']))
        return lines

    @staticmethod
    def codes_added_deleted(commit_detail):
        """
        This method returns development lines for each commit detail insertions + deletions
        :param commit_detail: sha -> details
        :return: lines of codes
        """
        lines_added = 0
        lines_deleted = 0
        for change in commit_detail['changes']:
            lines_added = lines_added + int(change['insertions'])
            lines_deleted = lines_deleted + int(change['deletions'])
        return lines_added, lines_deleted

    @staticmethod
    def refactoring_lines(commit_detail):
        """
        This method returns refactoring lines for each commit detail
        Important Notes: for Extract and Inline methods we only have multiple line changes. Others affect one single
        line
        :param commit_detail: sha -> details
        :return: lines of refactoring
        """
        lines = 0
        for refactoring in commit_detail['refactorings']:
            lines = lines + 1
            # for Extract and Inline method
            if "extracted/inlined lines" in refactoring:
                rows = refactoring['extracted/inlined lines']
                lines = lines + (int(rows[-1]) - int(rows[0]) + 1)

        return lines

    @staticmethod
    def files(commit_detail):
        """
        This method returns files for each commit detail
        :param commit_detail: sha -> details
        :return: number of files
        """
        return len(commit_detail['changes'])

    @staticmethod
    def sha_timestamp(merged, repo, date):
        """
        This method returns files for each commit detail
        :param commit_detail: sha -> details
        :return: number of files
        """
        for sha, details in merged[repo].items():
            if details['date'] == date:
                return sha
        return False
    
    @staticmethod
    def code_entropy(commit_detail):
        code_entropy = 0
        p = []
        sum_all_lines = sum(int(change["insertions"]) + int(change["deletions"]) for change in commit_detail['changes']) 
        for row in commit_detail['changes']:
            lines_in_file = int(row["deletions"]) + int(row["deletions"])
            if sum_all_lines == 0:
                p.append(0)
            else:
                p.append(lines_in_file / sum_all_lines) 
        if all(prob == 0 for prob in p):
            code_entropy = 0  # if all probabilities are zero, entropy is defined to be zero
        else:
            code_entropy = -sum(prob * math.log2(prob) for prob in p if prob > 0)
            
        return code_entropy


def repos(location):
    """
    Gets all repo names based on the location of json extracted files
    :param location: location of json files
    :return: Non
    """
    repo_names = []
    for filename in os.listdir(location):
        file_name = os.path.join(location, filename)
        if os.path.isfile(file_name):
            if file_name.endswith("json"):
                repo_name = file_name.split("/")[-1][:-5]
                repo_names.append(repo_name)
    return repo_names

def authors_fix(location):
    """
    fixes errors in the files for instance
    "author": "Jan "yenda" Trmal <jtrmal@gmail.com>"
    This is manual for now so you gotta change it in code
    :param location: location of the file
    :return:
    """
    # Read in the file
    with open(location, 'r') as file:
        filedata = file.read()

    # Replace the target string
    # filedata = filedata.replace('"decko"', 'decko')
    filedata = filedata.replace('\\', '-')

    # Write the file out again
    with open(location, 'w') as file:
        file.write(filedata)
        
def correct_sentence_spelling(sentence, d, suggestions_cache):
        # Split the sentence into words
        words = sentence.split()

        # Check each word for spelling errors and correct them
        corrected_words = []
        for word in words:
            if not d.check(word):
                if word in suggestions_cache:
                    suggestions = suggestions_cache[word]
                else:
                    suggestions = d.suggest(word)
                    suggestions_cache[word] = suggestions
                if len(suggestions) > 0:
                    corrected_word = suggestions[0]
                else:
                    corrected_word = word
            else:
                corrected_word = word
            corrected_words.append(corrected_word)

        # Join the corrected words back into a sentence
        corrected_sentence = " ".join(corrected_words)

        return corrected_sentence, suggestions_cache

def minmax_normalize(x_train, x_test):
    # Exclude columns that should not be normalized
    columns_to_normalize = x_test.columns.difference(['sha', 'repository'])

    # Normalize only the selected columns
    col_min = x_train[columns_to_normalize].min()
    col_max = x_train[columns_to_normalize].max()

    x_test_normalized = x_test.copy()
    x_test_normalized[columns_to_normalize] = (x_test[columns_to_normalize] - col_min) / (col_max - col_min)

    return x_test_normalized
