import helper
import json
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
import nltk
import re
import enchant
# import enchant


class Vocabulary:

    def __init__(self):
        self.sw = stopwords.words('english')
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = SnowballStemmer('english')

    @staticmethod
    def download():
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('omw-1.4')

    def text_preprocess(self, text):

        text = text.lower()
        text = text.replace('w', '')
        # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
        text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text)
        # Removing URLs
        # text = re.sub(r"http\S+", "", text)
        html = re.compile(r'<.*?>')
        # Removing html tags
        text = html.sub(r'', text)
        # Removing punctuations
        punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`" + '_'
        for p in punctuations:
            text = text.replace(p, '')
        # Removing emojis
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)

        # remove digits
        text = ''.join([i for i in text if not i.isdigit()])

        # removing stopwords
        text = [word.lower() for word in text.split() if word.lower() not in self.sw]

        # Lemmatized sentence
        lemmatized = [self.lemmatizer.lemmatize(word) for word in text]
        lemmatized = " ".join(lemmatized)
        # Stemmed sentence
        stemmed = [self.stemmer.stem(word) for word in text]
        stemmed = " ".join(stemmed)

        return lemmatized, stemmed

    def pre_process(self, directory, target):
        """
        This method first pre-process the commit messages and saves to the target location
        :param directory: the directory of the merged dataset
        :param target: the location which the results will be saved
        """
 
        with open(directory) as json_file:
            repositories = json.load(json_file)

        count_all = len(repositories.keys())

        done = 0

        d = enchant.Dict("en_US")
        suggestions_cache = {}

        for repository, logs in repositories.items():
            for sha, details in logs.items():

                # print(suggestions_cache)
                message_lower = repositories[repository][sha]['message'].replace("-", " ").replace("_", " ").lower()
                spell_checked = helper.correct_sentence_spelling(message_lower, d, suggestions_cache)
                suggestions_cache = spell_checked[1]
                repositories[repository][sha]['message'] = spell_checked[0]
                message = self.text_preprocess(repositories[repository][sha]['message'])
                repositories[repository][sha]['pre_processed'] = message[1]

            done = done + 1
            print("\r", end='')
            print(str(done) + "/" + str(count_all) + " " + repository, end='', flush=True)

        out_file = open(target, "w")
        json.dump(repositories, out_file, indent=4)
        out_file.close()