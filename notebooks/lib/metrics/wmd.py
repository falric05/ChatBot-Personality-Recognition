import gensim.downloader as api
model = api.load("glove-twitter-25")

from nltk.corpus import stopwords
from nltk import download
download('stopwords')

# Function to compute the wmd metric
def wmd(sentence_a, sentence_b):
    sentence_a = [w for w in sentence_a.lower().split() if w not in stopwords.words('english')]
    sentence_b = [w for w in sentence_b.lower().split() if w not in stopwords.words('english')]
    # Use wmdistance function from gensim
    # See https://radimrehurek.com/gensim/auto_examples/tutorials/run_wmd.html
    if sentence_a and sentence_b:
        return model.wmdistance(sentence_a, sentence_b)
    else:
        return None