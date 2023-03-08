from regex import D
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

import scipy.sparse as sp
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

def sentence_preprocess(sentence, stopwords=stopwords.words(), min_sentence_len=3):
    """
    This function can be used in order to preprocess a given character sentence
    ## Params
    * `sentence`: the sentence to preprocess
    * `stopwords`: is a set of stopwords, the default value of such parameter is the set of 
    english stopwords of the library nltk
    * `min_sentence_len`: is the minimum lenght a sentence should have in order to be preprocessed
    and remain in the dataset
    ## Returns
    The preprocessed sentence
    """
    # substitute punctuations with single space
    sentence = re.sub(r'[^A-Za-z\s]', ' ', sentence)
    # substitutes multiple spaces with one single
    sentence = re.sub(r'\s+', ' ', sentence)
    # split sentences
    sentence_splitted = sentence.split()
    # check if the sentence is too short
    is_short = len(sentence_splitted) <= min_sentence_len
    # check if the sentence contains at least a non stopword
    is_relevant = False
    for word in sentence_splitted:
        if word not in stopwords:
            is_relevant = True
            break
    # returns the sentence preprocessed and the final relence of it
    return sentence, (not is_short and is_relevant)

###
# credits: https://towardsdatascience.com/creating-a-class-based-tf-idf-with-scikit-learn-caea7b15b858
class CTFIDFVectorizer(TfidfTransformer):
    """
    The goal of the class-based TF-IDF is to supply all documents within a single class
    with the same class vector. In order to do so, we have to start looking at TF-IDF 
    from a class-based point of view instead of individual documents.
    If documents are not individuals, but part of a larger collective, then it might be 
    interesting to actually regard them as such by joining all documents in a class together.
    """
    def __init__(self, *args, **kwargs):
        super(CTFIDFVectorizer, self).__init__(*args, **kwargs)

    def fit(self, X: sp.csr_matrix, n_samples: int):
        """Learn the idf vector (global term weights) 
        ## Params
        * `n_samples`: is the total number of unjoined documents.
        This is necessary as the IDF values become too small if the number of joined documents 
        is passed instead."""
        # extract the number of features
        _, n_features = X.shape
        df = np.squeeze(np.asarray(X.sum(axis=0)))
        idf = np.log(n_samples / df)
        self._idf_diag = sp.diags(idf, offsets=0,
                                  shape=(n_features, n_features),
                                  format='csr',
                                  dtype=np.float64)
        return self

    def transform(self, X: sp.csr_matrix) -> sp.csr_matrix:
        """Transform a count-based matrix to c-TF-IDF """
        X = X * self._idf_diag
        X = normalize(X, axis=1, norm='l1', copy=False)
        return X
###

class FrequencyClassifier:
    """
    A word frequency classifier for characters classification
    """
    def __init__(self, characters, mode):
        """
        Construct a `FrequencyClassifier` 
        ## Params
        * `characters`: list of characters which corresponds to the labels of the model
        * `mode`: set the modality of the classifier based on the following possibilities:
           (`word frequency`: based on a count frecquency vectorizer, `tf-idf`: based on a 
            list of tf-idf, `c-tf-idf`: based on a class defined tf-idf)
        """
        # initialize parameters
        self.characters = characters
        if mode != "word frequency" and mode != "tf-idf" and mode != "c-tf-idf":
            raise Exception("Unknown mode!")
        self.mode = mode
        self.loaded = False
        self.model = None
        self.distances = None

    def train(self, docs):
        """
        Train the model
        ## Params
        * `docs`: list of documents, in order to be done correctly the training, `len(docs)` must
        be equal to the number of characters you want to consider
        """
        # assert coherence with the number of classes
        self.model = None
        self.loaded = False
        if len(docs) != len(self.characters):
            raise Exception("Mismatch between classifier classes and provided documents!")
        
        # compute total length of the unjoined documents
        n_tot_docs = 0
        for doc_character in docs:
            n_tot_docs +=  len(doc_character)

        docs = [' '.join(doc) for doc in docs]
        ### word frequency model
        if self.mode == 'word frequency':
            # fit count vectorizer
            vectorizer = CountVectorizer(stop_words='english').fit(docs)
            count = vectorizer.transform(docs)
            self.model = {
                'vectorizer': vectorizer,
                'vectors': count,
                'docs': docs
            }
        ###
        ### tf-idf
        elif self.mode == 'tf-idf':
            tfidf_vectorizer = TfidfVectorizer(input='content', stop_words='english')
            vectors = tfidf_vectorizer.fit_transform(docs)
            self.model = {
                'vectorizer': tfidf_vectorizer,
                'vectors': vectors,
                'docs': docs
            }
        ###
        ### c-tf-idf
        elif self.mode == 'c-tf-idf':
            # fit count vectorizer
            count_vectorizer = CountVectorizer(stop_words='english').fit(docs)
            count = count_vectorizer.transform(docs)
            # fit c-tf-idf vectorizer
            ctfidf_vectorizer = CTFIDFVectorizer().fit(count, n_samples=n_tot_docs)
            ctfidf = ctfidf_vectorizer.transform(count)
            self.model = {
                'count_vectorizer': count_vectorizer,
                'vectorizer': ctfidf_vectorizer,
                'vectors': ctfidf,
                'docs': docs
            }
        # store the flag for the fact the model had been loaded and trained on dataset labels
        self.loaded = True

    def predict(self, doc):
        """
        Classify a new document based on the technology previously selected in the `mode` parameter
        ## Params
        * `doc`: new document do classify
        ## Returns
        a dictionary where the keys are the character' names and their corresponding values are the 
        cosine similarity among the new document and the training documents
        """
        # check if the the model had been already loaded
        if not self.loaded:
            raise Exception("Classifier must be trained first!")
        # prepare the document to test
        doc1 = ' '.join(doc)
        predictions = dict()
        self.distances = None
        ### word frequency model || tf-idf
        if self.mode in ['word frequency', 'tf-idf']:
            test_vector = self.model['vectorizer'].transform([doc1])
            self.distances = cosine_similarity(test_vector, self.model['vectors'])[0]
        ###
        ### c-tf-idf
        elif self.mode == 'c-tf-idf':
            count = self.model['count_vectorizer'].transform([doc1])
            test_vector = self.model['vectorizer'].transform(count)
            self.distances = cosine_similarity(test_vector, self.model['vectors'])[0]
        # prepare the predictions for output
        for character, idx_ch in zip(self.characters, range(len(self.characters))):
            predictions[character] = self.distances[idx_ch]
        return predictions