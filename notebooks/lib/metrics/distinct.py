from nltk.util import ngrams
import numpy as np

# Function to compute the distinct metric
def distinct(sentences, ngram_size=3):
    scores = []
    # For each sentence...
    for sentence in sentences:
        # Get the ngrams of required size, and encode them as a set (-> no repeated elements)
        distinct_ngrams = set(ngrams(sentence.split(), ngram_size))
        # Divide the length of this set by the number of tokens in the sentence (approx. = the number of ngrams) to get the distinct
        # score for this sentence
        scores.append(len(distinct_ngrams) / max(1, len(sentence.split()) - (ngram_size - 1)))
    # Compute mean and std of scores
    return np.mean(np.array(scores)), np.std(np.array(scores))