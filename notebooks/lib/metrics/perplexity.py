from tqdm import tqdm
import numpy as np

# Function to compute the perplexity of the model
def perplexity(model, encoded_test_set):
    # list of negative log-likelihoods
    nlls = []
    # Get an iterator over the test set, already encoded by the tokenizer
    iterator = iter(encoded_test_set)
    # For each iteration...
    for _ in tqdm(range(len(encoded_test_set))):
        # Get the batch, and evaluate it through the model, returning the loss (= average nll)
        batch = next(iterator)
        loss = model.evaluate(batch, verbose=0)
        # Append the nll to the list
        nlls.append(loss)
    # Average over the nlls and exponentiate them to get the perplexity score
    return np.exp(np.array(nlls).sum() / len(encoded_test_set))