import numpy as np
from typing import List
from sentence_transformers.readers import InputExample
from sentence_transformers import SentenceTransformer


class TripletEvaluator:
    """
    Evaluate a model based on a triplet: (sentence, positive_example, negative_example).
        Checks if distance(sentence, positive_example) < distance(sentence, negative_example).
    """

    def __init__(
        self,
        anchors: List[str],
        positives: List[str],
        negatives: List[str],
        batch_size: int = 16,
        verbose: bool = False,
    ):
        """
        :param anchors: Sentences to check similarity to. (e.g. a query)
        :param positives: List of positive sentences
        :param negatives: List of negative sentences
        :param batch_size: Batch size used to compute embeddings
        :param verbose: If true, prints a progress bar
        """
        self.anchors = anchors
        self.positives = positives
        self.negatives = negatives

        assert len(self.anchors) == len(self.positives)
        assert len(self.anchors) == len(self.negatives)

        self.batch_size = batch_size
        self.verbose = verbose

    @classmethod
    def from_input_examples(cls, examples: List[InputExample], **kwargs):
        anchors = []
        positives = []
        negatives = []

        for example in examples:
            anchors.append(example.texts[0])
            positives.append(example.texts[1])
            negatives.append(example.texts[2])
        return cls(anchors, positives, negatives, **kwargs)

    def __call__(self, model: SentenceTransformer) -> float:

        if self.verbose:
            print('Encoding anchors, positives and negatives')
        embeddings_anchors = model.encode(self.anchors,
                                          batch_size=self.batch_size,
                                          show_progress_bar=self.verbose,
                                          convert_to_numpy=True)
        embeddings_positives = model.encode(self.positives,
                                            batch_size=self.batch_size,
                                            show_progress_bar=self.verbose,
                                            convert_to_numpy=True)
        embeddings_negatives = model.encode(self.negatives,
                                            batch_size=self.batch_size,
                                            show_progress_bar=self.verbose,
                                            convert_to_numpy=True)

        # Euclidean
        pos_distances = np.linalg.norm(embeddings_anchors -
                                       embeddings_positives,
                                       axis=1)
        neg_distances = np.linalg.norm(embeddings_anchors -
                                       embeddings_negatives,
                                       axis=1)
        assert len(pos_distances) == len(neg_distances) == len(self.anchors)

        return sum(pos_distances < neg_distances) / len(self.anchors)
