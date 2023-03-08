import json
import random
import torch
import collections
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from numpy.typing import NDArray
from os import system
from tqdm import tqdm
from pandas import DataFrame, read_csv
from os.path import join, exists
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from .distil_bert_embedder import BarneyEmbedder  # pylint: disable = relative-beyond-top-level
from sentence_transformers.readers import InputExample

from lib.BBData import character_dict, random_state

characters_all = list(character_dict.keys())
if 'Default' in characters_all:
    characters_all.remove('Default')

random.seed(random_state)


class DistilBertClassifier:

    def __init__(self,
                 embedder_path: str = None,
                 from_pretrained: bool = False,
                 embedding_size: int = 32,
                 use_cuda: bool = False) -> None:
        """
        initialize DistilBert classifier:
        embedder_path (str): the path used to upload the embedder, if default value is given (None) the embedder il built from zero with a default structure 
        from_pretrained (bool): if true the embedder is built starting from embedder_path (must be not None), otherwise the embedder is initialized randomly, default is False
        embedding_size (int): dimensionality of the embedder output, default is 32
        use_cuda (bool): if True the training tensors are transfered to the GPU, default is False
        """

        ### save the characters for training purposes
        self.characters = characters_all

        ### defining train, validation and test percentage size
        self.train_size = 0.85
        self.val_size = 0.05

        ### given a single anchor, define the number of triplets to build for the triplet loss
        self.n_triplets_x_anchor: int = 10

        ### initialize the ambedder
        self.embedder = BarneyEmbedder(embedding_size=embedding_size,
                                       embedder_path=embedder_path,
                                       from_pretrained=from_pretrained,
                                       use_cuda=use_cuda)
        ### initialize the knn classifier
        self.classifier = KNeighborsClassifier()

    #

    def set_characters(self, characters: List[str]) -> None:
        """
        setting the list of characters to consider during training and/or testing
        """
        self.characters = characters

    def set_classifier(self, classifier) -> None:
        """
        setting the classifier for the embedder output
        the default classifier is knn
        """
        self.classifier = classifier

    #

    @staticmethod
    def get_character_df(series_df: DataFrame, n_shuffles: int,
                         n_sentences: int) -> DataFrame:

        """
        returns the dataframe of character / lines used for training and testing, shiffled
        the column 'character' indicates the character name,
        the column 'line' indicates a sentence uttered by the character in the dataset scripts
        for each character collects randomly a number of 'n_sentences' sentences from 'series_df'
        this will be the set of sentences used as input for the embedder

        series_df (pandas.DataFrame): the reference dataframe, must have a 'character' and a 'line' column
        n_shuffles (int): multiplication factor for the output dataset dimensionality, if n_shuffles > 1, the output dataset will be n_shuffles times larger 
        n_sentences (int): cardinality of the set of sentences that will be used as input for the embedder
        """
        n_sentences = n_sentences // 2
        # Separate lines by character from all the others
        series_df_char = series_df[series_df['character'] == 1].copy()
        # Define triplet dataset as having a character label and the line, already encoded
        df_rows = {'character': [], 'line': []}
        # Shuffle by a parametrized amount
        for i in range(n_shuffles):
            # Shuffle the dataset and balance number of 0s (we suppose its cardinality is higher than that of 1s)
            series_df_char = series_df_char.sample(frac=1,
                                                   random_state=random_state +
                                                   i).reset_index(drop=True)
            # Iterate over lines
            for i in range(n_sentences, len(series_df_char) - n_sentences + 1):
                # Get 2*'n_sentences'+1 consecutive lines for the character, and concatenate them in one sample
                lines = ' '.join(series_df_char['line'][i - n_sentences:i +
                                                        n_sentences])
                df_rows['character'].append(1)
                df_rows['line'].append(lines)
        # Create a new dataframe from the rows we have built
        df = DataFrame(data=df_rows)
        # Sample the dataset one last time to shuffle it
        return df.sample(frac=1,
                         random_state=random_state).reset_index(drop=True)

    def create_data(
            self,
            val: bool,
            source_encoded_path: str,
            n_shuffles: int = 2,
            merge_sentences: bool = True,
            n_sentences: int = 3,
            save_dataset: bool = True,
            verbose: bool = False) -> Tuple[List[DataFrame], List[DataFrame]]:

        """
        create and save dataset starting from the csv at 'source_encoded_path'
        this dataset will be used for the embedder training

        val (bool): if true splits the dataset also into a validation set, and saves it
        source_encoded_path (str): the path of the csv dataset folder, the csv must have a 'line' and a 'character' column
        n_shuffles (int): multiplication factor for the output dataset dimensionality, if n_shuffles > 1, the output dataset will be n_shuffles times larger, default is 2
        merge_sentences (bool): if True each sample of the returned dataset will have one character and more than one related sentences, default is True
        n_sentences (int): size of the sentence set for each sample in the dataset, default is 3
        save_dataset (bool): save the dataset built from the csv, default is True
        verbose (bool): set to False to avoid printings, default is False

        returns a list of training dataframes, a list of validation dataframes and a list of test dataframes
        we have a list since we need a dataframe for each character
        """


        ### load dataset
        if verbose:
            print('Loading encoded lines...')

        df_list = [
            read_csv(join(source_encoded_path, self.characters[c],
                          self.characters[c].lower() + '_classifier.csv'),
                     dtype={
                         'line': str,
                         'character': int
                     }) for c in range(len(self.characters))
        ]

        ### balance dataset
        max_len = min([len(df) for df in df_list])
        train_len = int(self.train_size * max_len)
        val_len = int(self.val_size * max_len) if val else 0
        val_end_idx = train_len + val_len
        df_list = [df[:max_len] for df in df_list]

        ### split in train and test
        df_list_train = []
        df_list_val = [] if val else None
        df_list_test = []
        for df in df_list:
            df_shuffled = df.sample(frac=1, random_state=random_state)

            df_list_train.append(df_shuffled[:train_len])
            df_list_test.append(df_shuffled[val_end_idx:])
            if val:
                df_list_val.append(df_shuffled[train_len:val_end_idx])

        ### augment dataset
        for c in tqdm(range(len(self.characters)), disable=not verbose):
            ### Load the preprocessed dataset
            series_df_train = df_list_train[c]
            series_df_test = df_list_test[c]
            if val: series_df_val = df_list_val[c]

            ### if 'merge_sentences' is True, consider for the embedding 'n_sentences' at a time
            if merge_sentences:
                series_df_train = self.get_character_df(
                    series_df_train,
                    n_shuffles=n_shuffles,
                    n_sentences=n_sentences)
                series_df_test = self.get_character_df(series_df_test,
                                                       n_shuffles=n_shuffles,
                                                       n_sentences=n_sentences)
                if val:
                    series_df_val = self.get_character_df(
                        series_df_val,
                        n_shuffles=n_shuffles,
                        n_sentences=n_sentences)

            ### otherwise use as embedding input only one sentence
            else:
                series_df_train = series_df_train[series_df_train['character']
                                                  == 1].reset_index()[[
                                                      'line', 'character'
                                                  ]]
                series_df_test = series_df_test[series_df_test['character'] ==
                                                1].reset_index()[[
                                                    'line', 'character'
                                                ]]
                if val:
                    series_df_val = series_df_val[series_df_val['character'] ==
                                                  1].reset_index()[[
                                                      'line', 'character'
                                                  ]]

            ### correct labels
            series_df_train['character'] = [
                c for _ in range(len(series_df_train))
            ]
            series_df_test['character'] = [
                c for _ in range(len(series_df_test))
            ]
            if val:
                series_df_val['character'] = [
                    c for _ in range(len(series_df_val))
                ]

            df_list_train[c] = series_df_train
            df_list_test[c] = series_df_test
            if val: df_list_val[c] = series_df_val

        ### save train and test datasets
        if save_dataset:
            with open(join(source_encoded_path, 'embedder_dataset_train.json'),
                      'w',
                      encoding='utf-8') as file:
                json.dump([df.to_dict() for df in df_list_train], file)

            with open(join(source_encoded_path, 'embedder_dataset_test.json'),
                      'w',
                      encoding='utf-8') as file:
                json.dump([df.to_dict() for df in df_list_test], file)

            if val:
                with open(join(source_encoded_path,
                               'embedder_dataset_val.json'),
                          'w',
                          encoding='utf-8') as file:
                    json.dump([df.to_dict() for df in df_list_val], file)

        return df_list_train, df_list_val, df_list_test

    def get_data(
        self,
        source_path: str,
        val: bool,
        override: bool = False,
        merge_sentences: bool = True,
        n_sentences: int = 3,
        verbose: bool = False,
    ) -> Tuple[List[str], List[int], List[str], List[int], List[str],
               List[int]]:

        """
        get the dataset  for the embedding training
        if asked ('override'=True) or needed, a new dataset is built starting from the csv at 'source_path'

        source_path (str): the path of the dataset folder
        val (bool): if true splits the dataset also into a validation set, and saves it
        override (bool): force building the dataset from the csv, if any dataset was already saved, it will be overwritten
        merge_sentences (bool): if True each sample of the returned dataset will have one character and more than one related sentences, default is True
        n_sentences (int): size of the sentence set for each sample in the dataset, default is 3
        verbose (bool): set to False to avoid printings, default is False

        returns the saved X_train, y_train, X_val, y_val, X_test, y_test
        """

        ### create dataset if needed
        if override:
            df_list_train, df_list_val, df_list_test = self.create_data(
                val=val,
                source_encoded_path=source_path,
                merge_sentences=merge_sentences,
                n_sentences=n_sentences,
                verbose=verbose)

        ### otherwise open the existing one
        else:
            ### load training set
            with open(join(source_path, 'embedder_dataset_train.json'),
                      'r',
                      encoding='utf-8') as f:
                df_list_train = json.load(f)
            df_list_train = [DataFrame.from_dict(d) for d in df_list_train]

            ### load validation set
            if val:
                with open(join(source_path, 'embedder_dataset_val.json'),
                          'r',
                          encoding='utf-8') as f:
                    df_list_val = json.load(f)
                df_list_val = [DataFrame.from_dict(d) for d in df_list_val]

            ### load testing set
            with open(join(source_path, 'embedder_dataset_test.json'),
                      'r',
                      encoding='utf-8') as f:
                df_list_test = json.load(f)
            df_list_test = [DataFrame.from_dict(d) for d in df_list_test]

        ### separate dataset into inputs and labels for train, validation and test
        X_train = sum([df['line'].tolist() for df in df_list_train], [])
        y_train = sum([df['character'].tolist() for df in df_list_train], [])
        X_val = sum([df['line'].tolist()
                     for df in df_list_val], []) if val else None
        y_val = sum([df['character'].tolist()
                     for df in df_list_val], []) if val else None
        X_test = sum([df['line'].tolist() for df in df_list_test], [])
        y_test = sum([df['character'].tolist() for df in df_list_test], [])

        return X_train, y_train, X_val, y_val, X_test, y_test

    def get_triplet_dataset(self,
                            X: List[str],
                            y: List[int],
                            verbose: bool = False) -> List[InputExample]:

        """
        create the triplets for the triplet loss
        the triplet is composed by an 'anchor', a 'positive' sample where the input sentences have the same character label as the 'anchor'
        and a 'negative' sample where the input sentences have a different character label from the 'anchor'

        X (List[str]): list of sentences given as input to the embedder
        y (List[int]): list of character labels (expressed as integers)
        verbose (bool): set to False to avoid printings, default is False

        returns the dataset ready to be passed to the embedder
        """


        assert len(X) == len(y)

        if verbose:
            print('Creating triplets...')

        ### initialize dataset
        examples = []

        ### creating triplets
        for i in tqdm(range(len(X)), disable=not verbose):
            ### selecting anchor label
            y_ref = y[i]

            ### list of possible positive samples (list of the dataset indexes)
            pos_idxs = [y_i for y_i in y if y_i == y_ref]
            random.shuffle(pos_idxs)
            
            ### list of possible negative samples (list of the dataset indexes)
            neg_idxs = [y_i for y_i in y if y_i != y_ref]
            random.shuffle(neg_idxs)

            ### check if we can collect the predefined number of triples starting from the same anchor
            assert len(pos_idxs) > self.n_triplets_x_anchor
            assert len(neg_idxs) > self.n_triplets_x_anchor

            ### selecting the triplets
            ### not all combinations can be saved -> the list of possible indexes are shuffled 
            ### and we consider the possible combination of only the first 'self.n_triplets_x_anchor' indexes
            for pos in pos_idxs[:self.n_triplets_x_anchor]:
                for neg in neg_idxs[:self.n_triplets_x_anchor]:
                    positive = X[pos]
                    negative = X[neg]

                    examples.append(
                        InputExample(texts=[X[i], positive, negative]))

        random.shuffle(examples)

        return examples

    #

    def train_embedder(self,
                       patience: int,
                       train_examples: List[InputExample],
                       val_examples: List[InputExample],
                       save_path: str,
                       verbose: bool = False,
                       statistics_path: str = None) -> None:

        """
        train only the embedder with the triplet loss

        patience (int): patience parameter for early stopping
        train_examples (List[InputExamples]): train dataset
        val_examples (List[InputExamples]): validation dataset
        save_path (str): path where to save the trained embedder
        verbose (bool): verbose parameter, default is False
        statistics_path (str): path where to save accuracy history and other training statistics, default is None

        returns the train history and the validation history of the embedder training
        """

        if verbose:
            print('Training embedder')

        return self.embedder.train(patience=patience,
                                   train_examples=train_examples,
                                   val_examples=val_examples,
                                   save_path=save_path,
                                   verbose=verbose,
                                   statistics_path=statistics_path)

    def train_classifier(self,
                         X_train: List[str],
                         y_train: List[int],
                         verbose: bool = False) -> None:

        """
        train only the classifier (default classifier is knn)
        first the sentences are passed to the embedder and the output vectors build the actual dataset (together with y_train) for the classifier

        X_train (List[str]): list of sentences given as input to the embedder
        y_train (List[str]): list of character labels
        verbose (bool): verbose parameter, default is False
        """

        if verbose:
            print('Training classifier')

        self.classifier = KNeighborsClassifier()
        
        ### get the sentence ambeddings
        train_embeddings = self.embedder.model.encode(
            X_train, show_progress_bar=verbose)
        
        ### fit the classifier
        self.classifier.fit(train_embeddings, y_train)

    def train(self,
              characters_path: str,
              model_path: str,
              train_embedder: bool = False,
              override_data: bool = False,
              merge_sentences: bool = True,
              n_sentences: int = 3,
              verbose: bool = False,
              test: bool = False,
              patience: int = None,
              save_statistics: bool = False,
              shutdown_at_end: str = None) -> None:

        """
        train both the embedder and the classifier
        and also test it if asked ('test'=True)

        characters_path (str): the path of the dataset folder
        model_path (str): the path of the pre-trained model
        train_embedder (bool): if True also the embedder is trained (takes a lot of time), default is False
        override_data (bool): force building the dataset from the csv, if any dataset was already saved, it will be overwritten
        merge_sentences (bool): if True each sample of the returned dataset will have one character and more than one related sentences, default is True
        n_sentences (int): size of the sentence set for each sample in the dataset, default is 3
        verbose (bool): verbose parameter, default is False
        test (bool): if True also test is done, default is False
        patience (int): patience parameter for early stopping
        statistics_path (str): path where to save accuracy history and other training statistics, default is None
        """

        ### if patience>0, create also validation set
        val = isinstance(patience, int) and patience >= 0

        ### if save statistics, define statistics folder
        statistics_path = join(model_path,
                               'statistics') if save_statistics else None

        ### get/create dataset
        X_train, y_train, X_val, y_val, X_test, y_test = self.get_data(
            source_path=characters_path,
            override=override_data,
            merge_sentences=merge_sentences,
            n_sentences=n_sentences,
            verbose=verbose,
            val=val,
        )

        if train_embedder:
            ### create triplet for triplet loss
            train_examples = self.get_triplet_dataset(X_train,
                                                      y_train,
                                                      verbose=verbose)
            ### create triplet for validation loss
            val_examples = self.get_triplet_dataset(
                X_val, y_val, verbose=verbose) if val else None

            ### train embedder
            self.train_embedder(patience=patience,
                                train_examples=train_examples,
                                val_examples=val_examples,
                                save_path=model_path,
                                verbose=verbose,
                                statistics_path=statistics_path)

        ### train classifier
        self.train_classifier(X_train=X_train,
                              y_train=y_train,
                              verbose=verbose)

        ### test the entire model if required
        if test:
            self.test(X_test=X_test,
                      y_test=y_test,
                      verbose=verbose,
                      statistics_path=statistics_path)

        ### used to save energy when training takes long time
        if shutdown_at_end is not None:
            if shutdown_at_end not in ['h', 's']:
                shutdown_at_end = 's'
            system('shutdown -' + shutdown_at_end)

    def test(self,
             X_test: List[str],
             y_test: List[int],
             verbose: bool = False,
             statistics_path: str = None) -> None:

        """
        testing the model: sentences are passed to the embedder and then classified (default by knn)
        
        X_test (List[str]): test input sentences
        y_test (List[int]): test character labels
        statistics_path (str): path where to save accuracy history and other training statistics, default is None
        """

        if verbose:
            print('Testing')

        ### compute the embeddings for the set of sentences
        embeddings = self.embedder.compute(sentences=X_test, verbose=verbose)

        ### classify embeddings with self.classifier
        predictions = self.classifier.predict(embeddings)

        assert len(self.characters) == len(
            set(y_test)
        ), f'Length of self.characters=={len(self.characters)} while y_test has {len(set(y_test))} different labels'

        ### compute confusion metrics of predictions/labels
        cm = confusion_matrix(y_true=y_test,
                              y_pred=predictions,
                              labels=list(range(len(self.characters))),
                              normalize='pred')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=self.characters)

        ### plot confusion metrics results
        disp.plot()
        plt.show()

        ### save confusion metrics results into the statisctics path
        if statistics_path is not None:
            assert exists(statistics_path)
            disp.figure_.savefig(join(statistics_path, 'cm.png'))

    #

    def compute(self,
                sentences: List[str],
                verbose: bool = False,
                count_neighbors: bool = False) -> NDArray:

        """
        predict the character from a set of sentences

        sentences (List[str]): set os sentences to predict the character from
        verbose (bool), verbose parameter, default is False
        count_neighbors (bool): if knn is used not only the final prediction of knn is used for statistics, but all the nighbors also

        returns the normalized distribution (among the characters) of predictions
        """

        ### compute the embeddings
        embeddings = self.embedder.compute(sentences=sentences,
                                           verbose=verbose)
        ### save predictions for each embedding
        ### if count_neighbors the output prediction is substituted with the label of all the selected neighbors
        if count_neighbors:
            predictions = self.classifier.kneighbors(
                embeddings, return_distance=False).ravel()
        else:
            predictions = self.classifier.predict(embeddings)
        
        ### counting number of correct predictions and normalize it
        predictions_count = np.zeros(len(self.characters))
        for pred in predictions:
            predictions_count[pred] += 1
        predictions_count = predictions_count / float(sum(predictions_count))

        # ### softmax
        # predictions = np.exp(predictions) / sum(np.exp(predictions))

        return predictions_count
