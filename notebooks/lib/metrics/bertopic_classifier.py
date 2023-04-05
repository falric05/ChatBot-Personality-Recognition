import os
import random
import json
from os.path import join
from typing import List, Tuple, Dict
from pandas import DataFrame, read_csv
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.dimensionality import BaseDimensionalityReduction
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from torch import load, device
import matplotlib.pyplot as plt

from lib.BBData import character_dict, random_state, model_name
from lib.BBDataLoad import load_char_df, get_chatbot_predictions
from lib.metrics.frequency import sentence_preprocess

characters_all = list(character_dict.keys())
if 'Default' in characters_all:
    characters_all.remove('Default')

random.seed(random_state)

class BERTopic_classifier():
    def __init__(self,
                 path: str = None,
                 from_pretrained: bool = False,
                 use_cuda: bool = False) -> None:
        """
        initialize DistilBert classifier:
        * `embedder_path` (str): the path used to upload the embedder, if default value is given (None) the embedder il built from zero with a default structure 
        * `from_pretrained` (bool): if true the embedder is built starting from embedder_path (must be not None), otherwise the embedder is initialized randomly, default is False
        * `embedding_size` (int): dimensionality of the embedder output, default is 32
        * `use_cuda` (bool): if True the training tensors are transfered to the GPU, default is False
        """

        ### save the characters for training purposes
        self.characters = characters_all

        ### defining train, validation and test percentage size
        self.train_size = 0.85
        self.val_size = 0.05
        self.path = path

        ### initialize the BERTopic model
        empty_dimensionality_model = BaseDimensionalityReduction()
        clf = MLPClassifier(learning_rate='adaptive')
        ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
        vectorizer_model = CountVectorizer(stop_words="english")

        # if not use_cuda:
        #     load(map_location=device('cpu'))

        if not from_pretrained:
            self.topic_model = BERTopic(umap_model=empty_dimensionality_model,
                                        hdbscan_model=clf,
                                        ctfidf_model=ctfidf_model,
                                        vectorizer_model=vectorizer_model, 
                                        verbose=True)
            self.is_trained = False
        else:
            self.topic_model = BERTopic(umap_model=empty_dimensionality_model,
                                        hdbscan_model=clf,
                                        ctfidf_model=ctfidf_model,
                                        vectorizer_model=vectorizer_model, 
                                        verbose=True)
            self.topic_model.load(path)
            self.is_trained = True
        
    #

    def set_characters(self, characters: List[str]) -> None:
        """
        setting the list of characters to consider during training and/or testing
        """
        self.characters = characters

    #

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
        create and save dataset starting from the csv at `source_encoded_path`
        this dataset will be used for the embedder training

        ## Params
        * `val` (bool): if true splits the dataset also into a validation set, and saves it
        * `source_encoded_path` (str): the path of the csv dataset folder, the csv must have a 'line' and a 'character' column
        * `n_shuffles` (int): multiplication factor for the output dataset dimensionality, if `n_shuffles` > 1, the output dataset will be n_shuffles times larger, default is 2
        * `merge_sentences` (bool): if True each sample of the returned dataset will have one character and more than one related sentences, default is True
        * `n_sentences` (int): size of the sentence set for each sample in the dataset, default is 3
        * `save_dataset` (bool): save the dataset built from the csv, default is True
        * `verbose` (bool): set to False to avoid printings, default is False

        ## Return
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
            with open(join(source_encoded_path, 'model_dataset_train.json'),
                      'w',
                      encoding='utf-8') as file:
                json.dump([df.to_dict() for df in df_list_train], file)

            with open(join(source_encoded_path, 'model_dataset_test.json'),
                      'w',
                      encoding='utf-8') as file:
                json.dump([df.to_dict() for df in df_list_test], file)

            if val:
                with open(join(source_encoded_path,
                               'model_dataset_val.json'),
                          'w',
                          encoding='utf-8') as file:
                    json.dump([df.to_dict() for df in df_list_val], file)

        return df_list_train, df_list_val, df_list_test
    
    #

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
        
        ## Params
        * `source_path` (str): the path of the dataset folder
        * `val` (bool): if true splits the dataset also into a validation set, and saves it
        * `override` (bool): force building the dataset from the csv, if any dataset was already saved, it will be overwritten
        * `merge_sentences` (bool): if True each sample of the returned dataset will have one character and more than one related sentences, default is True
        * `n_sentences` (int): size of the sentence set for each sample in the dataset, default is 3
        * `verbose` (bool): set to False to avoid printings, default is False

        ## Returns 
        the saved `X_train`, `y_train`, `X_val`, `y_val`, `X_test`, `y_test`
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
            with open(join(source_path, 'model_dataset_train.json'),
                      'r',
                      encoding='utf-8') as f:
                df_list_train = json.load(f)
            df_list_train = [DataFrame.from_dict(d) for d in df_list_train]

            ### load validation set
            if val:
                with open(join(source_path, 'model_dataset_val.json'),
                          'r',
                          encoding='utf-8') as f:
                    df_list_val = json.load(f)
                df_list_val = [DataFrame.from_dict(d) for d in df_list_val]

            ### load testing set
            with open(join(source_path, 'model_dataset_test.json'),
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
    
    #

    def get_mapping(self) -> Tuple[Dict, Dict]:
        """
        BERTopic is a topic modelling model, thus predictions of BERTopic are topics, which could not mathc directly
        with original y labels. Thus it is necessary to retrieve a map funcion.

        ## Returns
        * `mapping`: a dictionary which keys are topics and values are document labels
        * `inv_mapping`: a dictionary which keys are original labels and values are document labels
        """
        assert self.is_trained, 'The model must be trained before to make prediction. Please call method `train`!'

        mapping = self.topic_model.topic_mapper_.get_mappings()
        inv_mapping = {v: k for k, v in mapping.items()}
        return mapping, inv_mapping
   
    #
    
    def train(self, train_data: Tuple[List[str], List[int]]) -> None:
        """
        Train BERTopic model

        ### Params
        * `train_data`: couple of list of documents and labels
        """
        X_train, y_train = train_data
        # fit the model
        y_train_pred, _ = self.topic_model.fit_transform(X_train, y=y_train)
        self.topic_model.save(self.path)
        self.is_trained = True
        # The resulting topics may be a different mapping from the y labels. 
        # To map y to topics, we can run the following:
        _, inv_mapping = self.get_mapping()
        y_pred_mapped = [inv_mapping[val] for val in y_train_pred]
        # plot confusion matrix
        cm = confusion_matrix(y_true=y_train,
                              y_pred=y_pred_mapped,
                              labels=list(range(len(self.characters))),
                              normalize='pred')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=self.characters)
        disp.plot(xticks_rotation='vertical')


    # 

    def predict(self, test_data: Tuple[List[str], List[int]]) -> List[int]:
        """
        Test BERTopic model
        
        ### Params
        * `test_data`: couple of list of documents and labels
        """
        assert self.is_trained, 'The model must be trained before to make prediction. Please call method `train`!'

        X_test, y_test = test_data

        y_test_pred, _ = self.topic_model.transform(X_test)
        _, inv_mapping = self.get_mapping()
        y_pred_mapped = [inv_mapping[val] for val in y_test_pred]
        ### compute confusion metrics of predictions/labels
        cm = confusion_matrix(y_true=y_test,
                              y_pred=y_pred_mapped,
                              labels=list(range(len(self.characters))),
                              normalize='pred')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=self.characters)
        disp.plot(xticks_rotation='vertical')


        return y_pred_mapped
    
    #

    def plot_documents(self, x: List[str]):
        return self.topic_model.visualize_documents(x, title='Characters lines and Topics')