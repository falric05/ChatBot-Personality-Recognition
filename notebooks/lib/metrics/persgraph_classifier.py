import os
from os.path import join
import torch
import pandas as pd
from pandas import DataFrame
import random
from collections import Counter
from typing import List, Tuple, Dict
import string
import json
from tqdm import tqdm
import numpy as np 
import nltk
from typing import Dict, List
#
#from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
#
import torch
from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data, Batch, Dataset
from torch_geometric.loader import DataLoader
#
from lib.BBData import character_dict, random_state, model_name
from lib.BBDataLoad import load_char_df, get_chatbot_predictions
from lib.metrics.frequency import sentence_preprocess

characters_all = list(character_dict.keys())
if 'Default' in characters_all:
    characters_all.remove('Default')


class TVShowGraphDataset(Dataset):
    def __init__(self, 
                 root_dataset, 
                 flag,
                 transform = None, 
                 pre_transform = None, 
                 ):
        """
        flag:
        """
        assert flag in ['train', 'valid', 'test']
        super(TVShowGraphDataset, self).__init__(None, transform, pre_transform)

        self.flag = flag
        self.rootDS = join(root_dataset, flag+'Set')
        self.rootDS_files = os.listdir(self.rootDS)

    @staticmethod
    def _list_files_for_pt(the_path):
        files = []
        for name in os.listdir(the_path):
            if os.path.splitext(name)[-1] == '.pt':
                files.append(name)
        return files

    def len(self):
        return len(self.rootDS_files)

    def get(self, idx):
        idx_infolder = self.rootDS_files[idx]
        idx_data = torch.load(join(self.rootDS, idx_infolder))
        return idx_data

class PersGRAPH_classifier():
    def __init__(self,
                 path: str = None,
                 from_pretrained: bool = False,
                 use_cuda: bool = False) -> None:
        ### save the characters for training purposes
        self.characters = characters_all

        ### defining train, validation and test percentage size
        self.test_size  = 0.25
        self.valid_size = 0.15

        ### create the sentence embedder model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    #

    def set_characters(self, characters: List[str]) -> None:
        """
        setting the list of characters to consider during training and/or testing
        """
        self.characters = characters

    #

    def __graph_preprocess(self,
                           sns_list: List[str],
                           min_chat_len: int = 3,
                           max_chat_len: int = 7,
                           seed: int = 42,
                           verbose: bool = False):
        print()
        np.random.seed(seed)
        sns_list_prepr = []
        pbar = tqdm(range(len(sns_list)-max_chat_len), position=0, 
                    leave=True, disable=(not verbose))
        for i in pbar:
            j = np.random.randint(min_chat_len, max_chat_len) + 1
            sentences = [s.lower().translate(
                                str.maketrans('', '', string.punctuation)
                            ) for s in sns_list[i:i+j]]
            node = {'sentence_edges': [[k, k+1] for k in range(j-1)],
                    'sentences': self.embedding_model.encode(sentences).tolist(),
                    }
            sns_list_prepr.append(node)
        return sns_list_prepr

    #

    def __graph_to_pygObject(path_graphs, verbose=False):
        index = 0
        for k in tqdm(['train', 'valid', 'test'], position=0, leave=True, disable=(not verbose)):
            for jsonfile in os.listdir(join(path_graphs, k+'Set')):
                graph = os.path.join(path_graphs, k+'Set', jsonfile)

                with open(graph, "r", encoding="utf-8") as file:
                    graphConversation = json.load(file)

                    graph_edges = graphConversation["x"]["sentence_edges"]
                    graph_sentences = graphConversation["x"]["sentences"]
                    graph_label = graphConversation["y"]

                    torch.save(Data(sentences_edges=graph_edges,sentences=graph_sentences,
                                    label=graph_label),"/content/pt_graphs/{}Set/{}.pt".format(k, index))
                    index += 1

    #

    def create_data(
            self,
            data_folder: str,
            n_sentences_range: int = (3, 7),
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
        graphs_path = join(data_folder, 'graphs')
        if not os.path.exists(graphs_path):
            os.mkdir(join(graphs_path))
            os.mkdir(join(graphs_path, 'train'))
            os.mkdir(join(graphs_path, 'valid'))
            os.mkdir(join(graphs_path, 'test'))
        
        ptgraphs_path = join(data_folder, 'pt_graphs')
        if not os.path.exists(ptgraphs_path):
            os.mkdir(join(ptgraphs_path))
            os.mkdir(join(ptgraphs_path, 'train'))
            os.mkdir(join(ptgraphs_path, 'valid'))
            os.mkdir(join(ptgraphs_path, 'test'))
        
        ### load dataframes and divide them in train, valid and test set
        # prepare the dataset of all documents
        character_docs = dict()
        for c in self.characters:
            if c == 'Default':
                ### read Default dataset and sample 0.02 fraction of the whole dataset
                df = pd.read_csv(os.path.join(data_folder, c, f'{c}.tsv'), 
                                names=[c, 'response'], sep='\t')
                df = df.sample(frac=0.02, random_state=random_state)
                df['response'] = df['response'].apply(lambda x: x[3:])
                df[c] = df[c].apply(lambda x: x[3:])
                df_train, df_test  = train_test_split(df, test_size=self.test_size, shuffle=False)
                df_train, df_valid = train_test_split(df_train, test_size=self.valid_size, shuffle=False)
            else:
                ### read other dataset 
                df = pd.read_csv(os.path.join(data_folder, c, f'{c}.csv'))
                df_train, df_test  = train_test_split(df, test_size=self.test_size, shuffle=False)
                df_train, df_valid = train_test_split(df_train, test_size=self.valid_size, shuffle=False)
            ### convert dataframes to list
            df_train['response'] = df_train['response'].apply(lambda x: x.replace('\'', ' '))
            df_valid['response'] = df_valid['response'].apply(lambda x: x.replace('\'', ' '))
            df_test['response'] = df_test['response'].apply(lambda x: x.replace('\'', ' '))
            ### attach the list to the dictionary of documents of character
            character_docs[c] = {'train': df_train['response'].tolist(),
                                 'valid': df_valid['response'].tolist(),
                                 'test':  df_test['response'].tolist(),}
        print()
        print()
        # CREATE JSON OF GRAPHS
        ### preprocess data
        if verbose:
            print('Preprocess data lines...')
        j = 0
        ### for every kind of 
        for k in ['train', 'valid', 'test']:
            pbar_characters = tqdm(self.characters, position=0, leave=True, disable=(not verbose))
            for c in pbar_characters:
                pbar_characters.set_description_str(k + ': #### '+c)
                graphs = self.__graph_preprocess(character_docs[c][k], min_chat_len=n_sentences_range[0],
                                                 max_chat_len=n_sentences_range[1])
                for i in range(len(graphs)):
                    with open(join(graphs_path, k+'Set', f'raw_{i+j:05d}.json'), 'w') as f:
                        json.dump({'x': graphs[i],
                                   'y': self.characters.index(c)}, 
                                    f)
                pbar_characters.update()
                j = i + 1
        print()
        print()
        # CREATE PT DATA OF GRAPHS
        ### preprocess data
        if verbose:
            print('Transform data in pt objects...')
        self.__graph_to_pygObject(ptgraphs_path, verbose=verbose)

        ### load dataset
        if verbose:
            print('Loading data lines...')

        return df_list_train, df_list_val, df_list_test
    
    #