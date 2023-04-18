import os
from os.path import join
import pandas as pd
pd.options.mode.chained_assignment = None
from typing import List, Tuple
import string
import json
from tqdm import tqdm
import numpy as np 
from typing import List
#
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
#
import torch
from sentence_transformers import SentenceTransformer
from torch import nn
from torch.nn import Linear, Dropout, ReLU, Softmax, CrossEntropyLoss
from torch.optim import Optimizer, Adam, AdamW
# from torch.nn import functional as pt_f
from torchmetrics import F1Score
from torch_geometric.data import Data, Batch, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn.conv import SAGEConv
from torch_geometric.nn.glob import global_mean_pool
#
from lib.BBData import character_dict, random_state, model_name
from lib.BBDataLoad import load_char_df, get_chatbot_predictions

characters_all = list(character_dict.keys())


class TVShowGraphDataset(Dataset):
    def __init__(self, 
                 root_dataset, 
                 flag,
                 transform = None, 
                 pre_transform = None):
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
        # print("idx_data", idx_data)
        return idx_data
    
class PersGraphNeuralNetwork(nn.Module):
    def __init__(self, in_channels: int, n_characters: int) -> None:
        super(PersGraphNeuralNetwork, self).__init__()
        self.sage1 = SAGEConv(in_channels=in_channels, out_channels=in_channels, bias=True)
        self.sage2 = SAGEConv(in_channels=in_channels, out_channels=in_channels, bias=True)
        self.dropout = Dropout(p=0.2)
        self.linear1 = Linear(in_features=in_channels, out_features=128)
        self.linear2 = Linear(in_features=128, out_features=32)
        self.linear3 = Linear(in_features=32, out_features=n_characters)
        self.activation = ReLU(inplace=True)
        self.last_activation = Softmax(dim=1)

    #

    @staticmethod
    def cosine_attention(mtx1, mtx2):
        def div_with_small_value(n, d, eps=1e-8):
            d = d * (d > eps).float() + eps * (d <= eps).float()
            return n / d
        
        v1_norm = mtx1.norm(p=2, dim=2, keepdim=True)
        v2_norm = mtx2.norm(p=2, dim=2, keepdim=True).permute(0, 2, 1)
        a = torch.bmm(mtx1, mtx2.permute(0, 2, 1))
        d = v1_norm * v2_norm
        
        return div_with_small_value(a, d)

    #

    def forward(self, x_in: Batch):
        ###
        x_out = self.sage1(x=x_in.x, edge_index=x_in.edge_index)
        x_out = self.activation(x_out)
        x_out = self.dropout(x_out)
        ###
        x_out = self.sage2(x=x_out, edge_index=x_in.edge_index)
        x_out = self.activation(x_out)
        x_out = self.dropout(x_out)
        ###
        x_out = global_mean_pool(x=x_out, batch=x_in.batch)
        ###
        x_out = self.linear1(x_out)
        x_out = self.activation(x_out)
        x_out = self.linear2(x_out)
        x_out = self.activation(x_out)
        x_out = self.linear3(x_out)
        # print(x_out.shape)
        x_out = self.last_activation(x_out)
        return x_out

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

        ### create one hot encoder labels
        self.onehotenc = OneHotEncoder(handle_unknown='ignore')
        self.onehotenc.fit(np.arange(len(self.characters)).reshape((-1, 1)))
        # print(self.onehotenc.transform(np.array([6]).reshape((-1, 1))).toarray())

        ### create the sentence embedder model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        if not from_pretrained:
            self.model = PersGraphNeuralNetwork(self.embedding_model.get_sentence_embedding_dimension(),
                                                len(self.characters))

        ### move model to cude is asked
        self.use_cuda = use_cuda
        if use_cuda:
            assert torch.cuda.is_available()
            self.device = torch.device('cuda')  # pylint: disable = no-member
            self.embedding_model.to(self.device)
            self.model.to(self.device)

    #

    def set_characters(self, characters: List[str]) -> None:
        """
        setting the list of characters to consider during training and/or testing
        """
        self.characters = characters
        self.onehotenc = OneHotEncoder(handle_unknown='ignore')
        self.onehotenc.fit(np.arange(len(self.characters)).reshape((-1,1)))

    #

    def __graph_preprocess(self,
                           sns_list: List[str],
                           min_chat_len: int = 3,
                           max_chat_len: int = 7,
                           seed: int = 42,
                           verbose: bool = False):
        """
        preprocess a list of a character lines to create a chat graph
        
        ### Params 
        * sns_list
        """
        np.random.seed(seed)
        sns_list_prepr = []
        pbar = tqdm(range(len(sns_list)-max_chat_len), position=0, 
                    leave=True, disable=(not verbose))
        for i in pbar:
            j = np.random.randint(min_chat_len, max_chat_len) + 1
            sentences = [s.lower().translate(
                            str.maketrans('', '', string.punctuation)
                        ) for s in sns_list[i:i+j]]
            node = {'sentence_edges': [[k for k in range(j-1)],
                                       [k+1 for k in range(j-1)]
                                    ],
                    'sentences': self.embedding_model.encode(sentences).tolist(),
                    }
            sns_list_prepr.append(node)
        return sns_list_prepr

    #

    def __graph_to_pygObject(self, path_graphs, path_pt_graphs, verbose=False):
        for k in tqdm(['train', 'valid', 'test'], position=0, leave=True, disable=(not verbose)):
            index = 0
            for jsonfile in os.listdir(join(path_graphs, k+'Set')):
                graph = os.path.join(path_graphs, k+'Set', jsonfile)

                with open(graph, "r") as file:
                    graphConversation = json.load(file)
                    ### convert lists to tensors
                    graph_sentences = graphConversation["x"]["sentences"]
                    graph_edges = graphConversation["x"]["sentence_edges"]
                    graph_labels = graphConversation["y"]
                    
                    # graph_sentences = []
                    # for s in graphConversation["x"]["sentences"]:
                    #     graph_sentences.append(torch.tensor(s, dtype=torch.float))
                    # graph_edges = []
                    # for e in graphConversation["x"]["sentence_edges"]:
                    #     graph_edges.append(torch.tensor(e, dtype=torch.long))
                    
                    # graph_labels = [torch.tensor(graphConversation["y"], dtype=torch.float)]

                    # TODO: modifica
                    # torch.save(Data(sentences=graph_sentences, sentences_edges=graph_edges, labels=graph_labels),
                    #            "{}/{}Set/{}.pt".format(path_pt_graphs, k, index))
                    torch.save(Data(x=torch.tensor(graph_sentences, dtype=torch.float), 
                                    edge_index=torch.tensor(graph_edges, dtype=torch.long), 
                                    labels=torch.tensor([graph_labels], dtype=torch.float)
                                ), "{}/{}Set/{}.pt".format(path_pt_graphs, k, index))
                    index += 1

    #

    def create_data(
            self,
            data_folder: str,
            characters_folder: str,
            n_sentences_range: int = (3, 7),
            override_graphs: bool = False,
            verbose: bool = False) -> None:

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
        ptgraphs_path = join(data_folder, 'pt_graphs')

        if override_graphs:
            print("Removing previous dataset...")
            for dir in os.listdir(graphs_path):
                for file in os.listdir(join(graphs_path, dir)):
                    os.remove(join(graphs_path, dir, file))
            for dir in os.listdir(ptgraphs_path):
                for file in os.listdir(join(ptgraphs_path, dir)):
                    os.remove(join(ptgraphs_path, dir, file))

        if not os.path.exists(graphs_path):
            os.mkdir(graphs_path)
            os.mkdir(join(graphs_path, 'trainSet'))
            os.mkdir(join(graphs_path, 'validSet'))
            os.mkdir(join(graphs_path, 'testSet'))

        if not os.path.exists(ptgraphs_path):
            os.mkdir(join(ptgraphs_path))
            os.mkdir(join(ptgraphs_path, 'trainSet'))
            os.mkdir(join(ptgraphs_path, 'validSet'))
            os.mkdir(join(ptgraphs_path, 'testSet'))
        

        ### load dataframes and divide them in train, valid and test set
        # prepare the dataset of all documents
        character_docs = dict()
        for c in self.characters:
            if c == 'Default':
                ### read Default dataset and sample 0.02 fraction of the whole dataset
                df = pd.read_csv(os.path.join(characters_folder, c, f'{c}.tsv'), 
                                names=[c, 'response'], sep='\t')
                df = df.sample(frac=0.02, random_state=random_state)
                df['response'] = df['response'].apply(lambda x: x[3:])
                df[c] = df[c].apply(lambda x: x[3:])
                df_train, df_test  = train_test_split(df, test_size=self.test_size, shuffle=False)
                df_train, df_valid = train_test_split(df_train, test_size=self.valid_size, shuffle=False)
            else:
                ### read other dataset 
                df = pd.read_csv(os.path.join(characters_folder, c, f'{c}.csv'))
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

        # CREATE JSON OF GRAPHS
        ### preprocess data
        if verbose:
            print('Preprocess data lines...')
        
        ### for every kind of 
        for idx_set in ['train', 'valid', 'test']:
            j = 0
            pbar_characters = tqdm(self.characters, position=0, disable=(not verbose))
            for c_idx, c in enumerate(pbar_characters):
                pbar_characters.set_postfix({'set':idx_set, 'character':c})
                graphs = self.__graph_preprocess(character_docs[c][idx_set], 
                                                 min_chat_len=n_sentences_range[0],
                                                 max_chat_len=n_sentences_range[1])
                y_onehot = self.onehotenc.transform(np.array([c_idx]).reshape((-1, 1))).toarray()[0]
                # print(c, y_onehot)
                for i in range(len(graphs)):
                    json_path = join(graphs_path, idx_set+'Set', f'raw_{i+j:08d}.json')
                    assert not os.path.exists(json_path), f'file {json_path} already exists'
                    with open(json_path, 'w') as f:
                        json.dump({'x': graphs[i], 'y': list(y_onehot)}, f)
                # pbar_characters.update()
                j += i + 1
        # CREATE PT DATA OF GRAPHS
        ### preprocess data
        if verbose:
            print('Transform data in pt objects...')
        self.__graph_to_pygObject(graphs_path, ptgraphs_path, verbose=verbose)

        ### load dataset
        # TODO: return data loader
        return None
    
    #

    def get_data(self, path_to_ptgraph: str, get_validation: bool = True, batch_size: int =16) -> Tuple[DataLoader, DataLoader, DataLoader]:
        train_dataset = TVShowGraphDataset(root_dataset=path_to_ptgraph, flag='train')
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

        if get_validation:
            valid_dataset = TVShowGraphDataset(root_dataset=path_to_ptgraph, flag='valid')
            valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)
        else:
            valid_loader = None

        test_dataset = TVShowGraphDataset(root_dataset=path_to_ptgraph, flag='test')
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

        return train_loader, valid_loader, test_loader 

    #

    def __validate(self, valid_loader: DataLoader, optimizer: Optimizer, loss: CrossEntropyLoss, idx_epoch: int):
        valid_loss    = 0.0
        valid_f1score = 0.0
        pbar = tqdm(valid_loader, desc="## Valid epoch = {}".format(idx_epoch), leave=True, 
                    postfix={'valid_loss':     valid_loss,
                              'valid_f1score': valid_f1score})
        for _idx_bt, _batch in enumerate(pbar):
            self.model.eval()
            ### move tensor to device
            if self.use_cuda:
                assert torch.cuda.is_available()
                _batch.to(self.device)
            ### get the onehot vector and argmax labels
            _ys_true_onehot = _batch['labels']
            _ys_true = torch.argmax(_ys_true_onehot, dim=1)
            _ys_true.to(self.device)
            ### make predictions
            optimizer.zero_grad()
            ys_pred_onehot = self.model(_batch)
            ys_pred = torch.argmax(ys_pred_onehot, dim=1)
            ys_pred.to(self.device)
            ### compute loss
            batch_loss = loss(ys_pred_onehot, _ys_true_onehot)
            valid_loss += float(batch_loss)
            ### compute f1score
            ys_pred = ys_pred.to('cpu')
            _ys_true = _ys_true.to('cpu')
            batch_f1score = f1_score(_ys_true.numpy(), ys_pred.numpy(), average='macro')
            valid_f1score += float(batch_f1score)
            ### update progressbar
            pbar.set_postfix({'valid_loss':    valid_loss / (_idx_bt+1),
                              'valid_f1score': valid_f1score / (_idx_bt+1)})
        
        return valid_loss / (_idx_bt+1), valid_f1score / (_idx_bt+1)

    #

    def __train_one_epoch(self, train_loader: DataLoader, valid_loader: DataLoader, 
                          loss: CrossEntropyLoss, optimizer: Optimizer, 
                          idx_epoch: int, best_loss, patience) -> Tuple[float, float, float, float]:
        self.model.train()
        train_loss    = 0.0
        train_f1score = 0.0
        pbar = tqdm(train_loader, desc="Training epoch = {}".format(idx_epoch))
        for _idx_bt, _batch in enumerate(pbar):
            if self.use_cuda:
                assert torch.cuda.is_available()
                _batch.to(self.device)

            self.model.train()
            ### get the onehot vector and argmax labels
            _ys_true_onehot = _batch['labels']
            _ys_true = torch.argmax(_ys_true_onehot, dim=1)
            _ys_true.to(self.device)
            ### make predictions
            optimizer.zero_grad()
            ys_pred_onehot = self.model(_batch)
            ys_pred = torch.argmax(ys_pred_onehot, dim=1)
            ys_pred.to(self.device)
            ### compute loss
            batch_loss = loss(ys_pred_onehot, _ys_true_onehot)
            ### backward pass
            batch_loss.backward()
            optimizer.step()
            train_loss += float(batch_loss)
            ### compute f1score
            ys_pred = ys_pred.to('cpu')
            _ys_true = _ys_true.to('cpu')
            batch_f1score = f1_score(_ys_true.numpy(), ys_pred.numpy(), average='macro')
            train_f1score += float(batch_f1score)
            ### update progressbar
            pbar.set_postfix({'training_loss':    train_loss / (_idx_bt+1),
                              'training_f1score': train_f1score / (_idx_bt+1)})
        ### validation step
        valid_loss, valid_f1score = self.__validate(valid_loader, optimizer, loss, idx_epoch)
        train_loss /= _idx_bt
        train_f1score /= _idx_bt
        pbar.set_postfix({'training_loss':    train_loss,
                          'training_f1score': train_f1score,
                          'validation_loss':    valid_loss,
                          'validation_f1score': valid_f1score})
        return train_loss, train_f1score, valid_loss, valid_f1score
    
    #

    def train(self, X_train: DataLoader, X_valid: DataLoader,
              optimizer: str = 'adam', epochs: int = 3, 
              patience: int = 3) -> Tuple[List[float], List[float], List[float], List[float]]:
        ### cross entropy loss
        loss = CrossEntropyLoss()
        ### optizer
        if optimizer == 'adam':
            optimizer = Adam(self.model.parameters(), lr=1e-3)
        elif optimizer == 'adamw':
            optimizer = AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        else:
            raise NotImplementedError()
        train_loss_history = []
        valid_loss_history = []
        train_f1score_history = []
        valid_f1score_history = []
        for e in range(epochs):
            ### execute one epoch
            epoch_result = self.__train_one_epoch(X_train, X_valid, loss, optimizer, e, None, patience)
            ### store results
            train_loss_history.append(epoch_result[0])
            train_f1score_history.append(epoch_result[1])
            valid_loss_history.append(epoch_result[2])
            valid_f1score_history.append(epoch_result[3])
        ### return histories
        return train_loss_history, train_f1score_history, valid_loss_history, valid_f1score_history
    
    def compute(self, X_test: DataLoader):
        # TODO: da implementare ancora
        test_f1score = 0.0
        pbar = tqdm(X_test, desc="## Test ")
        ys_pred = []
        _ys_true = []
        for _idx_bt, _batch in enumerate(pbar):
            _batch.to(self.device)
            self.model.eval()
            ### get the onehot vector and argmax labels
            _ys_true_onehot = _batch['labels']
            _ys_true.append(torch.argmax(_ys_true_onehot, dim=1))
            ### make predictions
            ys_pred_onehot = self.model(_batch)
            ys_pred.append(torch.argmax(ys_pred_onehot, dim=1))

        ### compute f1score
        ys_pred = torch.hstack(ys_pred).to('cpu')
        _ys_true = torch.hstack(_ys_true).to('cpu')
        f1score = f1_score(_ys_true.numpy(), ys_pred.numpy(), average='macro')
        return f1score