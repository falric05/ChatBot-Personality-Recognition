import random
from multiprocessing import cpu_count
from os.path import exists, join
from os import mkdir
from typing import List, Tuple

import numpy as np
import torch
import json
from joblib import Parallel, delayed
from numpy.typing import NDArray
from sentence_transformers import models, SentencesDataset
from sentence_transformers.readers import InputExample
from sentence_transformers.SentenceTransformer import SentenceTransformer
from sentence_transformers.losses import TripletLoss, TripletDistanceMetric
#from sentence_transformers.evaluation import TripletEvaluator
from .triplet_evaluator import TripletEvaluator  # pylin: disable = relative-beyond-top-level
from torch.nn import Identity
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.BBData import character_dict, random_state

characters_all = list(character_dict.keys())
if 'Default' in characters_all:
    characters_all.remove('Default')

# random.seed(random_state)


class BarneyEmbedder:

    def __init__(self,
                 embedding_size: int,
                 embedder_path: str = None,
                 from_pretrained: bool = False,
                 use_cuda: bool = False) -> None:

        """
        initialize DistilBert Embedder

        embedding_size (int): dimensionality of the embedder output
        embedder_path (str): the path used to upload the embedder, if default (None) the embedder il built from zero with a default structure
        from_pretrained (bool): if true the embedder is built starting from embedder_path (must be not None), otherwise the embedder is initialized randomly, default is False 
        use_cuda (bool): if True the training tensors are transfered to the GPU, default is False
        """

        ### set model params
        self.embedding_size: int = embedding_size

        ### set some training params
        self.batch_size: int = 64
        self.lr: float = 1e-3
        self.epochs: int = 5
        self.training_steps: int = 50
        self.margin: int = self.embedding_size * 10

        ### create the model
        self.model = self.create_model(embedder_path=embedder_path,
                                       from_pretrained=from_pretrained)
        
        ### move model to cude is asked
        if use_cuda:
            assert torch.cuda.is_available()
            device = torch.device('cuda')  # pylint: disable = no-member
            self.model.to(device)

    #

    def create_model(self,
                     embedder_path: str = None,
                     from_pretrained: bool = False) -> SentenceTransformer:

        """
        create the embedder model

        embedder_path (str): the path used to upload the embedder, if default (None) the embedder il built from zero with a default structure
        from_pretrained (bool): if true the embedder is built starting from embedder_path (must be not None), otherwise the embedder is initialized randomly, default is False 

        returns the embedder model
        """

        if not from_pretrained:
            ### build embedder from "scratch"
            model = SentenceTransformer('distilbert-base-nli-mean-tokens')
            model.train()
            dense = models.Dense(
                in_features=model.get_sentence_embedding_dimension(),
                out_features=self.embedding_size,
                activation_function=Identity())
            model.add_module('dense', dense)
        else:
            ### load embedder from path
            assert embedder_path is not None, 'Give model path to start from pre-trained model'
            assert exists(embedder_path
                          ), f'Current path "{embedder_path}" does not exists'
            model = SentenceTransformer(embedder_path)

        return model

    #

    def semi_hard_negative_mining(
            self,
            examples: List[InputExample],
            model: SentenceTransformer,
            margin: float,
            verbose: bool = False,
            parallel: bool = False) -> List[InputExample]:

        """
        filters the dataset for the triplet loss training, splitting it into easy_positives, semi_hard_negatives and hard_negatives

        examples (List[InputExample]): the dataset to filter
        model (SentenceTransformer): model used to have a first estimation of the embeddings
        margin (float): margin parameter of the triple loss
        verbose (bool): verbose parameter, default is False
        parallel (bool): parallel the mining work, highly discouraged if working with cuda, default is False

        returns the semi_hard_negatives dataset, the dataset_length, the number of hard negatives and the number of easy positives
        """

        def _check_triplet(
                triplet_embedded: NDArray) -> Tuple[bool, bool, bool]:
            """
            returns
                - is triplet semi-hard
                - is triplet hard
                - is triplet easy
            """

            ### split the triplet in anchor, positive and negative
            anchor_emb = triplet_embedded[0]
            positive_emb = triplet_embedded[1]
            negative_emb = triplet_embedded[2]

            ### compute the distance anchor-positive
            dist_ap = np.linalg.norm(anchor_emb - positive_emb)
            ### compute the distance anchor-negative
            dist_an = np.linalg.norm(anchor_emb - negative_emb)

            ### distinguish easy, semi-hard and hard cases
            if dist_ap < dist_an:
                if dist_an < dist_ap + margin:
                    return True, False, False
                else:
                    return False, False, True
            else:
                return False, True, False

        ###############  create all embeddings  ###############
        if verbose:
            print('Creating all updated embeddings...')
        concat_examples = []
        for input_example in examples:
            concat_examples += input_example.texts

        ### concatenating triplets (anchor, positive, negative),
        ### the total lenght must be divisible by 3
        example_len = len(concat_examples)
        assert example_len % 3 == 0

        concat_embeddings = model.encode(concat_examples,
                                         show_progress_bar=verbose,
                                         batch_size=self.batch_size * 2)
        embeddings = [[
            concat_embeddings[i], concat_embeddings[i + 1],
            concat_embeddings[i + 2]
        ] for i in range(0, example_len - 2, 3)]

        assert len(embeddings) == len(
            examples
        ), f'Lenght of embeddings ({len(embeddings)}) != lenght of examples ({len(examples)})'

        #

        ############### check examples hardness ###############
        if verbose:
            print('Checking triplets hardness...')
        if parallel:
            n_jobs = cpu_count()
            hardness_idxs = Parallel(n_jobs=n_jobs)(
                delayed(_check_triplet)(triplet)
                for triplet in tqdm(embeddings, disable=not verbose))
        else:
            hardness_idxs = [
                _check_triplet(triplet)
                for triplet in tqdm(embeddings, disable=not verbose)
            ]

        ### unpack indexes
        semi_hard_idxs = []
        hard_idxs = []
        easy_idxs = []
        for idx in hardness_idxs:
            semi_hard_idxs.append(idx[0])
            hard_idxs.append(idx[1])
            easy_idxs.append(idx[2])

        #

        ###############     filter dataset      ###############
        if verbose:
            print('Filtering dataset...')
        filtered = []
        for i, input_example in enumerate(examples):
            if semi_hard_idxs[i]:
                filtered.append(input_example)

        ###############    count statistics     ###############
        dataset_count = sum(semi_hard_idxs)
        hard_negatives_count = sum(hard_idxs)
        easy_positives_count = sum(easy_idxs)

        return filtered, dataset_count, hard_negatives_count, easy_positives_count

    #

    def train(self,
              patience: int,
              train_examples: List[InputExample],
              val_examples: List[InputExample],
              save_path: str,
              verbose: bool = False,
              statistics_path: str = None) -> List[float]:

        """
        train the DistilBert embedder with TripleLoss
        and saves train and validation history (accuracy values over the epochs)

        patience (int): patience parameter for early stopping
        train_examples (List[InputExamples]): train dataset
        val_examples (List[InputExamples]): validation dataset
        save_path (str): path where to save the trained embedder
        verbose (bool): verbose parameter, default is False
        statistics_path (str): path where to save accuracy history and other training statistics, default is None

        returns the train and validation history (of the accuracy metric)
        """

        ### if patience>0, create also validation set
        val = isinstance(patience, int) and patience >= 0

        ### set loss
        train_loss = TripletLoss(
            model=self.model,
            triplet_margin=self.margin,
            distance_metric=TripletDistanceMetric.EUCLIDEAN,
        )

        ### prepare evaluator and statistics
        train_accuracy = []
        train_accuracy_max = 0
        train_length = len(train_examples)
        easy_positives_max = 0

        val_accuracy = None  # to avoid return error
        val_length = len(val_examples)
        if val:
            triplet_evaluator = TripletEvaluator.from_input_examples(
                examples=val_examples,
                batch_size=self.batch_size,
                verbose=verbose,
            )

            val_accuracy = []
            val_accuracy_max = 0
            patience_count = 0

        ### set learning rate steps
        decrease_factor = 0.5
        self.lr /= decrease_factor

        ### train loop
        for step in range(self.training_steps):

            ### semi-hard negative mining
            if verbose:
                print('#' * 100)
                print(f'step {step+1}/{self.training_steps}')

            filtered_examples, dataset_count, hard_negatives_count, easy_positives_count = self.semi_hard_negative_mining(
                train_examples, self.model, self.margin, verbose=verbose)
            
            ### prepare training set
            train_dataset = SentencesDataset(filtered_examples, self.model)
            train_dataloader = DataLoader(train_dataset,
                                          shuffle=True,
                                          batch_size=self.batch_size)

            ### update train accuracy
            last_train_accuracy = 1 - hard_negatives_count / train_length
            train_accuracy.append(last_train_accuracy)
            train_accuracy_improvement = last_train_accuracy - train_accuracy_max
            if train_accuracy_improvement > 0:
                train_accuracy_max = last_train_accuracy
            dataset_line = f'\nDataset lenght    : {dataset_count}'
            train_accuracy_line = f'\nTrain Accuracy    : {last_train_accuracy:6.2%}\t({round(train_accuracy_improvement*100, 2):+}%)'

            ### check easy positives increment
            easy_positives_improvement = easy_positives_count - easy_positives_max
            if easy_positives_improvement > 0:
                easy_positives_max = easy_positives_count
            easy_positives_line = f'\nEasy Positives    : {easy_positives_count:6}\t({easy_positives_improvement:+})'

            ### validation
            if val:
                assert val_length > 0

                if verbose:
                    print('Validation...')

                last_val_accuracy = triplet_evaluator(model=self.model)

                val_accuracy.append(last_val_accuracy)

                ### check improvements and update patience_count
                val_accuracy_improvement = last_val_accuracy - val_accuracy_max
                if val_accuracy_improvement > 0:
                    val_accuracy_max = last_val_accuracy
                val_accuracy_line = f'\nVal Accuracy      : {last_val_accuracy:6.2%}\t({round(val_accuracy_improvement*100, 2):+}%)'

                patience_reset = val_accuracy_improvement > 0 or (
                    last_val_accuracy == val_accuracy_max
                    and train_accuracy_improvement > 0) or (
                        last_val_accuracy == val_accuracy_max
                        and last_train_accuracy == train_accuracy_max
                        and easy_positives_improvement > 0)

                patience_count = 0 if patience_reset else patience_count + 1

            if verbose:
                print('#' * 60)
                if val:
                    print(val_accuracy_line)
                print(dataset_line)
                print(train_accuracy_line)
                print(easy_positives_line)
                print('#' * 60, '\n')

            if patience_count >= patience:
                break

            ### training
            if verbose:
                print('Training...')

            self.model.fit([(train_dataloader, train_loss)],
                           epochs=self.epochs,
                           optimizer_params={'lr': self.lr},
                           show_progress_bar=verbose)

            ### decrease lr
            self.lr *= decrease_factor

        ### save model
        if save_path is None:
            print('Save path not setted, model will note be saved')
        else:
            self.model.save(save_path)

        if statistics_path is not None:
            if not exists(statistics_path):
                mkdir(statistics_path)
            with open(join(statistics_path, 'train_accuracy.json'),
                      'w',
                      encoding='utf-8') as file:
                json.dump(train_accuracy, file)
            with open(join(statistics_path, 'val_accuracy.json'),
                      'w',
                      encoding='utf-8') as file:
                json.dump(val_accuracy, file)

        return train_accuracy, val_accuracy

    #

    def compute(self, sentences: List[str], verbose: bool = False) -> NDArray:
        """
        run the DistilBert Embedder on a set of sentences,
        returns a list of embeddings
        """
        return self.model.encode(sentences, show_progress_bar=verbose)
