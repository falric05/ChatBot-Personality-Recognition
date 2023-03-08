import pandas as pd
from tqdm import tqdm
import os

from lib.BBData import character_dict, random_state

from .frequency import sentence_preprocess, FrequencyClassifier

# Class defining the frequency chatbot classifier
class FrequencyChatbotClassifier:
    def __init__(self) -> None:
        self.train_size = 0.85
        self.test_size = 0.10
        self.defalt_size = 0.02
        self.character = None
        self.characters = list(character_dict.keys())
        self.classifier_model = None
        self.character_docs = None

    def reset_state(self) -> None:
        """Reset state of the model"""
        self.sentence_transformer = None
        self.character = None
        self.classifier_model = None

    def train(self,
              source_path,
              mode) -> None:
        # Flush the instance state cache
        self.reset_state()

        # Read the tv/series dataset of the character
        self.character_docs = dict()
        class_docs = [[], []]
        print("Loading data")
        for c in tqdm(self.characters):
            if c == 'Default':
                # The default dataset size (Small version) is sampled with a fraction value 0.02
                df = pd.read_csv(os.path.join(source_path, c, f'{c}.tsv'), 
                                names=[c, 'response'], sep='\t')
                df = df.sample(frac=self.defalt_size, random_state=random_state)
                # Removing 3 first chars for each line
                df['response'] = df['response'].apply(lambda x: x[3:])
                df[c] = df[c].apply(lambda x: x[3:])
            else:
                df = pd.read_csv(os.path.join(source_path, c, f'{c}.csv'))
            tmp_list = df['response'].tolist()
            self.character_docs[c] = tmp_list
            class_docs[0] += tmp_list
            class_docs[1] += [c for _ in tmp_list]
        print("Preprocessing data")
        # Preprocess and filter some data
        for c in tqdm(self.characters):
            for i in range(len(self.character_docs[c])):
                sentence, relevant = sentence_preprocess(self.character_docs[c][i])
                if relevant:
                    self.character_docs[c][i] = sentence
        print("Training model")
        self.classifier_model = FrequencyClassifier(self.characters, mode=mode)
        self.classifier_model.train(list(self.character_docs.values()))
        print("Training done!")        

    def compute(self,
                sentences):
            return self.classifier_model.predict(sentences)
