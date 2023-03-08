from tensorflow import keras
from keras import layers, callbacks, regularizers
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from sentence_transformers import SentenceTransformer
import random
import itertools
from sklearn.model_selection import train_test_split
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Class defining the semantic classifier
class BarneyBotTripletClassifier:
    # Initialization
    def __init__(self):
        # Training params for the classifier
        self.batch_size = 16
        self.lr = 1e-6
        self.patience = 6
        self.regularizer_weight_r = 1e-4
        self.regularizer_weight_s = 1e-3
        self.dropout_rate = 0.2
        self.train_size = 0.85
        self.test_size = 0.10
        # Instance state, for caching, in case of repeated usage of this metric
        self.sentence_transformer = None
        self.character = None
        self.classifier_model = None

    # Function to flush instance state cache
    def reset_state(self):
        self.sentence_transformer = None
        self.character = None
        self.classifier_model = None

    # Function to create the keras model underneath the classifier
    def create_model(self, input_size):
        # Input is a concatenated triplet of sentences
        inputs = keras.Input(shape=input_size)
        # Model is a concatenation of dense layers alternated by batch normalizations
        x = layers.Dense(
            1024,
            activation='relu',
        )(inputs)
        x = layers.BatchNormalization()(x)

        x = layers.Dense(
            1024,
            activation='relu',
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(
            512,
            activation='relu',
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(
            256,
            activation='relu',
        )(x)
        # The last layers have L2 regularization, better suited for sigmoid output
        x = layers.BatchNormalization()(x)
        x = layers.Dense(
            128,
            activation='relu',
            kernel_regularizer=regularizers.l2(self.regularizer_weight_r),
            bias_regularizer=regularizers.l2(self.regularizer_weight_r))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        # Output is a single probability value
        out = layers.Dense(
            1,
            activation='sigmoid',
            kernel_regularizer=regularizers.l2(self.regularizer_weight_s),
            bias_regularizer=regularizers.l2(self.regularizer_weight_s))(x)
        # Create and compile keras model
        classifier_model = keras.Model(inputs, out)
        classifier_model.compile(
            loss=keras.losses.BinaryCrossentropy(),
            optimizer=keras.optimizers.Adam(learning_rate=self.lr),
            metrics=[keras.metrics.BinaryAccuracy(),
                     keras.metrics.Recall()])
        return classifier_model

    # Function to create a dataset composed of triples from a dataset of single sentences. Used in training only.
    def get_triplet_df(self, series_df, n_shuffles, random_state):
        # Separate lines by character from all the others
        series_df_1 = series_df[series_df['character'] == 1].copy()
        series_df_0 = series_df[series_df['character'] == 0].copy()
        # Define triplet dataset as having a character label and the line, already encoded
        df_rows = {'character': [], 'encoded_lines': []}
        # Shuffle by a parametrized amount
        for i in range(n_shuffles):
            print("Running shuffle " + str(i) + "/" + str(n_shuffles))
            # Shuffle the dataset and balance number of 0s (we suppose its cardinality is higher than that of 1s)
            series_df_1 = series_df_1.sample(frac=1,
                                             random_state=random_state +
                                             i).reset_index(drop=True)
            series_df_0 = series_df_0.sample(n=len(series_df_1),
                                             random_state=random_state +
                                             i).reset_index(drop=True)
            # Iterate over lines
            for i in tqdm(range(2, len(series_df_1))):
                # Get a triple of consecutive lines for the character, and concatenate them in one sample
                lines = list(series_df_1['encoded_line'][i - 2:i + 1])
                lines = np.concatenate(lines)
                df_rows['character'].append(1)
                df_rows['encoded_lines'].append(lines)
                # Do the same for non-character lines
                lines = list(series_df_0['encoded_line'][i - 2:i + 1])
                lines = np.concatenate(lines)
                df_rows['character'].append(0)
                df_rows['encoded_lines'].append(lines)
        # Create a new dataframe from the rows we have built
        df = pd.DataFrame(data=df_rows)
        # Sample the dataset one last time to shuffle it
        return df.sample(frac=1,
                         random_state=random_state).reset_index(drop=True)

    # Function to run the semantic classifier for a given character in evaluation mode
    def compute(self,
                sentences,
                character,
                load_path,
                n_sentences='all',
                verbose=False):
        # If cached classifier is not the required one, re-load it
        if not self.classifier_model or character != self.character:
            self.classifier_model = keras.models.load_model(load_path)
            self.character = character
        if not self.sentence_transformer:
            self.sentence_transformer = SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        # Inform the user of successful loading
        if verbose:
            print("Using classifier at " + load_path)
        # Encode single sentences
        samples = np.array(
            [self.sentence_transformer.encode(line) for line in sentences])
        # Set a fixed random seed
        random.seed(1)
        # If n_sentences is set to 'all', we select all sentences. If it is an integer n, we instead randomly choose n sentences
        if type(n_sentences) == int:
            sampled_indices = np.random.randint(0,
                                                len(samples),
                                                size=n_sentences)
            samples = samples[sampled_indices]
        # Construct all triples from the selected sentences
        inputs = np.array([
            np.concatenate(triplet)
            for triplet in itertools.permutations(samples, 3)
        ])
        # Get semantic classifier probability for each triple, and return all of them
        outputs = self.classifier_model(inputs)
        return outputs

    # Function to train the semantic classifier on a specific character
    def train(self,
              character,
              source_path,
              source_encoded_path,
              source_save_path,
              save_path,
              random_state,
              n_shuffles=10,
              shutdown_at_end=False):
        # Flush the instance state cache
        self.reset_state()
        # The (semantic classifier) encoded lines dataset is stored to speed up re-training if needed.
        # If it does not exists, then we create it now
        if not source_encoded_path:
            print('Creating encoded lines')
            # Read the tv/series dataset of the character
            series_df = pd.read_csv(source_path)
            # Apply class labelling to the dataset sentences
            series_df['character'] = series_df['character'].apply(
                lambda x: 1 if x == character else 0)
            # Throw away unnecessary dataset rows
            series_df = series_df[['character', 'line']]
            # Load the sentence transformer to encode lines
            self.sentence_transformer = SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
            # Encode lines and add them to the dataset as a new row
            series_df['encoded_line'] = [
                self.sentence_transformer.encode(line)
                for line in tqdm(series_df['line'])
            ]
            # Save the dataset rows as a csv file
            series_df[['line', 'character']].to_csv(os.path.join(source_save_path, character.lower() + '_classifier.csv'),
                                                    index=False)
            # The encoded lines are saved separately via numpy due to their type (array)
            np.save(os.path.join(os.path.join(source_save_path, character.lower() + '_encoded_lines.npy')),
                    series_df['encoded_line'].to_numpy())
            print("Saved encoded lines at " + source_save_path)
            source_encoded_path = source_save_path
            
        # Load the preprocessed dataset
        series_df = pd.read_csv(os.path.join(source_encoded_path, character.lower() + '_classifier.csv'),
                                dtype={
                                    'line': str,
                                    'character': int
                                })
        
        # Load encoded lines dataset, via numpy, and add it as a new row in the dataset
        series_df['encoded_line'] = np.load(os.path.join(source_encoded_path, character.lower() + '_encoded_lines.npy'),
                                            allow_pickle=True)
        print(
            "Loaded encoded lines from " + source_encoded_path)
        # Perform train-val-test split on the dataset
        series_train_df, series_test_df = train_test_split(
            series_df, test_size=self.test_size, random_state=random_state)
        series_train_df, series_val_df = train_test_split(
            series_train_df,
            test_size=1 - self.train_size - self.test_size,
            random_state=random_state)
        # Get triples from the dataset
        shuffled_df = self.get_triplet_df(series_df,
                                          n_shuffles=n_shuffles,
                                          random_state=random_state)
        # Store into variables the train, val, test, total lengths of the new (triplets) dataset
        tot_len = len(shuffled_df)
        train_len = int(tot_len * self.train_size)
        test_len = int(tot_len * self.test_size)
        val_len = tot_len - train_len - test_len
        # Load triples into numpy arrays, separating data and labels
        print('Loading training data...')
        X_train = np.array(
            [[float(e) for e in s]
             for s in tqdm(shuffled_df['encoded_lines'][:train_len])])
        y_train = np.array(
            [c for c in tqdm(shuffled_df['character'][:train_len])])
        print('Loading test data...')
        X_test = np.array(
            [[float(e) for e in s]
             for s in tqdm(shuffled_df['encoded_lines'][:test_len])])
        y_test = np.array(
            [c for c in tqdm(shuffled_df['character'][:test_len])])
        print('Loading validation data...')
        X_val = np.array([[float(e) for e in s]
                          for s in tqdm(shuffled_df['encoded_lines'][:val_len])
                          ])
        y_val = np.array([c for c in tqdm(shuffled_df['character'][:val_len])])
        # Create the keras model for the semantic classifier
        self.classifier_model = self.create_model(len(X_train[0], ))
        # Define early stop behavior
        earlystop_callback = callbacks.EarlyStopping(
            monitor="val_binary_accuracy",
            min_delta=0,
            patience=self.patience,
            verbose=0,
            mode="max",
            baseline=None,
            restore_best_weights=True,
        )
        # Fit the semantic classifier
        train_history = self.classifier_model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=1000,
            verbose=1,
            callbacks=[earlystop_callback],
            batch_size=self.batch_size)
        self.character = character
        # Display a confusion matrix, to show the results of the semantic classifier
        print('#' * 25 + ' Model Test ' + '#' * 25)
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        y_pred = self.classifier_model.predict(X_test).round()
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=['Others', character])
        disp.plot(ax=ax)
        plt.show()
        # Save the semantic classifier and its training history
        classifier_path = os.path.join(save_path, character.lower() + "_classifier")
        self.classifier_model.save(classifier_path)
        filename = character.lower() + '_training_history.json'
        output_string = json.dumps(train_history.history)
        with open(os.path.join(save_path, character.lower() + "_classifier", filename),
                  'w',
                  encoding='utf-8') as file:
            file.write(output_string)
        # If a shutdown at the end is required, do so
        if shutdown_at_end:
            os.system('shutdown /' + shutdown_at_end)