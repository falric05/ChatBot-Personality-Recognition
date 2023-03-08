from os.path import join, exists
from os import listdir
from tqdm import tqdm
import json
import pandas as pd
import re
from datasets import DatasetDict
from datasets import load_dataset
from transformers import AutoTokenizer

from .BBData import source_dict, character_dict, random_state, model_name, top_p, top_k, n_beams

# Open the dataset documents and store their data into a DataFrame
def open_char_dataset(character, base_folder):
    ### Loading functions from other files
    # Load the dataset from How I Met Your Mother or
    #                       The Big Bang Theory   or
    #                       Friends
    def _load_himym_friends_tbbt_dataset(sources_folder):
        dataframe_rows = []
        # Get number of documents and their names
        documents_n = len(listdir(sources_folder))
        documents_names = listdir(sources_folder)
        # Loop over documents
        for i in tqdm(range(documents_n)):
            # Extract filename which correspond to the link of the episode
            filename = documents_names[i]
            # the last 5 chars takes the form `sxe` with s the number of the current serie and
            # and e as the number of the episode
            sources_label = filename[:-4]
            # Open document
            with open(join(sources_folder, filename), encoding="utf8") as file:
                # Loop over lines (= words)
                for line in file.readlines():
                    dataframe_row = {
                        "source": sources_label,
                        "line": line,
                    }
                    dataframe_rows.append(dataframe_row)
        # Build the dataframe from the words
        df = pd.DataFrame(dataframe_rows)
        return df

    # Load the dataset from Futurama
    def _load_futurama_dataset(sources_folder):
        futurama_txt = ''
        # Loop over documents
        for filename in tqdm(listdir(sources_folder)):
            futurama_txt += open(join(sources_folder, filename),
                                 encoding='utf-8').read()
        # Split lines
        start_idx = 0
        end_idx = 0
        lines = []
        while start_idx < len(futurama_txt):
            # eventually bold tag are present, discard them
            start_idx = futurama_txt.find('<b>', end_idx)
            if start_idx == -1:  # if no '<b>' is found, just save the rest
                lines.append(futurama_txt[end_idx:].replace('</b>', ''))
                break
            elif start_idx != end_idx:  # '<b>' is found
                lines.append(futurama_txt[end_idx + 4:start_idx])
            end_idx = futurama_txt.find('</b>', start_idx)
            if end_idx == -1:  # if no '</b>' is found, just save the rest
                lines.append(futurama_txt[start_idx:].replace('<b>', ''))
                break
            lines.append(futurama_txt[start_idx + 3:end_idx])
        df = pd.DataFrame(lines, columns=['line'])
        return df

    # Load the dataset from Harry Potter
    def _load_hp_dataset(sources_folder):
        sep = ';'
        df = None
        df_files = []
        # for each movie append the dataset which refers to it
        for filename in listdir(sources_folder):
            df_files.append(
                pd.read_csv(join(sources_folder, filename),
                            sep=sep).rename(columns=lambda x: x.lower()))
        df = pd.concat(df_files)
        df = df.rename(columns={'character': 'character', 'sentence': 'line'})
        return df

    # Load the dataset from Star Wars
    def _load_sw_dataset(source_folder):
        dataframe_rows = []
        # Get number of documents and their names
        documents_n = len(listdir(source_folder))
        documents_names = listdir(source_folder)
        # Loop over documents
        for i in tqdm(range(documents_n)):
            filename = documents_names[i]
            film_name = filename[:-4]
            # Open document
            with open(join(source_folder, filename), encoding='utf-8') as file:
                film_rows = []
                sentence = ""
                empty_line_allow = False
                between_numbers = False
                found_character = False
                for line in file.readlines():
                    if re.search(
                            r"^[0-9]+.", line
                    ) != None:  # Line is number followed by dot (page number)
                        pass
                    elif re.search(
                            r"^[A-Z]{2,}", line
                    ) != None:  # Line begins with an-all caps (a character)
                        sentence += line
                        found_character = True
                        empty_line_allow = True
                    elif line.isspace():
                        if empty_line_allow:
                            pass
                        else:
                            if found_character:
                                film_row = {
                                    "film": film_name,
                                    "line": sentence,
                                }
                                film_rows.append(film_row)
                                sentence = ""
                                found_character = False
                    elif found_character:
                        sentence += line
                        empty_line_allow = False
                dataframe_rows.extend(film_rows)
        # Build the dataframe from the words
        df = pd.DataFrame(dataframe_rows)
        return df

    ### Function starts here
    # if character selected is 'Default' so we don't need any dataset
    if character == 'Default':
        # no dataset is loaded
        return None
    # otherwise let's take from the source dictionary the folder which contains the datasets
    # sources_subfolder is a parameter which contains the path where all data are stored, it can
    #   be different from null if data are stored in a different subfolder
    source = character_dict[character]['source']
    sources_subfolder = source_dict[source]['dataset_folder']
    if sources_subfolder:
        sources_folder = join(base_folder, "Data", "Sources", source,
                              sources_subfolder)
    else:
        sources_folder = join(base_folder, "Data", "Sources", source)
    # each tv shows loads data by a call to its respective function
    if source == 'HIMYM' or source == 'Friends' or source == 'TBBT':
        df = _load_himym_friends_tbbt_dataset(sources_folder)
    elif source == 'Futurama':
        df = _load_futurama_dataset(sources_folder)
    elif source == 'HP':
        df = _load_hp_dataset(sources_folder)
    elif source == 'SW':
        df = _load_sw_dataset(sources_folder)
    return df

def process_char_dataset(df, character):
    def _process_himym_dataset(df):
        # Removes lines which starts with brackets
        df = df[~df['line'].str.startswith("[")]
        df = df[~df['line'].str.startswith("(")]
        # Removes white space
        df['line'] = df['line'].str.strip()
        # Removes everything is inside the round brackets
        df['line'] = df['line'].str.replace(r"\(.*\)", "")
        # Removes bracket char, newline, tabular char and special chars replacing them with a space
        df['line'] = df['line'].str.replace(r"[\/(){}\[\]\|@_#]|\\t|\\n", " ")
        # Removes every char which is not present in the following "white list"
        df['line'] = df['line'].str.replace(r"[^.\',;:?!0-9a-zA-Z \-]", "")
        df = df[~df['line'].isnull()]
        df[['character', 'line']] = df['line'].str.split(":", 1, expand=True)
        # Removes empty lines
        df = df.dropna()
        # Removes white space
        df['line'] = df['line'].str.strip()
        df['line'] = df['line'][df['line'].str.len() >= 2]
        # Removes empty lines
        df = df[~df['line'].isnull()]
        df = df.replace(r'^s*$', float('NaN'), regex=True)
        df = df.dropna()
        df = df.reset_index(drop=True)
        return df

    def _process_tbbt_dataset(df):
        # Removes lines which starts with brackets
        df = df[~df['line'].str.startswith("[")]
        df = df[~df['line'].str.startswith("(")]
        df = df[~df['line'].str.startswith("Scene: ")]
        # Removes white space
        df['line'] = df['line'].str.strip()
        # Removes everything is inside the round brackets
        df['line'] = df['line'].str.replace(r"\(.*\)", "")
        # Removes bracket char, newline, tabular char and special chars replacing them with a space
        df['line'] = df['line'].str.replace(r"[\/(){}\[\]\|@_#]|\\t|\\n", " ")
        # Removes every char which is not present in the following "white list"
        df['line'] = df['line'].str.replace(r"[^.\',;:?!0-9a-zA-Z \-]", "")
        # Removes empty lines
        df = df[~df['line'].isnull()]
        df[['character', 'line']] = df['line'].str.split(":", 1, expand=True)
        df = df.dropna()
        # Removes white space
        df['line'] = df['line'].str.strip()
        df['line'] = df['line'][df['line'].str.len() >= 2]
        # Removes empty lines
        df = df[~df['line'].isnull()]
        df = df.replace(r'^s*$', float('NaN'), regex=True)
        # Removes empty lines
        df = df.dropna()
        df = df.reset_index(drop=True)
        return df

    def _process_futurama_dataset(df):
        # Remove white space
        df['line'] = df['line'].str.strip()
        # Removes everything is inside the round and square brackets
        df['line'] = df['line'].str.replace(r"\[.*\]", "")
        df['line'] = df['line'].str.replace(r"\(.*\)", "")
        # Removes everything is inside the tags
        df['line'] = df['line'].str.replace(r"\<.*\>", "")
        df['line'] = df['line'].str.replace(r"\s+", " ")
        df['line'] = df['line'].str.replace("\n", "")
        # Removes lines which starts with brackets
        df = df[~df['line'].str.startswith("(")]
        df = df[~df['line'].str.startswith("[")]
        # Removes bracket char, newline, tabular char and special chars replacing them with a space
        df['line'] = df['line'].str.replace(r"[\/(){}\[\]\|@_#]|\\t|\\n", " ")
        # Removes every char which is not present in the following "white list"
        df['line'] = df['line'].str.replace(r"[^.\',;:?!0-9a-zA-Z \-]", "")
        df['line'] = df['line'][df['line'].str.len() >= 2]
        # Removes empty lines
        df = df.dropna()
        df = df.reset_index(drop=True)
        df_rows = []
        for row in tqdm(range(len(df) - 1)):
            if df['line'][row].isupper():
                df_row = {
                    'line': df['line'][row + 1].strip()[:512],
                    'character': df['line'][row].strip().capitalize()
                }
                df_rows.append(df_row)
        df = pd.DataFrame(df_rows)
        # Discard titles
        df = df[df['character'].str.contains('Futurama') == False]
        df = df.replace(r'^s*$', float('NaN'), regex=True)
        # Removes empty lines
        df = df.dropna()
        df = df.reset_index(drop=True)
        return df

    def _process_friends_dataset(df):
        # Removes lines which starts with brackets
        df = df[~df['line'].str.startswith("[")]
        df = df[~df['line'].str.startswith("(")]
        # Removes white space
        df['line'] = df['line'].str.strip()
        # Removes everything is inside the round brackets
        df['line'] = df['line'].str.replace(r"\(.*\)", "")
        # Removes bracket char, newline, tabular char and special chars replacing them with a space
        df['line'] = df['line'].str.replace(r"[\/(){}\[\]\|@_#]|\\t|\\n", " ")
        # Removes every char which is not present in the following "white list"
        df['line'] = df['line'].str.replace(r"[^.\',;:?!0-9a-zA-Z \-]", "")
        df = df[~df['line'].isnull()]
        df[['character', 'line']] = df['line'].str.split(":", 1, expand=True)
        # Removes empty lines
        df = df.dropna()
        # Removes white space
        df['line'] = df['line'].str.strip()
        df['line'] = df['line'][df['line'].str.len() >= 2]
        # Removes empty lines
        df = df[~df['line'].isnull()]
        df = df[~(df['character'] == 'Written by')]
        df = df.replace(r'^s*$', float('NaN'), regex=True)
        # Removes empty lines
        df = df.dropna()
        df = df.reset_index(drop=True)
        return df

    def _process_sw_dataset(df):
        # Removes lines which starts with brackets
        df = df[~df['line'].str.startswith("[")]
        df = df[~df['line'].str.startswith("(")]
        # Removes white space
        df['line'] = df['line'].str.strip()
        # Removes everything is inside the round brackets
        df['line'] = df['line'].str.replace(r"\(.*\)", "")
        df[['character', 'line']] = df['line'].str.split("\n", 1, expand=True)
        # Removes bracket char, newline, tabular char and special chars replacing them with a space
        df['line'] = df['line'].str.replace(r"[\/(){}\[\]\|@_#]|\\t|\\n", " ")
        # Removes every char which is not present in the following "white list"
        df['line'] = df['line'].str.replace(r"[^.\',;:?!0-9a-zA-Z \-]", "")
        # Removes empty lines
        df = df[~df['line'].isnull()]
        df = df[df['character'].str.split().apply(lambda l: len(l)) <= 6]
        df = df.replace(r'^s*$', float('NaN'), regex=True)
        # Removes empty lines
        df = df.dropna()
        df = df.reset_index(drop=True)
        return df

    def _process_hp_dataset(df):
        # Removes white space
        df['line'] = df['line'].str.strip()
        # Removes everything is inside the round brackets
        df['line'] = df['line'].str.replace(r"\(.*\)", "")
        # Removes bracket char, newline, tabular char and special chars replacing them with a space
        df['line'] = df['line'].str.replace(r"[\/(){}\[\]\|@_#]|\\t|\\n", " ")
        # Removes every char which is not present in the following "white list"
        df['line'] = df['line'].str.replace(r"[^.\',;:?!0-9a-zA-Z \-]", "")
        # Remove empty lines
        df = df[~df['line'].isnull()]
        df = df.dropna()
        # Removes white space
        df['line'] = df['line'].str.strip()
        df['character'] = [line.lower() for line in df['character']]
        # Removes empty lines
        df = df[~df['line'].isnull()]
        df = df.replace(r'^s*$', float('NaN'), regex=True)
        # Removes empty lines
        df = df.dropna()
        df = df.reset_index(drop=True)
        return df

    # Function starts here
    if character == 'Default':
        return None
    source = character_dict[character]['source']
    if source == 'HIMYM':
        df = _process_himym_dataset(df)
    elif source == 'Friends':
        df = _process_friends_dataset(df)
    elif source == 'Futurama':
        df = _process_futurama_dataset(df)
    elif source == 'TBBT':
        df = _process_tbbt_dataset(df)
    elif source == 'HP':
        df = _process_hp_dataset(df)
    elif source == 'SW':
        df = _process_sw_dataset(df)
    return df

# Function used to load the dataframe of the character selected
def load_char_df(character, base_folder):
    """
    docstring
    """
    # Takes the folder of the character
    dataset_path = join(base_folder, "Data", "Characters", character,
                        character + '.csv')

    # Load HuggingFace dataset
    character_hg = load_dataset('csv',
                                data_files=dataset_path,
                                cache_dir=join("cache"))

    # Perform 85% train / 10% test / 5% validation with a fixed seed
    train_test_hg = character_hg['train'].train_test_split(test_size=0.15,
                                                           seed=random_state)
    test_val = train_test_hg['test'].train_test_split(test_size=0.33,
                                                      seed=random_state)

    # Store splits into a HuggingFace dataset
    character_hg = DatasetDict({
        'train': train_test_hg['train'],
        'test': test_val['train'],
        'val': test_val['test']
    })
    return character_hg

# Function defining the formatting of the dataset rows, so that they can be fed to dialogpt
# (DialoGPT requires a conversation to be fed as a single string of concatenated exchanges)
def dialogpt_preprocess_function(examples, tokenizer):
    # Function to construct a conversation from the rows of a dataset
    def _construct_conv(row, tokenizer):
        """
        docstring
        """
        # Max conversation length
        MAX_LENGTH = 512
        # Reverse the rows, since they are originally in order response->context/0->context/1...
        row = list(reversed(list(row.values())))
        # Pass row into the tokenizer, getting a dictionary with input_ids and attention_mask
        model_inputs = tokenizer(row)
        # Get pad token encoding
        tokenizer_pad_token_id = tokenizer.encode('#')[0]
        # Append to each row element the eos token, to separate the sentences
        for i in range(len(model_inputs['input_ids'])):
            model_inputs['input_ids'][i].append(tokenizer.eos_token_id)
            model_inputs['attention_mask'][i].append(1)
        # Transform the lists into a single concatenated conversation
        model_inputs['input_ids'] = [
            item for sublist in model_inputs['input_ids'] for item in sublist
        ]
        model_inputs['attention_mask'] = [
            item for sublist in model_inputs['attention_mask'] for item in sublist
        ]
        # If there is extra space, append padding tokens with attention mask 0
        if MAX_LENGTH > len(model_inputs['input_ids']):
            model_inputs['input_ids'] += [tokenizer_pad_token_id] * (
                MAX_LENGTH - len(model_inputs['input_ids']))
            model_inputs['attention_mask'] += [0] * (
                MAX_LENGTH - len(model_inputs['attention_mask']))
        # If on the other hand the conversation is too long, truncate it, always setting eos as the last token
        elif MAX_LENGTH < len(model_inputs['input_ids']):
            model_inputs['input_ids'] = model_inputs['input_ids'][:MAX_LENGTH - 1]
            model_inputs['input_ids'][-1] = tokenizer.eos_token_id
            model_inputs['attention_mask'] = model_inputs[
                'attention_mask'][:MAX_LENGTH - 1]
            model_inputs['attention_mask'][-1] = 1
        # Since dialogpt is an autoregressive model, labels should be equal to inputs during training
        model_inputs["labels"] = model_inputs["input_ids"]
        # Return the dictionary
        return model_inputs

    # Run the function to construct conversations defined above
    model_inputs = _construct_conv(examples, tokenizer)
    return model_inputs

# Function that extract predictions from a stored file to speed up the process
def get_chatbot_predictions(sample_questions,
                           model,
                           filename,
                           generation_method,
                           character,
                           tokenizer,
                           base_folder,
                           file_caching=True,
                           override_predictions=False):

    prediction_path = join(base_folder, 'Data', 'Characters', character,
                           filename)
    # If the predictions have already been created
    if exists(prediction_path) and not override_predictions and file_caching:
        # It loads them
        print("Loading predictions from stored file")
        with open(prediction_path, 'r', encoding='utf-8') as file:
            json_string = file.read()
        predictions = json.loads(json_string)
        print("Loaded predictions from stored file")
    else:
        # Otherwise they are created
        print("Creating predictions")
        predictions = list()
        for x in tqdm(sample_questions):
            tokenized_question = tokenizer.encode(x + tokenizer.eos_token,
                                                  return_tensors='tf')
            # Max length of each tokenized sequence must be the following
            max_length = 128 + tokenized_question.shape[1]
            if generation_method == "greedy":  # Greedy generation method
                generated_answer = model.generate(
                    tokenized_question,
                    pad_token_id=tokenizer.eos_token_id,
                    max_length=max_length)[0].numpy().tolist()
            elif generation_method == "nbeams":  # Beam Search generation method
                generated_answer = model.generate(
                    tokenized_question,
                    pad_token_id=tokenizer.eos_token_id,
                    max_length=max_length,
                    n_beams=n_beams)[0].numpy().tolist()
            elif generation_method == "sampling":  # Sampling generation method
                b = True
                c = 0
                while b:
                    generated_answer = model.generate(
                        tokenized_question,
                        pad_token_id=tokenizer.eos_token_id,
                        max_length=max_length,
                        do_sample=True,
                        top_k=top_k,
                        top_p=top_p)[0].numpy().tolist()
                    c += 1
                    if len(generated_answer[len(tokenized_question[0]):]) > 1:
                        b = False
                    if c > 100:
                        generated_answer[len(tokenized_question[0]
                                             ):] = tokenizer.encode('hi') + [
                                                 tokenizer.eos_token_id
                                             ]
                        break
            # Append predictions
            predictions.append(generated_answer[len(tokenized_question[0]):])
        if file_caching:
            # Save predictions as a JSON file
            output_string = json.dumps(predictions)
            with open(prediction_path, 'w', encoding='utf-8') as file:
                file.write(output_string)

        assert all([len(p) > 1 for p in predictions])

    return predictions