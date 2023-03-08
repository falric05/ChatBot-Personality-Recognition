import tensorflow as tf
import os
import pandas as pd
import numpy as np

# Questions asked to all chatbots as part of the "human - consistency" metric
consistency_questions = [
    "Who are you?", "What is your name?", "What is your job?",
    "Where do you live?"
]

# Function to run a conversation between a chatbot and a human, used to train the "human - coherence" metric.
# In "compute" mode, returns the average of the scores given by humans on these types of conversation, instead
def conversation(model, tokenizer, filepath, train, length=5):
    # If we are training the metric, go ahead with the chat
    if train:
        # Initialize an empty chat
        chat_history = []
        chat_history_ids = []
        # Chat for 'length' times
        for step in range(length):
            # Get prompt from user
            user_sentence = input(">> User:")
            # Add the user sentence to the chat history
            chat_history.append(user_sentence)
            # Encode the new user input, add the eos_token and return a tensor
            new_user_input_ids = tokenizer.encode(user_sentence +
                                                  tokenizer.eos_token,
                                                  return_tensors='tf')
            # Append the new user input tokens to the chat history
            bot_input_ids = tf.concat([chat_history_ids, new_user_input_ids],
                                      -1) if step > 0 else new_user_input_ids
            # Generate a response while limiting the current answer to 128 tokens, using sampling
            max_length = 128 + bot_input_ids.shape[1]
            chat_history_ids = model.generate(
                bot_input_ids,
                max_length=max_length,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                top_p=0.92,
                top_k=50)
            # Get the last ouput tokens from the bot
            bot_sentence = tokenizer.decode(
                chat_history_ids[:, bot_input_ids.shape[-1]:][0],
                skip_special_tokens=True)
            # Append the bot answer to the chat history
            chat_history.append(bot_sentence)
            # Pretty print the bot answer
            print("DialoGPT: {}".format(bot_sentence))
        # Once the chat is over, we ask the user for a score between 0 and 5 (integers only).
        # Initialize variables to get the score
        got_score = False
        score = None
        # While used for input sanity checks on the user input...
        while not got_score:
            # Get the score as a user input
            score = input("How do you rate this conversation (0 to 5)? ")
            # If the score is a valid character, cast it to an integer and proceed
            if score == "0" or score == "1" or score == "2" or score == "3" or score == "4" or score == "5":
                score = int(score)
                got_score = True
            # Otherwise, inform the user that its input is not valid, and ask again
            else:
                print(
                    "Invalid score! Must be a single integer between 0 and 5!")
        # Make directories if they do not exists, where to store the csv containing the chats and the user scores
        if not os.path.exists(filepath.rsplit(os.path.sep, 1)[0]):
            os.makedirs(filepath.rsplit(os.path.sep, 1)[0], exist_ok=True)
        # If the csv containing chats and user scores exists, first load it, then append a new entry
        if os.path.exists(filepath):
            human_convo_df = pd.read_csv(filepath)
            human_convo_df = human_convo_df.append(
                {
                    "chat": chat_history,
                    "score": score
                }, ignore_index=True)
        # Otherwise, create it from scratch
        else:
            human_convo_df = pd.DataFrame.from_dict({
                "chat": [chat_history],
                "score": score
            })
        # Save the csv file
        human_convo_df.to_csv(filepath, index=False)
    else:
        # If we only want to compute the metric, read the csv file containing chat-user_scores pairs, and return
        # mean and std of the scores
        human_convo_df = pd.read_csv(filepath)
        return np.average(human_convo_df['score'].to_numpy() / 5), np.std(
            human_convo_df['score'].to_numpy() / 5)


# Function to run single rounds of question-answer between a chatbot and a pre-set of questions, used to train the style and consistency
# human metrics. In "compute" mode, returns the average of the scores given by humans on these types of queries, instead
def single_answers(model, tokenizer, filepath, train, questions):
    # If we are training the metric, go ahead with the queries
    if train:
        # Initialize an empty query
        questions_history = []
        # Ask each question separately
        for question in questions:
            print("Question: {}".format(question))
            # Encode the question, adding the eos_token at its end
            question_input_ids = tokenizer.encode(question +
                                                  tokenizer.eos_token,
                                                  return_tensors='tf')
            # Generate a response while limiting the current answer to 128 tokens, using sampling
            max_length = 128 + question_input_ids.shape[1]
            chat_history_ids = model.generate(
                question_input_ids,
                max_length=max_length,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                top_p=0.92,
                top_k=50)
            # Pretty print the bot answer
            bot_sentence = tokenizer.decode(
                chat_history_ids[:, question_input_ids.shape[-1]:][0],
                skip_special_tokens=True)
            # Append the question-answer pair to the history
            questions_history.append((question, bot_sentence))
            # Pretty print the bot answer
            print("DialoGPT: {}".format(bot_sentence))
        # Once the queries are over, we ask the user for a score between 0 and 5 (integers only).
        # Initialize variables to get the score
        got_score = False
        score = None
        # While used for input sanity checks on the user input...
        while not got_score:
            # Get the score as a user input
            score = input("How do you rate these answers (0 to 5)? ")
            # If the score is a valid character, cast it to an integer and proceed
            if score == "0" or score == "1" or score == "2" or score == "3" or score == "4" or score == "5":
                score = int(score)
                got_score = True
            # Otherwise, inform the user that its input is not valid, and ask again
            else:
                print(
                    "Invalid score! Must be a single integer between 0 and 5!")
        # Make directories if they do not exists, where to store the csv containing the questions and the user scores
        if not os.path.exists(filepath.rsplit(os.path.sep, 1)[0]):
            os.makedirs(filepath.rsplit(os.path.sep, 1)[0], exist_ok=True)
        # Otherwise, create it from scratch
        if os.path.exists(filepath):
            questions_df = pd.read_csv(filepath)
            questions_df = questions_df.append(
                {
                    "questions": questions_history,
                    "score": score
                },
                ignore_index=True)
        else:
            questions_df = pd.DataFrame.from_dict({
                "questions": [questions_history],
                "score":
                score
            })
        # Save the csv file
        questions_df.to_csv(filepath, index=False)
    else:
        # If we only want to compute the metric, read the csv file containing questions-user_scores pairs, and return
        # mean and std of the scores
        questions_df = pd.read_csv(filepath)
        return np.average(questions_df['score'].to_numpy() / 5), np.std(
            questions_df['score'].to_numpy() / 5)