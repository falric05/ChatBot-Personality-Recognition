# See "Metrics Usage Example.ipynb" for how to save metrics using these functions.
import hashlib
import json
from enum import Enum
import os
from .BBData import character_dict

class MetricArity(int, Enum):
    SINGLE = 1  # Metric depends on a single actor
    PAIRWISE = 2  # Metric depends on two actors
    TRIPLET = 3  # Metric depends on three actors

class MetricDeterminism(int, Enum):
    DETERMINISTIC = 0  # There is a closed-form equation for this metric, which is fully computed
    PROBABILISTIC = 1  # The metric is obtained through explainable approx., e.g. SGD, partial computation on a subset...
    NEURAL = 2  # The metric is obtained via a neural network
    HUMAN = 4  # The metric is obtained via human surveying

class MetricActor(int, Enum):
    DATASET_CHARCONTEXT = 0     # Context sentences [any character but not 'Default', including "Common"]
    DATASET_CHAR = 1   # Labels or the entire dataset [any character but not 'Default', including "Common"]
    DIALOGPT_GREEDY = 10  # [any character including 'Base']
    DIALOGPT_NBEAMS = 11  # [any character including 'Base']
    DIALOGPT_SAMPLE = 12  # [any character including 'Base']

# Simple function to distinguish character vs. non-character strings
def is_character(char):
    if char in character_dict.keys() and char != 'Default':
        return True
    elif char == 'Base' or char == 'Common':
        return False
    else:
        raise Exception("Unknown character name " + char + "!")

# Function to save a dictionary as a JSON
def save_as_json(filepath, filename, data):
    if not os.path.exists(filepath):
        os.makedirs(filepath, exist_ok=True)
    with open(os.path.join(filepath, filename + ".json"), 'w') as f:
        f.write(json.dumps(data, indent=4))

# Function to load a JSON as a dictionary
def load_from_json(filepath, filename):
    if not os.path.exists(os.path.join(filepath, filename + '.json')):
        return dict()
    with open(os.path.join(filepath, filename + '.json'), 'r') as f:
        return json.load(f)

# Function to compute the hash (MD5) of a metric dictionary entry.
def dict_hash(dictionary):
    dhash = hashlib.md5()
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()

# Function returning the arity of a metric given its name
def get_metric_arity(metric_name):
    if metric_name == 'google bleu' or metric_name == 'rouge l' or \
       metric_name == 'mpnet embedding similarity' or metric_name == 'roberta crossencoding similarity' or \
       metric_name == 'meteor' or metric_name == 'neural chatbot classifier' or metric_name == 'perplexity' or \
       metric_name == 'bertscore' or metric_name == 'translation error rate' or metric_name == 'bleurt' or metric_name == 'bartscore' or \
       metric_name == "word mover distance" or metric_name == "extended edit distance":
        return MetricArity.PAIRWISE
    elif metric_name == 'distinct' or metric_name == 'emotion classifier' or metric_name == 'lines count' or \
         metric_name == 'repetitiveness' or metric_name == "t5 grammar correction edit distance" or \
         metric_name == 'distilbert-embedded chatbot classifier' or metric_name == "frequency chatbot classifier" or \
         metric_name == 'flesch-kincaid index':
        return MetricArity.SINGLE
    elif metric_name == 'comet':
        return MetricArity.TRIPLET
    elif metric_name == 'dummy metric':
        return MetricArity.PAIRWISE
    else:
        raise Exception("Unknown arity for metric " + metric_name)

# Function returning the determinism of a metric given its name and version
def get_metric_determinism(metric_name, metric_version):
    if metric_name == 'google bleu' and metric_version == 1:
        return MetricDeterminism.DETERMINISTIC
    elif metric_name == 'rouge l' and metric_version == 1:
        return MetricDeterminism.DETERMINISTIC
    elif metric_name == 'meteor' and metric_version == 1:
        return MetricDeterminism.DETERMINISTIC
    elif metric_name == 'mpnet embedding similarity' and metric_version == 1:
        return MetricDeterminism.NEURAL
    elif metric_name == 'roberta crossencoding similarity' and metric_version == 1:
        return MetricDeterminism.NEURAL
    elif metric_name == 'perplexity' and metric_version == 1:
        return MetricDeterminism.DETERMINISTIC
    elif metric_name == 'neural chatbot classifier' and metric_version == 1:
        return MetricDeterminism.PROBABILISTIC
    elif metric_name == 'distinct' and metric_version == 1:
        return MetricDeterminism.DETERMINISTIC
    elif metric_name == 'emotion classifier' and metric_version == 1:
        return MetricDeterminism.NEURAL
    elif metric_name == 'repetitiveness' and metric_version == 1:
        return MetricDeterminism.DETERMINISTIC
    elif metric_name == 'translation error rate' and metric_version == 1:
        return MetricDeterminism.DETERMINISTIC
    elif metric_name == 'bertscore' and metric_version == 1:
        return MetricDeterminism.NEURAL
    elif metric_name == 'comet' and metric_version == 1:
        return MetricDeterminism.NEURAL
    elif metric_name == 'bleurt' and metric_version == 1:
        return MetricDeterminism.NEURAL
    elif metric_name == 'bartscore' and metric_version == 1:
        return MetricDeterminism.NEURAL
    elif metric_name == 'word mover distance' and metric_version == 1:
        return MetricDeterminism.PROBABILISTIC
    elif metric_name == 'extended edit distance' and metric_version == 1:
        return MetricDeterminism.DETERMINISTIC
    elif metric_name == 't5 grammar correction edit distance' and metric_version == 1:
        return MetricDeterminism.NEURAL
    elif metric_name == 'distilbert-embedded chatbot classifier' and metric_version == 1:
        return MetricDeterminism.PROBABILISTIC
    elif metric_name == 'frequency chatbot classifier' and metric_version == 1:
        return MetricDeterminism.DETERMINISTIC
    elif metric_name == 'flesch-kincaid index' and metric_version == 1:
        return MetricDeterminism.DETERMINISTIC
    elif metric_name == 'dummy metric':
        return MetricDeterminism.DETERMINISTIC
    else:
        raise Exception("Unknown determinism for metric " + metric_name)

# Function to save a metric dictionary as a JSON
def save_metric_by_name(path, filename, metric_dict):
    if os.path.exists(os.path.join(path, filename)):
        metrics = load_from_json(path, filename)
    else:
        metrics = dict()
    metrics.update(metric_dict)
    save_as_json(path, filename, metrics)

# Function to load metrics from a JSON into a dictionary
def load_metric_by_name(path, filename):
    metrics = load_from_json(path, filename)
    # Substitute integer entries for their corresponding enums
    for entry in metrics.values():
        for actor in entry['metric_actors'].values():
            actor[0] = MetricActor(actor[0])
        entry['metric_determinism'] = MetricDeterminism(entry['metric_determinism'])
        entry['metric_arity'] = MetricArity(entry['metric_arity'])
    return metrics
