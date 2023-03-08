from enum import Enum

class EnumBase(Enum):
    @classmethod
    def tolist(cls):
        """Returns the list of all possible values that can be used as metric"""
        return list(map(lambda c: c.value, cls))
    
    @classmethod
    def contains(cls, value):
        """A classmethod for looking up values not found in cls."""
        value = value.lower()
        for member in cls:
            if member.value == value:
                return member
        return None
    
    def __eq__(self, cls1):
        """A classmethod for looking up members in cls."""
        if type(cls1) == str:
            value = cls1
            value = value.lower()
            return not self.contains(value) is None
        else:
            return self.__eq__(cls1)
    
character_dict = {
    'Barney': {
        'df_filename':'Barney.csv',
        'prediction_filename':'barney_prediction',
        'checkpoint_folder':'barney_model',
        'classifier_folder':'barney_classifier',
        'source':'HIMYM',
        'delete_names': [
            "Barney's Secretary", 'Marshall to Barney', "Barney's mom",
            'Ted, from seeing Barney', 'Lily, holding Barney',
            'Marshall, on the phone with Barney',
            "At Casa a pezzi. Barney is playing the piano.Ted's father",
            'Marshall, to the girl Barney is talking to'
        ]
    },
    'Sheldon': {
        'df_filename': 'Sheldon.csv',
        'prediction_filename': 'sheldon_prediction',
        'checkpoint_folder': 'sheldon_model',
        'classifier_folder':'sheldon_classifier',
        'source': 'TBBT',
        'delete_names': []
    },
    'Harry': {
        'df_filename': 'Harry.csv',
        'prediction_filename': 'harry_prediction',
        'checkpoint_folder': 'harry_model',
        'classifier_folder':'harry_classifier',
        'source': 'HP',
        'delete_names': []
    },
    'Fry': {
        'df_filename': 'Fry.csv',
        'prediction_filename': 'fry_prediction',
        'checkpoint_folder': 'fry_model',
        'classifier_folder': 'fry_classifier',
        'source': 'Futurama',
        'delete_names': ['Mrs fry', 'Mr fry', 'Luck of the fryrish']
    },
    'Bender': {
        'df_filename': 'Bender.csv',
        'prediction_filename': 'bender_prediction',
        'checkpoint_folder': 'bender_model',
        'classifier_folder': 'bender_classifier',
        'source': 'Futurama',
        'delete_names': []
    },
    'Vader': {
        'df_filename': 'Vader.csv',
        'prediction_filename': 'vader_prediction',
        'checkpoint_folder': 'vader_model',
        'classifier_folder': 'vader_classifier',
        'source': 'SW',
        'delete_names': ["INT. DARTH VADER'S WINGMAN - COCKPIT"]
    },
    'Joey': {
        'df_filename':'Joey.csv',
        'prediction_filename':'joey_prediction',
        'checkpoint_folder':'joey_model',
        'classifier_folder':'joey_classifier',
        'source':'Friends',
        'delete_names': [
            "Joeys Sisters", 'Joey\'s Date', "Joey's Look-A-Like",
            'Joeys Sister', "Joey's Doctor", "Joey's Hand Twin", 'Joeys Date',
            'Joeys Grandmother'
        ]
    },
    'Phoebe': {
        'df_filename': 'Phoebe.csv',
        'prediction_filename': 'phoebe_prediction',
        'checkpoint_folder': 'phoebe_model',
        'classifier_folder': 'phoebe_classifier',
        'source': 'Friends',
        'delete_names': ['Amy turns around to Phoebe', 'Phoebe Waitress']
    },
    'Default': None
}

source_dict = {
    'HIMYM': {
        'dataset_folder': 'Episodes',
        'df_filename': 'HIMYM.csv'
    },
    'Futurama': {
        'dataset_folder': 'Episodes',
        'df_filename': 'Futurama.csv'
    },
    'Friends': {
        'dataset_folder': None,
        'df_filename': 'Friends.csv'
    },
    'HP': {
        'dataset_folder': None,
        'df_filename': 'HP.csv'
    },
    'SW': {
        'dataset_folder': 'Scripts',
        'df_filename': 'SW.csv'
    },
    'TBBT': {
        'dataset_folder': 'Episodes',
        'df_filename': 'TBBT.csv'
    },
}

random_state = 31239812

model_name = 'microsoft/DialoGPT-small'
n_beams = 3
top_k = 50
top_p = 0.92
context_n = 5