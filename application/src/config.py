"""
Configuration file to keep track of all our project-wide variables
"""


import torch

from application.src.utils import get_root_path

DEVICE = torch.device('cpu')
# DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

BATCH_SIZE = 2
NUM_EPOCHS = 2
RESIZE_TO = 400

OUT_DIR = str(get_root_path()) + '/data/outputs'
TRAIN_DIR = str(get_root_path()) + '/data/datasets/LU-data/train'
VALID_DIR = str(get_root_path()) + '/data/datasets/LU-data/valid'
WIKI_DIR = str(get_root_path()) + '/data/datasets/wikipedia-data/wikipedia'

TEST_PATH = str(get_root_path()) + '/data/datasets/play_data'

MODEL_NAME = 'weightsCNN'
SAVE_PLOTS_EPOCH = 2
SAVE_MODEL_EPOCH = 2

LOWER_ACCURACY_LIMIT = 60
VISUALIZE_TRANSFORMED_IMAGES = False


# Data classes for datasets
CLASSES = [
    'BACKGROUND', 'PEDESTRIAN_CROSSING', 'PRIORITY_ROAD', 'PASS_RIGHT_SIDE', '50_SIGN', '70_SIGN', '80_SIGN',
    'NO_PARKING', 'GIVE_WAY', '100_SIGN', 'NO_STOPPING_NO_STANDING', 'STOP', 'WARNING_BAD_ROAD', 'WARNING_DANGER',
    'WARNING_RAIL_CROSSING', 'WARNING_CROSSING_ROAD', 'WARNING_PEDESTRIANS', 'WARNING', '30_SIGN', 'WARNING_CHILDREN',
    'WARNING_SHARP_CURVES', 'WARNING_EQUISTARIANS', 'PASS_EITHER_SIDE'
]

NUM_CLASSES = len(CLASSES)

WIKIPEDIA_LABELS = {
    'A': 'Warning Sign',
    'B': 'Priority Sign',
    'C': 'Prohibitory Sign',
    'D': 'Mandatory Sign',
    'E': 'Instruction Sign',
    'F': 'Location Sign\n(Directions)',
    'G': 'Location Sign\n(Information on Public Institutions etc.)',
    'H': 'Location Sign\n(Information on Service Facilities etc.)',
    'I': 'Location Sign\n(Information on Interesting Destinations etc.)',
    'J': 'Information Sign',
    'P': 'Signals by Policemen',
    'S': 'Symbols',
    'T': 'Additional Board',
    'X': 'Other',
    'Z': 'Uncategorized',
}
