import numpy as np

UINT16_MAX = (np.iinfo(np.uint16).max)
UINT8_MAX = (np.iinfo(np.uint8).max)

# Image colour correction constants
RED_CORRECTION = 2.5
GREEN_CORRECTION = 1.5
BLUE_CORRECTION = 0.7
GAMMA_CORRECTION = 1.2

# Directory paths
DATA_DIR = 'data'
OUTPUT_DIR = 'preprocessed_data'
TEST_OUTPUT_DIR = 'test_preprocessed_data'

TILE_SIZE = 512

SPLITS = ['train', 'val', 'test']
SPLIT_RATIO = [0.7, 0.2, 0.1]

# RGB Mask color dictionary
MASK_COLORS = {
    'background': [0, 0, 0],
    'green_space': [0, 0.8, 0],
    'urban': [0.8, 0, 0]
}

BINARY_UGS_VIS_COLORS = { 0: [0, 0, 0], 1: [1, 1, 0] } # Black, Yellow

# Define all class information in a single source of truth
CLASS_INFO = {
    0: {"name": "unlabeled", "rgb": [0, 0, 0]},                # Black
    1: {"name": "industrial area", "rgb": [200, 0, 0]},        # Dark red
    2: {"name": "paddy field", "rgb": [0, 200, 0]},            # Green
    3: {"name": "irrigated field", "rgb": [150, 250, 0]},      # Light green/Lime
    4: {"name": "dry cropland", "rgb": [150, 200, 150]},       # Pale Green/Grey green
    5: {"name": "garden land", "rgb": [200, 0, 200]},          # Magenta
    6: {"name": "arbor forest", "rgb": [150, 0, 250]},         # Purple
    7: {"name": "shrub forest", "rgb": [150, 150, 250]},       # Light Purple
    8: {"name": "park", "rgb": [200, 150, 200]},               # Pink
    9: {"name": "natural meadow", "rgb": [250, 200, 0]},       # Yellow
    10: {"name": "artificial meadow", "rgb": [200, 200, 0]},   # Olive/Light yellow
    11: {"name": "river", "rgb": [0, 0, 200]},                 # Blue
    12: {"name": "urban residential", "rgb": [250, 0, 150]},   # Pink/Magenta
    13: {"name": "lake", "rgb": [0, 150, 200]},                # Light Blue
    14: {"name": "pond", "rgb": [0, 200, 250]},                # Cyan/Light cyan
    15: {"name": "fish pond", "rgb": [150, 200, 250]},         # Light Cyan/Pale blue
    16: {"name": "snow", "rgb": [250, 250, 250]},              # White
    17: {"name": "bareland", "rgb": [200, 200, 200]},          # Light Gray/Grey
    18: {"name": "rural residential", "rgb": [200, 150, 150]}, # Mid Gray/Light grey
    19: {"name": "stadium", "rgb": [250, 200, 150]},           # Pale pink
    20: {"name": "square", "rgb": [150, 150, 0]},              # Olive/Dark yellow
    21: {"name": "road", "rgb": [250, 150, 150]},              # Light red/Salmon
    22: {"name": "overpass", "rgb": [250, 150, 0]},            # Orange
    23: {"name": "railway station", "rgb": [250, 200, 250]},   # Light Pink
    24: {"name": "airport", "rgb": [200, 150, 0]}              # Brown/Gold
}

# Class definitions
GREEN_SPACE_CLASSES = [5, 6, 7, 8, 9, 10]
URBAN_CLASSES = [1, 12, 18, 19, 20, 21, 22, 23, 24]

# Class indices
UNCLASSIFIED_INDEX = 0
GREEN_SPACE_INDEX = 1
URBAN_INDEX = 2

# Minimum content ratio for tile classification
MIN_MAJORITY_CONTENT_RATIO = 0.3 # 30% threshold for considering a tile rich in GS or Urban

# Color ID calculation constants for fast color matching
COLOR_R_MULTIPLIER = 1000000
COLOR_G_MULTIPLIER = 1000
COLOR_B_MULTIPLIER = 1

# Augmentation parameters
AUG_FLIP_PROB = 0.5
AUG_BLUR_PROB = 0.3
AUG_BGR_NOISE_STRENGTH = 0.02
AUG_NIR_NOISE_STRENGTH = 0.01

# Utility functions for color conversion
def rgb_to_bgr(rgb_color):
    """Convert RGB color to BGR."""
    return [rgb_color[2], rgb_color[1], rgb_color[0]]

def get_class_name(class_id):
    """Get the name for a class ID."""
    return CLASS_INFO.get(class_id, {"name": f"Class {class_id}"})["name"]

def get_rgb_color(class_id):
    """Get the RGB color for a class ID."""
    return CLASS_INFO.get(class_id, {"rgb": [128, 128, 128]})["rgb"]

def get_bgr_color(class_id):
    """Get the BGR color for a class ID."""
    rgb = get_rgb_color(class_id)
    return rgb_to_bgr(rgb)

# Training & Model Configuration
DEVICE = "cuda"

SEG_MODEL_NAME = "DeepLabV3Plus"
SEG_MODEL_BACKBONE = "resnet18"
PRETRAINED_BACKBONE = True
INPUT_CHANNELS = 5 # B, G, R, NIR, EVI

SEG_LR = 1e-4 # Learning rate for segmentation model optimiser
SEG_BATCH_SIZE = 8 # Batch size for training segmentation model
EPOCHS_PER_CYCLE = 4 # Number of epochs to train segmentation model per AL cycle

LR_SCHEDULER_START_CYCLE = 10 # Cycle to start decaying the learning rate
LR_SCHEDULER_GAMMA = 0.99 # Factor for learning rate decay

# Active Learning Configuration
INITIAL_POOL_RATIO = 0.05 # Percentage of data for the initial labeled pool
BATCH_ACQUISITION_SIZE = 25 # Number of samples to query/label in each AL cycle
AL_CYCLES = 100 # Total number of active learning selection cycles
AL_CANDIDATES_PER_CYCLE = 2000 # How many unlabeled tiles to evaluate per cycle
VALIDATION_FREQ = 4 # How often to run validation

# DQN Agent Configuration
REP_SET_SIZE = 500 # Number of unlabeled samples in the representative set
DQN_STATE_SIZE = 10 # Dimension of the state
DQN_HIDDEN_SIZES = (64, 32) # Hidden layer sizes for Q-Network MLP
DQN_BUFFER_SIZE = 10000 # Max size of replay buffer
DQN_BATCH_SIZE = 64 # Batch size for sampling from replay buffer
DQN_LR = 5e-5 # Learning rate for DQN optimiser
DQN_TAU = 1e-3 # Interpolation parameter for soft target network update
DQN_UPDATE_EVERY = 1 # Train DQN every N agent steps

# Tile Classifier
TC_LR = 1e-4
TC_BATCH_SIZE = 32
TC_EPOCHS = 10
TC_VAL_SPLIT = 0.15 
TC_CHECKPOINT_DIR = 'checkpoints'
TC_LOG_DIR = 'logs'
TC_TILE_CLASSIFIER_INPUT_DIR = 'tile_classifier_inputs'
TC_EXPERT_LABELS = 'expert_train_labels.csv'
TC_SAVE_NAME = "tile_classifier_model.pth"

CHECKPOINT_DIR = 'checkpoints'
LOG_DIR = 'logs'
RUN_NAME = None

MODEL_GREEN_SUFFIX = "_green"
MODEL_URBAN_SUFFIX = "_urban"

# Model names
MODEL_GREEN = 'green'
MODEL_URBAN = 'urban'

