import os

# Base Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root (MTP-2/)
DATA_DIR = r"D:\M.Tech Project\Dataset\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"

# Data Configuration
SAMPLING_RATE = 500  # Hz
NUM_LEADS = 12
SEQUENCE_LENGTH = 5000  # 10 seconds * 500 Hz

# Model Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-4 # Better starting point
NUM_EPOCHS = 80 
PATIENCE = 15 

# Architecture Parameters
CNN_FILTERS = [64, 128, 256, 512]
KERNEL_SIZES = [3, 5, 7, 11]
TRANSFORMER_LAYERS = 4
TRANSFORMER_HEADS = 8
HIDDEN_DIM = 512
DROPOUT = 0.3
NUM_CLASSES = 6  # Added AFib as 6th class

# Diagnostic Superclasses
DIAGNOSTIC_CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP', 'AFib']
LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# Optimized Decision Thresholds (Default: 0.5)
THRESHOLDS = [0.59, 0.48, 0.46, 0.51, 0.51, 0.61]  # Boosts accuracy by ~0.3%
