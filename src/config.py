import os  # Used for operating system-dependent functionalities, like path joining.
import glob  # Used to find files matching a specific pattern (like all .h5 files).
from datetime import datetime  # Used to get the current timestamp for versioning.

"""
This module defines the configuration constants for the project.
All file paths, model parameters, and other constants are centralized here
to make the application easier to manage and configure.
"""

# --- 1. Main Project Paths ---

# Get the root directory of the project.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
# Define the path to the main data folder.
DATA_PATH = os.path.join(PROJECT_ROOT, "data")
# Define the path to the raw, unprocessed data.
RAW_DATA_PATH = os.path.join(DATA_PATH, "raw", "chest_xray")
# Define the path where processed data will be saved (if any).
PROCESSED_DATA_PATH = os.path.join(DATA_PATH, "processed")
# Define the path to the directory where models will be saved.
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
# Define the path to the directory where plots will be saved.
PLOT_DIR = os.path.join(PROJECT_ROOT, "plots")


# --- 2. Create Directories ---

# Create the "models" directory if it doesn't already exist.
# The `exist_ok=True` argument prevents an error if the directory is already there.
os.makedirs(MODEL_DIR, exist_ok=True)
# Create the "plots" directory if it doesn't already exist.
os.makedirs(PLOT_DIR, exist_ok=True)


# --- 3. Dataset Configuration ---

# Define the names of the subdirectories for the dataset splits.
# These must match the folder names inside your `raw` data directory.
TRAIN_DIR = "train"
VALIDATION_DIR = "val"
TEST_DIR = "test"


# --- 4. Model and Training Configuration ---

# The size (height, width) to which all images will be resized.
IMAGE_SIZE = (224, 224)
# The number of color channels in the images (1 for greyscale).
CHANNELS = 1
# The number of training examples to process in one go.
BATCH_SIZE = 32
# The number of complete passes through the entire training dataset.
EPOCHS = 10
# The learning rate for the optimizer.
LEARNING_RATE = 0.001
# The number of classes/categories the model needs to predict.
NUM_CLASSES = 2
# The full path for saving the evaluation plot.
PLOT_PATH = os.path.join(PLOT_DIR, "plot.png")


# --- 5. Model Naming and Versioning ---

# A descriptive name for the model being trained.
MODEL_NAME = "pneumonia_classifier"
# The version number for the current model architecture or experiment.
MODEL_VERSION = "v1"
# Get the current time as a string in the format YYYYMMDD_HHMMSS.
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Construct a unique, versioned filename for saving the model.
# Example: "pneumonia_classifier_v1_20250813_043046.h5"
VERSIONED_MODEL_NAME = f"{MODEL_NAME}_{MODEL_VERSION}_{TIMESTAMP}.h5"
# Create the full path where the new model will be saved.
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, VERSIONED_MODEL_NAME)


# --- 6. Helper Function to Find Latest Model ---

def get_latest_model_path():
    """
    Finds and returns the full path of the most recently created model file
    in the models directory. This allows other scripts to use the latest model
    without knowing its exact timestamped filename.
    """
    # Use glob to create a list of all files in MODEL_DIR that end with '.h5'.
    list_of_models = glob.glob(os.path.join(MODEL_DIR, '*.h5'))
    
    # Check if the list is empty (no models found).
    if not list_of_models:
        # If no models exist, return None. The calling script must handle this.
        return None
        
    # Find the newest file in the list based on its creation time on the system.
    latest_model = max(list_of_models, key=os.path.getctime)
    
    # Return the full path to this latest model file.
    return latest_model