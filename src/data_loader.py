import os # Used for operating system-dependent functionalities, like path joining.
import torch # The main PyTorch library.
from torch.utils.data import DataLoader # Helps create iterable data batches.
from torchvision import datasets, transforms # Provides access to datasets and image transformations.
from src.config import (
    RAW_DATA_PATH, 
    TRAIN_DIR, 
    VALIDATION_DIR, 
    TEST_DIR, 
    IMAGE_SIZE, 
    BATCH_SIZE, 
    CHANNELS
) # Import all necessary constants from our config file.
from src.logger import logger # Import our custom logger.

"""
This module handles data loading, preprocessing, and batching.
It defines a class `MedicalImageDataModule` to encapsulate all data-related logic.
"""

class MedicalImageDataModule:
    """
    A class to handle the loading and transforming of the medical image dataset.
    It prepares the data for training, validation, and testing by applying
    transformations and organizing it into batches using DataLoader.
    """
    def __init__(self):
        """
        Initializes the data module.
        Sets up paths and defines the image transformations.
        """
        # --- 1. Store configuration constants ---
        self.raw_data_path = RAW_DATA_PATH    # The base path to the raw data folders.
        self.train_dir = TRAIN_DIR            # Name of the training directory.
        self.validation_dir = VALIDATION_DIR  # Name of the validation directory.
        self.test_dir = TEST_DIR              # Name of the testing directory.
        self.image_size = IMAGE_SIZE          # Target image size.
        self.batch_size = BATCH_SIZE          # Number of images per batch.
        self.channels = CHANNELS              # Number of image channels (1 for greyscale).

        # --- 2. Define Image Transformations ---
        # `transforms.Compose` chains multiple transformations together.
        self.transform = transforms.Compose([
            # Resize the image to the size specified in the config.
            transforms.Resize(self.image_size),
            
            # Since the model expects a certain number of channels, ensure it's correct.
            # `num_output_channels=1` is for greyscale. Use 3 for RGB.
            transforms.Grayscale(num_output_channels=self.channels),

            # Convert the image from a PIL Image format to a PyTorch Tensor.
            # This also scales the image's pixel values from [0, 255] to [0.0, 1.0].
            transforms.ToTensor(),
            
            # Normalize the tensor image. For greyscale, we provide one mean and one std dev.
            # For 3-channel images, this would be (mean1, mean2, mean3), (std1, std2, std3).
            # These values are standard for many datasets and models.
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        logger.info("Data transformations initialized.")


    def get_data_loaders(self):
        """
        Creates and returns DataLoader instances for train, validation, and test datasets.
        """
        try:
            # --- 3. Create Datasets using ImageFolder ---
            # `ImageFolder` is a generic data loader where images are arranged in this way:
            # root/class_x/xxx.ext
            # root/class_y/yyy.ext
            
            # Create the training dataset.
            train_dataset = datasets.ImageFolder(
                root=os.path.join(self.raw_data_path, self.train_dir), # Path to the train folder.
                transform=self.transform # Apply the transformations we defined.
            )
            
            # Create the validation dataset.
            validation_dataset = datasets.ImageFolder(
                root=os.path.join(self.raw_data_path, self.validation_dir), # Path to the validation folder.
                transform=self.transform # Apply the same transformations.
            )
            
            # Create the test dataset.
            test_dataset = datasets.ImageFolder(
                root=os.path.join(self.raw_data_path, self.test_dir), # Path to the test folder.
                transform=self.transform # Apply the same transformations.
            )

            logger.info(f"Training dataset loaded with {len(train_dataset)} images.")
            logger.info(f"Validation dataset loaded with {len(validation_dataset)} images.")
            logger.info(f"Test dataset loaded with {len(test_dataset)} images.")

            # --- 4. Create DataLoaders ---
            # `DataLoader` takes a dataset and wraps it in an iterable to easily
            # access samples in batches.

            # Create the DataLoader for the training set.
            train_loader = DataLoader(
                dataset=train_dataset,      # The dataset to load from.
                batch_size=self.batch_size, # How many samples per batch to load.
                shuffle=True                # Shuffle the data at every epoch to improve training.
            )
            
            # Create the DataLoader for the validation set.
            validation_loader = DataLoader(
                dataset=validation_dataset, # The validation dataset.
                batch_size=self.batch_size, # Batch size.
                shuffle=False               # No need to shuffle validation data.
            )
            
            # Create the DataLoader for the test set.
            test_loader = DataLoader(
                dataset=test_dataset,       # The test dataset.
                batch_size=self.batch_size, # Batch size.
                shuffle=False               # No need to shuffle test data.
            )

            logger.info("DataLoaders created successfully for train, validation, and test sets.")
            return train_loader, validation_loader, test_loader

        except FileNotFoundError as e:
            # --- 5. Error Handling ---
            # If a directory is not found, log a critical error and re-raise the exception.
            logger.error(f"Error loading data: {e}. Check your data paths in config.py.")
            raise e