import torch
import torch.nn as nn
import torch.optim as optim
from src.config import (
    EPOCHS,
    LEARNING_RATE,
    MODEL_SAVE_PATH
)
from src.data_loader import MedicalImageDataModule
from src.model import MedicalImageClassifier
from src.logger import logger
import sys

"""
This module contains the training pipeline for the medical image classifier.
It orchestrates the data loading, model initialization, training loop,
and model saving.
"""

class TrainingPipeline:
    """
    Encapsulates the entire model training process.
    """
    def __init__(self):
        """
        Initializes the training pipeline components.
        """
        # --- 1. Device Configuration ---
        # Set the device to a CUDA-enabled GPU if available, otherwise use the CPU.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # --- 2. Data Loading ---
        # Instantiate the data module to handle data loading and preprocessing.
        self.data_module = MedicalImageDataModule()

        # --- 3. Model Initialization ---
        # Instantiate the classifier model.
        self.model = MedicalImageClassifier()
        # Move the model to the configured device (GPU or CPU).
        self.model.to(self.device)
        logger.info("Model initialized and moved to device.")

        # --- 4. Loss Function and Optimizer ---
        # Use CrossEntropyLoss, which is standard for multi-class classification problems.
        self.criterion = nn.CrossEntropyLoss()
        # Use the Adam optimizer, a popular and effective choice.
        # It will update the model's weights with the specified learning rate.
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        logger.info("Loss function and optimizer initialized.")

    def train(self):
        """
        Executes the main training and validation loop.
        """
        try:
            # Get the data loaders for training and validation.
            train_loader, validation_loader, _ = self.data_module.get_data_loaders()

            # Initialize a variable to track the best validation loss to save the best model.
            best_val_loss = float('inf')

            # --- 5. Main Training Loop ---
            # Loop for the specified number of epochs.
            for epoch in range(EPOCHS):
                # --- Training Phase ---
                self.model.train()  # Set the model to training mode.
                running_loss = 0.0  # Initialize loss for this epoch.

                # Loop over the training data in batches.
                for i, (images, labels) in enumerate(train_loader):
                    # Move images and labels to the configured device.
                    images, labels = images.to(self.device), labels.to(self.device)

                    # --- Forward pass ---
                    outputs = self.model(images)  # Get model predictions.
                    loss = self.criterion(outputs, labels)  # Calculate the loss.

                    # --- Backward pass and optimization ---
                    self.optimizer.zero_grad()  # Clear previous gradients.
                    loss.backward()  # Compute gradients of the loss w.r.t. model parameters.
                    self.optimizer.step()  # Update the model's weights.

                    # Add the batch loss to the running total.
                    running_loss += loss.item()

                # Calculate average training loss for the epoch.
                avg_train_loss = running_loss / len(train_loader)
                logger.info(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.4f}")

                # --- Validation Phase ---
                self.model.eval()  # Set the model to evaluation mode.
                val_loss = 0.0  # Initialize validation loss.
                correct = 0  # Initialize count of correct predictions.
                total = 0  # Initialize count of total predictions.

                # Disable gradient calculation for efficiency.
                with torch.no_grad():
                    # Loop over the validation data.
                    for images, labels in validation_loader:
                        # Move images and labels to the device.
                        images, labels = images.to(self.device), labels.to(self.device)

                        # Get model predictions.
                        outputs = self.model(images)
                        # Calculate loss.
                        loss = self.criterion(outputs, labels)
                        val_loss += loss.item()

                        # Calculate accuracy.
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                # Calculate average validation loss and accuracy.
                avg_val_loss = val_loss / len(validation_loader)
                val_accuracy = 100 * correct / total
                logger.info(f"Epoch [{epoch+1}/{EPOCHS}], Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

                # --- 6. Save the Best Model ---
                # Check if the current validation loss is the best we've seen so far.
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss  # Update the best validation loss.
                    torch.save(self.model.state_dict(), MODEL_SAVE_PATH)  # Save the model's state.
                    logger.info(f"New best model saved with validation loss: {best_val_loss:.4f}")

            logger.info("Training finished.")

        except Exception as e:
            logger.error(f"An error occurred during training: {e}")
            raise e

# --- Main execution block ---
if __name__ == '__main__':
    try:
        # Create an instance of the training pipeline.
        pipeline = TrainingPipeline()
        # Start the training process.
        pipeline.train()
    except Exception as e:
        # Log any exceptions that occur and exit.
        logger.critical(f"The training pipeline failed with error: {e}")
        sys.exit(1)