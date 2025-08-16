import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys
from src.config import get_latest_model_path, PLOT_PATH
from src.data_loader import MedicalImageDataModule
from src.model import MedicalImageClassifier
from src.logger import logger

"""
This module contains the evaluation pipeline for the trained classifier.
It loads the best model, runs it on the test dataset, and reports
key performance metrics.
"""

class EvaluationPipeline:
    """
    Encapsulates the entire model evaluation process.
    """
    def __init__(self):
        """
        Initializes the evaluation pipeline components.
        """
        # --- 1. Device Configuration ---
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # --- 2. Data Loading ---
        # Get the test data loader from our data module.
        _, _, self.test_loader = MedicalImageDataModule().get_data_loaders()
        logger.info("Test data loader prepared.")

        # --- 3. Model Loading ---
        # Instantiate the model architecture.
        self.model = MedicalImageClassifier()
        # Load the latest saved model weights from the file specified in config.
        latest_model_path = get_latest_model_path()
        self.model.load_state_dict(torch.load(latest_model_path, map_location=self.device))
        # Move the model to the configured device.
        self.model.to(self.device)
        # Set the model to evaluation mode.
        self.model.eval()
        logger.info("Model loaded from disk and set to evaluation mode.")


    def evaluate(self):
        """
        Runs the evaluation loop and prints performance metrics.
        """
        try:
            # Initialize lists to store all true labels and model predictions.
            all_labels = []
            all_preds = []

            # --- 4. Prediction Loop ---
            # Disable gradient calculations for inference.
            with torch.no_grad():
                # Loop over the test data in batches.
                for images, labels in self.test_loader:
                    # Move data to the configured device.
                    images, labels = images.to(self.device), labels.to(self.device)

                    # Get model predictions (logits).
                    outputs = self.model(images)
                    # Get the predicted class index with the highest score.
                    _, predicted = torch.max(outputs, 1)

                    # Append the true labels and predictions to our lists.
                    # We move them to the CPU to use with NumPy/scikit-learn.
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(predicted.cpu().numpy())

            logger.info("Finished making predictions on the test set.")
            
            # --- 5. Calculate and Display Metrics ---
            # Calculate overall accuracy.
            accuracy = accuracy_score(all_labels, all_preds)
            logger.info(f"Accuracy on the test set: {accuracy * 100:.2f}%")

            # Get the class names from the dataset object for the report.
            # This is a bit of a trick to get 'NORMAL', 'PNEUMONIA' etc.
            class_names = self.test_loader.dataset.classes
            
            # Print the detailed classification report.
            report = classification_report(all_labels, all_preds, target_names=class_names)
            logger.info("Classification Report:\n" + report)
            
            # Generate and save the confusion matrix plot.
            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plt.savefig(PLOT_PATH) # Save the plot to the file from config
            logger.info(f"Confusion matrix plot saved to: {PLOT_PATH}")


        except Exception as e:
            logger.error(f"An error occurred during evaluation: {e}")
            raise e


# --- Main execution block ---
if __name__ == '__main__':
    try:
        # Create an instance of the evaluation pipeline.
        pipeline = EvaluationPipeline()
        # Start the evaluation process.
        pipeline.evaluate()
    except Exception as e:
        logger.critical(f"The evaluation pipeline failed with error: {e}")
        sys.exit(1)