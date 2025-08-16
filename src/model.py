import torch  # The main PyTorch library.
import torch.nn as nn  # A submodule for building neural networks.
from torchvision import models  # Provides access to pre-trained models.
from src.config import NUM_CLASSES, CHANNELS  # Import our project-specific constants.
from src.logger import logger  # Import our custom logger.

"""
This module defines the neural network architecture for the image classifier.
It uses transfer learning with a pre-trained ResNet50 model.
"""

class MedicalImageClassifier(nn.Module):
    """
    A CNN model for classifying medical images.

    This class loads a pre-trained ResNet50 model and adapts it for our
    specific classification task. It handles modifications for greyscale input
    and replaces the final classification layer.
    """
    def __init__(self):
        """
        Initializes the model, loads the pre-trained ResNet50,
        and modifies it for our task.
        """
        # Call the constructor of the parent class (nn.Module).
        super(MedicalImageClassifier, self).__init__()

        # --- 1. Load the Pre-trained Model ---
        # Load the ResNet50 model that was pre-trained on the ImageNet dataset.
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        logger.info("Pre-trained ResNet50 model loaded.")

        # --- 2. Adapt for Greyscale Images (The Greyscale Challenge) ---
        # The original ResNet50 was trained on 3-channel (RGB) images. Our images
        # are 1-channel (greyscale). We must modify the first convolutional layer
        # to accept 1-channel input instead of 3.

        # Get the weights from the original first layer.
        original_weights = self.resnet.conv1.weight.clone()

        # Create a new first convolutional layer that accepts `CHANNELS` (1) input channels.
        # All other parameters (output channels, kernel size, etc.) remain the same.
        self.resnet.conv1 = nn.Conv2d(
            in_channels=CHANNELS,
            out_channels=64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False
        )

        # To leverage the pre-trained knowledge, we average the weights of the
        # original 3 channels to create a single-channel weight.
        with torch.no_grad():
            self.resnet.conv1.weight[:, :] = torch.mean(original_weights, dim=1, keepdim=True)
        
        logger.info(f"Modified the first convolutional layer to accept {CHANNELS}-channel greyscale images.")

        # --- 3. Freeze Pre-trained Layers ---
        # We freeze all the layers in the network to prevent their weights
        # from being updated during the initial training phases.
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        logger.info("Froze all layers of the pre-trained ResNet model.")

        # --- 4. Replace the Final Classification Layer ---
        # The final layer of the original ResNet50 (called `fc`) was designed to
        # classify 1000 ImageNet classes. We need to replace it with a new one
        # for our `NUM_CLASSES` (e.g., 2 for Normal/Pneumonia).

        # Get the number of input features for the final layer.
        num_ftrs = self.resnet.fc.in_features

        # Create a new fully connected (Linear) layer.
        # It takes `num_ftrs` as input and outputs `NUM_CLASSES` logits.
        self.resnet.fc = nn.Linear(num_ftrs, NUM_CLASSES)
        
        logger.info(f"Replaced the final classification layer for {NUM_CLASSES} classes.")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input batch of images.

        Returns:
            torch.Tensor: The output logits from the model.
        """
        # Pass the input tensor `x` through the modified ResNet model.
        return self.resnet(x)