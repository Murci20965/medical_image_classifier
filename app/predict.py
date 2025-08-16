import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io

from src.model import MedicalImageClassifier
from src.config import get_latest_model_path, IMAGE_SIZE, CHANNELS
from src.logger import logger

"""
This module contains the prediction service, which handles model loading
and inference for a single image.
"""

class PredictionService:
    """
    A service to handle the loading of the model and perform predictions.
    This class uses a singleton pattern to ensure the model is loaded only once.
    """
    _instance = None
    _model = None

    def __new__(cls):
        """
        Implements the singleton pattern. On the first call, it creates a new
        instance and loads the model. Subsequent calls return the existing instance.
        """
        # If no instance of this class exists yet...
        if cls._instance is None:
            logger.info("Creating new PredictionService instance.")
            # Create a new instance using the parent class's __new__ method.
            cls._instance = super(PredictionService, cls).__new__(cls)
            # --- 1. Load Model ---
            # Set the device to a CUDA-enabled GPU if available, otherwise use the CPU.
            cls.device = "cuda" if torch.cuda.is_available() else "cpu"
            # Instantiate the model architecture.
            cls._model = MedicalImageClassifier()
            # Load the saved model weights from the specified path.
            # `map_location` ensures the model loads correctly whether on CPU or GPU.
            latest_model_path = get_latest_model_path()
            cls._model.load_state_dict(torch.load(latest_model_path, map_location=cls.device))
            # Move the model to the configured device.
            cls._model.to(cls.device)
            # Set the model to evaluation mode, which is crucial for inference.
            cls._model.eval()
            logger.info(f"Model loaded on {cls.device} and set to evaluation mode.")
        # Return the existing instance.
        return cls._instance

    def _preprocess_image(self, image_bytes: bytes) -> torch.Tensor:
        """
        Preprocesses a raw image from bytes for model inference.

        Args:
            image_bytes (bytes): The raw bytes of the image file.

        Returns:
            torch.Tensor: The preprocessed image as a PyTorch tensor.
        """
        # --- 2. Define Image Transformations ---
        # The transformations must be IDENTICAL to the ones used for training/validation.
        transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.Grayscale(num_output_channels=CHANNELS),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Open the image from its raw bytes.
        image = Image.open(io.BytesIO(image_bytes))
        # Apply the transformations and add a batch dimension (B, C, H, W).
        # `unsqueeze(0)` adds the batch dimension, e.g., from [1, 224, 224] to [1, 1, 224, 224].
        return transform(image).unsqueeze(0)

    def predict(self, image_bytes: bytes) -> dict:
        """
        Performs a prediction on a single image.

        Args:
            image_bytes (bytes): The raw bytes of the image file to be classified.

        Returns:
            dict: A dictionary containing the predicted class and its confidence score.
        """
        try:
            # --- 3. Preprocess ---
            # Preprocess the input image bytes into a tensor.
            tensor = self._preprocess_image(image_bytes)
            # Move the tensor to the same device as the model.
            tensor = tensor.to(self.device)

            # --- 4. Inference ---
            # Perform inference within a `no_grad` context for efficiency.
            with torch.no_grad():
                # Get the raw model output (logits).
                outputs = self._model(tensor)

            # --- 5. Post-process ---
            # Apply the softmax function to convert logits to probabilities.
            probabilities = F.softmax(outputs, dim=1)[0]
            # Get the probability of the predicted class (the max probability).
            confidence = torch.max(probabilities).item()
            # Get the index of the predicted class.
            predicted_index = torch.argmax(probabilities).item()

            # Define class names (make sure this order matches training).
            class_names = ['NORMAL', 'PNEUMONIA']
            # Get the name of the predicted class.
            predicted_class = class_names[predicted_index]

            logger.info(f"Prediction: {predicted_class} with confidence {confidence:.4f}")
            
            # Return the result in a structured dictionary.
            return {
                'class': predicted_class,
                'confidence': confidence
            }

        except Exception as e:
            logger.error(f"An error occurred during prediction: {e}")
            # In a real app, you might return a specific error response.
            return {
                'error': 'Failed to process image.',
                'details': str(e)
            }

# --- Create a single, shared instance of the service ---
prediction_service = PredictionService()