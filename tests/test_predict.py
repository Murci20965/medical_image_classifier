import pytest
import torch
from unittest.mock import patch, MagicMock
from PIL import Image
import io

from app.predict import PredictionService

@patch('app.predict.get_latest_model_path')
@patch('app.predict.MedicalImageClassifier')
@patch('app.predict.torch.load')
def test_singleton_pattern(mock_torch_load, mock_model_class, mock_get_path):
    """
    Tests that the PredictionService correctly implements the singleton pattern.
    """
    # FIX: Reset the singleton state before testing its creation.
    PredictionService._instance = None
    mock_get_path.return_value = 'dummy/path/model.h5'

    service1 = PredictionService()
    service2 = PredictionService()

    assert service1 is service2, "PredictionService should be a singleton"
    mock_torch_load.assert_called_once()
    mock_get_path.assert_called_once()


@patch('app.predict.get_latest_model_path')
@patch('app.predict.MedicalImageClassifier')
@patch('app.predict.torch.load')
@patch('app.predict.Image.open')
def test_predict_success(mock_image_open, mock_torch_load, mock_model_class, mock_get_path):
    """
    Tests a successful prediction call.
    """
    mock_get_path.return_value = 'dummy/path/model.h5'

    mock_model_instance = MagicMock()
    mock_model_instance.to.return_value = mock_model_instance
    mock_model_instance.return_value = torch.tensor([[0.1, 2.5]])
    mock_model_class.return_value = mock_model_instance

    # --- Create fake image data ---
    img = Image.new('L', (1, 1))
    byte_io = io.BytesIO()
    img.save(byte_io, 'PNG')
    image_bytes = byte_io.getvalue()

    # --- Configure the mock for Image.open to return our dummy image ---
    mock_image_open.return_value = img

    # Reset the singleton instance to ensure the mocks are used for THIS test
    PredictionService._instance = None
    service = PredictionService()
    result = service.predict(image_bytes=image_bytes)

    # The Assertions
    assert isinstance(result, dict)
    assert 'error' not in result, f"Prediction returned an error: {result.get('details')}"
    assert result.get('class') == 'PNEUMONIA'
    assert isinstance(result.get('confidence'), float)
    assert 0.0 <= result.get('confidence') <= 1.0