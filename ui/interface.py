import gradio as gr
import requests
import io
from PIL import Image

"""
This module creates a Gradio web interface for the Medical Image Classifier.
It provides a user-friendly way to upload an image and see the model's prediction.
"""

# --- 1. Define the API Endpoint URL ---
# This is the URL where our FastAPI application is running.
API_URL = "http://127.0.0.1:8000/predict/"

def predict(image: Image.Image) -> dict:
    """
    Sends an image to the FastAPI backend and returns the prediction.

    Args:
        image (PIL.Image.Image): The image uploaded by the user via the Gradio interface.

    Returns:
        dict: A dictionary mapping class labels to their confidence scores.
    """
    try:
        # --- 2. Convert Image to Bytes ---
        # Create a byte stream in memory to hold the image data.
        byte_io = io.BytesIO()
        # Save the PIL image to the byte stream in PNG format.
        image.save(byte_io, format="PNG")
        # Rewind the stream to the beginning.
        byte_io.seek(0)

        # --- 3. Send Request to API ---
        # The 'requests' library requires the file data to be in a specific format.
        # We create a dictionary where the key 'file' matches the expected
        # parameter name in our FastAPI endpoint.
        files = {'file': ('image.png', byte_io, 'image/png')}

        # Send a POST request to our API with the image file.
        response = requests.post(API_URL, files=files)
        # Raise an error if the request was unsuccessful (e.g., 4xx or 5xx status code).
        response.raise_for_status()

        # --- 4. Process the Response ---
        # Get the JSON data from the response.
        data = response.json()
        
        # Gradio's Label component expects a dictionary of {label: confidence}.
        # We create this dictionary from our API's response.
        # We also include the "other" class with a confidence of 1 minus the predicted one.
        confidence = data['confidence']
        predicted_class = data['class']

        # Determine the other class to show a full probability distribution.
        other_class = "NORMAL" if predicted_class == "PNEUMONIA" else "PNEUMONIA"

        return {predicted_class: confidence, other_class: 1 - confidence}

    except requests.exceptions.RequestException as e:
        # Handle connection errors or bad responses from the server.
        print(f"API request failed: {e}")
        # Return an error message to be displayed in the UI.
        return {"Error": f"Could not connect to the API: {e}"}


# --- 5. Create and Launch the Gradio Interface ---
# Create the Gradio Interface object.
iface = gr.Interface(
    fn=predict,  # The function to call when the user interacts with the interface.
    inputs=gr.Image(type="pil", label="Upload Chest X-Ray Image"), # The input component is an image upload box.
    outputs=gr.Label(num_top_classes=2, label="Prediction Results"), # The output component is a label showing top classes.
    title="Pneumonia Detection from Chest X-Rays 🩺",
    description="Upload a chest X-ray image to classify it as 'Normal' or 'Pneumonia'. This interface communicates with a backend model to make predictions.",
    examples=[
        ["./assets/normal_example.jpeg"],
        ["./assets/pneumonia_example.jpeg"]
    ] # Optional: provide example images. You would need to create an 'assets' folder for this.
)

# Launch the web interface.
iface.launch()