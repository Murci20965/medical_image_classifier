from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import io

# Import the singleton instance of our prediction service
from app.predict import prediction_service
from src.logger import logger

"""
This module defines the FastAPI application and its API endpoints.
"""

# --- 1. Initialize the FastAPI App ---
# Create an instance of the FastAPI class.
# The title will be used in the automatic API documentation.
app = FastAPI(title="Medical Image Classifier API")


# --- 2. Define a Root Endpoint ---
# This decorator tells FastAPI to create a GET endpoint at the root URL ("/").
@app.get("/", tags=["Root"])
async def read_root():
    """
    A simple endpoint to check if the API is running.
    """
    # Return a welcome message.
    return {"message": "Welcome to the Medical Image Classifier API!"}


# --- 3. Define the Prediction Endpoint ---
# This decorator creates a POST endpoint at the "/predict/" URL.
# It's a POST request because the client is sending data (a file) to the server.
@app.post("/predict/", tags=["Prediction"])
async def predict_image(file: UploadFile = File(...)):
    """
    Receives an image file, performs a prediction, and returns the result.

    Args:
        file (UploadFile): The image file uploaded by the client.

    Returns:
        JSONResponse: A JSON object containing the prediction result or an error.
    """
    # --- 3a. Validate Input File ---
    # Check the content type of the uploaded file to ensure it's an image.
    if not file.content_type.startswith("image/"):
        # If it's not an image, raise an HTTPException (which FastAPI turns into an error response).
        logger.warning(f"Invalid file type uploaded: {file.content_type}")
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        # --- 3b. Read and Predict ---
        # Read the contents of the uploaded file into bytes.
        # `await` is used because reading a file is an I/O operation.
        image_bytes = await file.read()
        
        # Log that a prediction is being made.
        logger.info(f"Received file: {file.filename}. Making a prediction.")
        
        # Use our prediction service to get the result.
        result = prediction_service.predict(image_bytes)
        
        # --- 3c. Handle Prediction Errors ---
        # Check if the prediction service returned an error.
        if 'error' in result:
            # If so, raise an HTTPException with the error details.
            raise HTTPException(status_code=500, detail=result['error'])
            
        # --- 3d. Return Successful Prediction ---
        # If everything is successful, return the result as a JSON response.
        return JSONResponse(content=result)

    except Exception as e:
        # Catch any other unexpected errors during the process.
        logger.error(f"An unexpected error occurred for file {file.filename}: {e}")
        # Return a generic server error response.
        raise HTTPException(status_code=500, detail="An internal server error occurred.")