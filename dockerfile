# Dockerfile

# --- Stage 1: The Builder ---
# Use a specific version of Python for reproducibility. 'slim' is a smaller base image.
FROM python:3.11-slim AS builder

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies that some Python packages might need
RUN apt-get update && apt-get install -y --no-install-recommends gcc build-essential

# Copy the requirements file into the container
COPY requirements.txt .

# Create a virtual environment inside the builder stage
RUN python -m venv /opt/venv
# Activate it for subsequent commands
ENV PATH="/opt/venv/bin:$PATH"

# Install the Python packages into the virtual environment
RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt


# --- Stage 2: The Final Image ---
# Start from the same clean Python base image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the virtual environment (with all the installed packages) from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy the application source code and the trained model
COPY ./src ./src
COPY ./app ./app
COPY ./models/ ./models/

# Activate the virtual environment for the final image
ENV PATH="/opt/venv/bin:$PATH"

# Expose the port that the FastAPI application will run on
EXPOSE 8000

# Define the command to run the application when the container starts
# This runs our Uvicorn server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]