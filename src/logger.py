import logging
import os
from datetime import datetime

# --- 1. Define the Log File Name ---
# Create a log file name using the current timestamp.
# This ensures that each time the application runs, a new log file is created.
LOG_FILE_NAME = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# --- 2. Define the Log File Path ---
# Create the full path to the log file.
# It joins the current working directory with a "logs" folder and the log file name.
LOG_FILE_PATH = os.path.join(os.getcwd(), "logs", LOG_FILE_NAME)

# --- 3. Create the Log Directory ---
# Create the "logs" directory if it doesn't already exist.
# The `exist_ok=True` argument prevents an error if the directory is already there.
os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)

# --- 4. Configure the Logging ---
# This is the core setup for the logging system.
logging.basicConfig(
    # The format string defines how each log message will look.
    # It includes the timestamp, log level, module name, and the message itself.
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    
    # The level specifies the minimum severity level to be logged.
    # Here, `logging.INFO` means it will capture INFO, WARNING, ERROR, and CRITICAL messages.
    level=logging.INFO,
    
    # This sets up the handlers, which are responsible for dispatching the logs.
    handlers=[
        # `FileHandler` sends the log output to our specified log file.
        logging.FileHandler(LOG_FILE_PATH),
        
        # `StreamHandler` sends the log output to the console (your terminal).
        logging.StreamHandler()
    ]
)

# --- 5. Create a Logger Instance ---
# Create a logger object that you can import and use in other files.
# Naming it "MedicalImageClassifierLogger" makes it easy to identify in logs.
logger = logging.getLogger("MedicalImageClassifierLogger")