import logging
import os

from packages.utils.file import LOG_FILE_PATH

# Create a custom logger
logger = logging.getLogger(__name__)

def setup_logger():
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(LOG_FILE_PATH)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)