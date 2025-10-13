import logging
import os
from datetime import datetime

def setup_logging(log_dir='Logs', log_filename=None, level=logging.INFO):

    os.makedirs(log_dir, exist_ok=True)
    if log_filename is None:
        log_filename = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    log_filepath = os.path.join(log_dir, log_filename)

    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger()
    logger.info(f"Logging to {log_filepath}")

    return logger

def log_config(logger, config):
    logger.info("Configuration:")
    for section in config.sections():
        logger.info(f"[{section}]")
        for key, value in config[section].items():
            logger.info(f"  {key} = {value}")
    logger.info("------------------------")