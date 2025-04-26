import config
import logging


logging.basicConfig(format=config.LOGGING_FORMAT,
                    datefmt=config.LOGGING_DATE_FORMAT)
logging.root.setLevel(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # we just import the config.py and thus the folders are created
    logger.info("Folders initialized")


if __name__ == '__main__':
    main()