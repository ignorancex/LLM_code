from . import Config
from .Config import loadConfig
import logging
from . import LogConfig
from .LogConfig import logConfig
import argparse
import traceback

def programInit():
    print("Usage: --config_file file_name\n\tdefault value is trainConfig.json")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default='dataset.json')
    parser_args, _ = parser.parse_known_args()
    print("Load Config %s"%parser_args.config_file)
    config = loadConfig(parser_args.config_file)

    logConfig(config)
    logging.info("======================")
    logging.info("Program Init")
    logging.info("Config Path: %s"%parser_args.config_file)
    return config

def globalCatch(funcName):
    try:
        funcName()
    except BaseException as e:
        logging.fatal("Exception Occurs!")
        logging.fatal(str(e))
        logging.fatal(traceback.format_exc())