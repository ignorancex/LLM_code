import logging
import os
from . import Config
from .Config import config
import Dataset.UtilIO as uio
import sys


def logConfig(config):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    
    path = config.log.path
    uio.mkdirsExceptFile(path)
    format = "[%(asctime)-15s]<%(thread)-5d><%(levelname)s> \t%(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    logger=logging.getLogger()


    console = logging.StreamHandler(stream=sys.stdout)
    loggingLevel = logging.INFO#logging.DEBUG if config.utility.debug else logging.INFO
    console.setLevel(loggingLevel)
    fm=logging.Formatter(format,datefmt)
    console.setFormatter(fm)
    logger.addHandler(console)

    num=0
    finalPath=path
    while os.path.exists(finalPath):
        finalPath = path.replace(".log","_%d.log"%num)
        if path.find(".log")<0:
            finalPath = path+str(num)
        num+=1
    fh = logging.FileHandler(finalPath)
    fh.setFormatter(fm)
    console.setLevel(loggingLevel)
    logger.addHandler(fh)

    logger.setLevel(loggingLevel)
    logging.info("Init Log Complete")