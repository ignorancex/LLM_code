import logging
import os
#to console
stream_log = logging.StreamHandler()
stream_log.setLevel(logging.INFO)

stream_log.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))


#to file

dirname = "../log"
if not os.path.exists(dirname):
    os.makedirs(dirname)

file_log = logging.FileHandler('{}/test_logger.txt'.format(dirname), 'w')
file_log.setLevel(logging.INFO)
file_log.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
# root logger
logging.getLogger().addHandler(stream_log)
logging.getLogger().addHandler(file_log)
# level of root logger must minumum of loggers
logging.getLogger().setLevel(logging.INFO)

#import logging
#import logging.config
#from logging import getLogger, StreamHandler, DEBUG
#logger =logging.getLogger(__name__)
#handler = StreamHandler()
#handler.setLevel(DEBUG)
#logger.setLevel(DEBUG)
#logger.addHandler(handler)
#logger.propagate = False

#logging.config.fileConfig('log/logging.conf')
