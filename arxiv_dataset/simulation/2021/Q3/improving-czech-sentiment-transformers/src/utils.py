import matplotlib.pyplot as plt
import datetime

import logging
import tensorflow as tf

from config import LOGGING_FORMAT, LOGGING_DATE_FORMAT


logger = logging.getLogger(__name__)

def visaulize_training(history, path):
  plt.plot(history['train_acc'], label='train accuracy')
  plt.plot(history['val_acc'], label='validation accuracy')

  plt.title('Training history')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend()
  plt.ylim([0, 1])

  plt.savefig(path)



def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def disable_tensorflow_gpus():
    logger.info("Disabling GPUs for tensorflow")
    try:
        # Disable all GPUS
        tf.config.set_visible_devices([], 'GPU')
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            assert device.device_type != 'GPU'
    except:
        logger.error("Invalid device or cannot modify virtual devices once initialized.")

    logger.info("GPUs for tensorflow disabled")