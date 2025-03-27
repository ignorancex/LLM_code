import importlib
from src.TPP.utils import getLogger

'''
We use this parameter to control model's memory usage while running the event-time prediction task.
Used in FENN, FullyNN, SAHP, and THP
'''
memory_ceiling = 3e7

logger = getLogger(__name__)

'''
Similar to find_dataset(), these two functions load model by the name.
'''
def get_model(name, rank = 0):
    try:
        model = model_zoo(name)
        if rank == 0:
            logger.info(f"Model name: {name}.")
        return model
    except Exception as e:
        if rank == 0:
            logger.exception(f'{e}.')
            logger.exception(f"Model {name} not found! Please check the model name in your script. Available names are: fenn, fullynn, ifib_n, ifib_c, lognormmix, rmtpp, sahp, and thp.")


def model_zoo(name):
    module = importlib.import_module('.' + name, package = 'src.TPP.model')
    return module.get_model()