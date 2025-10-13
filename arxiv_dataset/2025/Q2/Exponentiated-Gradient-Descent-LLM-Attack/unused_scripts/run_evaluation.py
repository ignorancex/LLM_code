import logging
from sacred import Experiment
import seml

from src.evaluate import add_perplexity_and_toxicity_to_evaluation, run_evaluation



ex = Experiment()
seml.setup_logger(ex)


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))


@ex.automain
def run(batch_size: int, only_last_layer):
    import sys
    print(sys.path)

    logging.info('Received the following configuration:')
    logging.info(f'batch_size: {batch_size} | only_last_layer: {only_last_layer}')
    
    #run_evaluation(verbose=True, skip_existing=True)
    add_perplexity_and_toxicity_to_evaluation(batch_size=batch_size, only_last_layer=only_last_layer)
    
    return {"note":"results are not handled by seml"}
    