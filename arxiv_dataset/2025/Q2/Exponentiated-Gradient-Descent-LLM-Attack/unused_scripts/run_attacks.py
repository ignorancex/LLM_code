import logging
from sacred import Experiment
import seml

from src.embedding_attack import run_attack
from src.utils import (
    load_model_and_tokenizer,
    get_model_path
)

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
def run(model_path: str, model_name: str, dataset_name: str, test_split: float, skip_existing_experiments: bool, attack_config: dict):
    import sys
    print(sys.path)

    logging.info('Received the following configuration:')
    logging.info(f'model_path: {model_path} | model_name: {model_name} | dataset: {dataset_name} | attack_config: {attack_config}')
    logging.info(f'attack_config: {attack_config}')

    model_path_from_name = get_model_path(model_name)
    if model_path_from_name != model_path:
        logging.warning(f"model_path {model_path} does not get_model_path({model_path_from_name}) setting to auto detected model path")
        model_path = model_path_from_name

    if "paper_models" in model_path:
        complete_model_path = model_path
    else:
        complete_model_path = model_path + model_name
    
    model, tokenizer = load_model_and_tokenizer(
    complete_model_path, low_cpu_mem_usage=True, use_cache=False, device="cuda:0"
    )
    
    run_attack(model, tokenizer, model_name=model_name, dataset_name=dataset_name, test_split=test_split, skip_existing_experiments=skip_existing_experiments, attack_config=attack_config)
    
    # the returned result will be written into the database
    return {"note":"results are not handled by seml"}