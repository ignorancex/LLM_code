# template: https://github.com/ashleve/lightning-hydra-template/blob/main/run.py
import hydra
from omegaconf import DictConfig
import os 
# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dir_path = os.path.dirname(os.path.realpath(__file__))



@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from train_new import train
    from test import test
    from utils import utils
    from find_best_model import find_best_model
    from find_best_model_classification import find_best_model_cls
    from find_per_disease_performance import find_per_disease_performance

    # A couple of optional utilities:
    # - disabling python warnings
    # - easier access to debug mode
    # - forcing debug friendly configuration
    # You can safely get rid of this line if you don't want those
    utils.extras(config)
    

    # Pretty print config using Rich library
    if config.get("print_config"):
        utils.print_config(config, resolve=True)

    #Calculate the results for trained models
    if config.get("show_result",False):
        return find_best_model_cls(config)

    if config.get("show_disease_performance",False):
        return find_per_disease_performance(config)


    if config.get("load_checkpoint"):
        test(config)
    else:
        # Train model
        return train(config)


if __name__ == "__main__":
    print("in MAIN")
    main()
    print("Out MAIN")