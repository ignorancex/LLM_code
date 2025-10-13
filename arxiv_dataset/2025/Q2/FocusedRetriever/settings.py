import json
from pathlib import Path


class Settings:
    def __init__(self, dataset_name:str, llm_model:str=None, emb_model: str=None, configs_path: str=None):
        # load configs and prompts
        prompts_path = Path(__file__).parent / 'prompts.json'
        if configs_path is None:
            configs_path = Path(__file__).parent / "config.json"
        else:
            configs_path = Path(configs_path)
        self.dataset_name = dataset_name

        # read files
        with open(prompts_path) as json_file:
            prompts = json.load(json_file)

        with open(configs_path, "r") as json_file:
            configs = json.load(json_file)

        print(f"Loaded configs from {configs_path}.")

        # load data specific configs and prompts
        data_specific_prompts = prompts[dataset_name]
        data_specific_configs = configs[dataset_name]
        self.prompts = prompts["general"]
        self.prompts.update(data_specific_prompts)
        llm_configs = configs["llm"]
        self.configs = configs["general"]
        self.configs.update(data_specific_configs)
        self.configs["llm"] = llm_configs

        # overwrite settings
        if emb_model is not None:
            self.configs["emb_model"] = emb_model
        if llm_model is not None:
            self.configs["llm_model"] = llm_model

    def get(self, config_name: str):
        return self.configs[config_name]

    def edge_type2str(self, key: str) -> str:
        return self.configs["edge_type2str"][key]
