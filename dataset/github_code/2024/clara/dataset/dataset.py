import datasets
from os.path import join as pjoin
from ast import literal_eval
from tqdm import tqdm
import os
import json
import pandas as pd
import gdown
import sys
logger = datasets.logging.get_logger(__name__)

# spack load /u64kqpe /tymduoi && pip install -U datasets gdown
class ClaraDataConfig(datasets.BuilderConfig):
  def __init__(self, url, data_dir, **kwargs):
    super(ClaraDataConfig, self).__init__(**kwargs)
    self.URL = url
    self.DATASET_DIRECTORY = data_dir

class ClaraDataBuilder(datasets.GeneratorBasedBuilder):
    '''
    '''
    BUILDER_CONFIG_CLASS = ClaraDataConfig
    DATA_SOURCES = ["CLARA"]
    BUILDER_CONFIGS = [
        ClaraDataConfig(name='ClaraDataset', 
            version=datasets.Version("1.0.0"), 
            description="Parsed output of Clara program repair 1.",
            url="default",
            data_dir="default2")
    ]
    
    test_dummy_data = False
    def _info(self):
      print("HERE YOU GO")
      print(self.config)
      return datasets.DatasetInfo(
          description="Parsed output of Clara program repair 2.",
          features=datasets.Features(
            {
              "Problem": datasets.Value("string"),
              "First Output": datasets.Value("string"),
              "Error Output": datasets.Value("string"),
              "Correct File": datasets.Value("string"),
              "Incorrect File": datasets.Value("string"),
              "Test Available": datasets.Value("string"),
              "Correct Locs": datasets.Value("int32"),
              "Incorrect Locs": datasets.Value("int32"),
              "Old Incorrect Locs": datasets.Value("int32"),
              "Correct Exprs": datasets.Value("int32"),
              "Incorrect Exprs": datasets.Value("int32"),
              "Old Incorrect Exprs": datasets.Value("int32"),
              "Repairs": datasets.Value("int32"),
              "Technique": datasets.Value("int32"),
              "GM Score": datasets.Value("float64"),
              "Locs": datasets.Value("string"),
              "Count": datasets.Value("int64"),
              "Structure Mismatch": datasets.Value("bool"),
              "Repair": datasets.Value("bool"),
              "Repair Correct": datasets.Value("string"),
              "Parse Error": datasets.Value("string"),
              "Cost": datasets.Value("float64"),
              "Second Output": datasets.Value("string"),
              "Match": datasets.Value("string")
            }
          ),
          supervised_keys=None
      )
    

    def _split_generators(self, dl_manager):
        self.load_data(dl_manager)
        return [
            datasets.SplitGenerator(
                name=datasets.Split("DEFAULT"),
                gen_kwargs={"split": "default"},
            )
        ]
    
    
    def load_data(self, dl_manager):
      # URL = 'https://drive.google.com/uc?id=1D6UCjkcV3BoWJNNq8BPgz5w4Ciai-Q_x'
      downloaded = os.path.isfile(pjoin(self.config.DATASET_DIRECTORY, 'parsed_results.zip'))
      if not downloaded:
        gdown.download(self.config.URL, pjoin(self.config.DATASET_DIRECTORY, 'parsed_results.zip'), quiet=False)
      extracted_files = dl_manager.extract(pjoin(self.config.DATASET_DIRECTORY, 'parsed_results.zip'))
      print("Extracted Files", extracted_files)
      self.extracted_files = extracted_files
      

    def _generate_examples(self, split):
        # print(split)
        print(f"generating examples from {split}")
        # print(self.data)
        
        for problem in os.listdir(pjoin(self.extracted_files, 'parsed_results')):
          for json_f in os.listdir(pjoin(self.extracted_files, 'parsed_results', problem)):
            data = json.load(open(pjoin(self.extracted_files, 'parsed_results', problem, json_f)))
            for id in data.keys():
              if data[id].get("Correct File", None) is None:
                continue 
              # print(f'{problem}_{json_f}_{id}')
              yield f'{problem}_{json_f}_{id}', {
                "Problem": problem,
                "First Output": data[id]['First Output'],
                "Error Output": data[id]["Error Output"],
                "Correct File": data[id].get("Correct File", id.split("_")[0]),
                "Incorrect File": data[id].get("Incorrect File", json_f.split("_")[0]),
                "Test Available": True if data[id].get("Test Available", False) == "Yes" else False,
                "Correct Locs": data[id].get("Correct Locs", None),
                "Incorrect Locs": data[id].get("Incorrect Locs", None),
                "Old Incorrect Locs": data[id].get("Old Incorrect Locs", None),
                "Correct Exprs": data[id].get("Correct Exprs", None),
                "Incorrect Exprs": data[id].get("Incorrect Exprs", None),
                "Old Incorrect Exprs": data[id].get("Old Incorrect Exprs", None),
                "Repairs": data[id].get("Repairs", None),
                "Technique": data[id]["Technique"],
                "GM Score": data[id].get("GM Score", None),
                "Locs": data[id].get("Locs", None),
                "Count": data[id].get("Count",None),
                "Structure Mismatch": data[id]["Structure Mismatch"],
                "Repair": data[id]["Repair"],
                "Repair Correct": data[id]["Repair Correct"],
                "Parse Error": data[id]["Parse Error"],
                "Cost": data[id]["Cost"],
                "Second Output": data[id]["Second Output"],
                "Match": data[id]["Match"]
              }

if __name__=='__main__':
  home_dir = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
  print(f'The home directory is {home_dir}')
  dataset_dir = sys.argv[2] if len(sys.argv) > 2 else 'dataset'
  url = sys.argv[3] if len(sys.argv)>3 else ''
  print(f'Creating CLARA dataset in the following folder: {dataset_dir}')
  dt = datasets.load_dataset(pjoin(home_dir, dataset_dir), cache_dir=pjoin(home_dir, '.cache'), 
        num_proc=2, 
        url=url,
        data_dir=pjoin(home_dir, dataset_dir)
      )
  print("Dataset created:\n", dt)
  print("Saving dataset")
  dt.save_to_disk(pjoin(home_dir, dataset_dir))