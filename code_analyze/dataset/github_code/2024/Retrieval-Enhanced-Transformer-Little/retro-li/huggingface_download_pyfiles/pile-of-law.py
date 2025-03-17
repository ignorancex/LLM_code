"""Pile of Law"""


import gzip
import json

import datasets
try:
    import lzma as xz
except ImportError:
    import pylzma as xz


datasets.logging.set_verbosity_info()
logger = datasets.logging.get_logger(__name__)


_DESCRIPTION = """
We curate a large corpus of legal and administrative data. The utility of this data is twofold: (1) to aggregate legal and administrative data sources that demonstrate different norms and legal standards for data filtering; (2) to collect a dataset that can be used in the future for pretraining legal-domain language models, a key direction in access-to-justice initiatives.
"""

_CITATION = """
@misc{hendersonkrass2022pileoflaw,
  url = {https://arxiv.org/abs/2207.00220},
  author = {Henderson, Peter and Krass, Mark S. and Zheng, Lucia and Guha, Neel and Manning, Christopher D. and Jurafsky, Dan and Ho, Daniel E.},
  title = {Pile of Law: Learning Responsible Data Filtering from the Law and a 256GB Open-Source Legal Dataset},
  publisher = {arXiv},
  year = {2022}
}
"""

_URL = "https://huggingface.co/datasets/pile-of-law/pile-of-law"


_DATA_URL = {
    "atticus_contracts" : {
        "train" : [
            "https://huggingface.co/datasets/pile-of-law/pile-of-law/resolve/main/data/train.atticus_contracts.0.jsonl.xz",
            "https://huggingface.co/datasets/pile-of-law/pile-of-law/resolve/main/data/train.atticus_contracts.1.jsonl.xz",
            "https://huggingface.co/datasets/pile-of-law/pile-of-law/resolve/main/data/train.atticus_contracts.2.jsonl.xz",
            "https://huggingface.co/datasets/pile-of-law/pile-of-law/resolve/main/data/train.atticus_contracts.3.jsonl.xz",
            "https://huggingface.co/datasets/pile-of-law/pile-of-law/resolve/main/data/train.atticus_contracts.4.jsonl.xz"            
        ],
        "validation" : [
            "https://huggingface.co/datasets/pile-of-law/pile-of-law/resolve/main/data/validation.atticus_contracts.0.jsonl.xz",
            "https://huggingface.co/datasets/pile-of-law/pile-of-law/resolve/main/data/validation.atticus_contracts.1.jsonl.xz"
        ]
    },
    "founding_docs" : {
        "train" : ["https://huggingface.co/datasets/pile-of-law/pile-of-law/resolve/main/data/train.founding_docs.jsonl.xz"],
        "validation" : ["https://huggingface.co/datasets/pile-of-law/pile-of-law/resolve/main/data/validation.founding_docs.jsonl.xz"]        
    }
}

_VARIANTS = ["all"] + list(_DATA_URL.keys())


class PileOfLaw(datasets.GeneratorBasedBuilder):
    """TODO"""

    BUILDER_CONFIGS = [datasets.BuilderConfig(name) for name in _VARIANTS]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "created_timestamp": datasets.Value("string"),
                    "downloaded_timestamp": datasets.Value("string"),
                    "url": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=_URL,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_urls = {}
        if self.config.name == "all":
            data_sources = list(_DATA_URL.keys())
        else:
            data_sources = [self.config.name]
        for split in ["train", "validation"]:
            data_urls[split] = []
            for source in data_sources:
                for chunk in _DATA_URL[source][split]:
                    data_urls[split].append(chunk)

        train_downloaded_files = dl_manager.download(data_urls["train"])
        validation_downloaded_files = dl_manager.download(data_urls["validation"])
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepaths": train_downloaded_files}),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={"filepaths": validation_downloaded_files}
            ),
        ]

    def _generate_examples(self, filepaths):
        """This function returns the examples in the raw (text) form by iterating on all the files."""
        id_ = 0
        for filepath in filepaths:
            logger.info("Generating examples from = %s", filepath)
            try:
                with xz.open(open(filepath, "rb"), "rt", encoding="utf-8") as f:
                    for line in f:
                        if line:
                            example = json.loads(line)
                            if example is not None and isinstance(example, dict):
                                yield id_, {
                                    "text": example.get("text", ""),
                                    "created_timestamp": example.get("created_timestamp", ""),
                                    "downloaded_timestamp": example.get("downloaded_timestamp", ""),
                                    "url": example.get("url", "")
                                }
                                id_ += 1
            except:
                print("Error reading file:", filepath)
