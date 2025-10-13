import os
from datasets import load_dataset
from .document import SynthBioDoc

DATA_DIR = os.getenv("DATASET_DIR")

class QASynthBio(SynthBioDoc):
    def __init__(self, **configs):
        self.sentence_perplex = False
        self.set_variables(configs)
        self.key_title = "question"
        self.key_content = "answer"
        self.shuffle = False

    def make_instructions(self, examples):
        questions = examples[self.key_title]
        new_queries = []
        for question in questions:
            if self.inst_mode:
                ## This is used in llama.
                query = "[INST] " + question + " [/INST]"
                new_queries.append(query)
            else:
                new_queries.append(question)

        return new_queries

    def shuffle_items(self, sentences):
        return sentences

 