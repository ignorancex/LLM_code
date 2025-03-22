import os
from transformers.data.processors.utils import DataProcessor,InputExample
from cartography.classification.multiple_choice_utils import MCInputExample
import json
import pandas as pd


class MEDQAProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self.read_data(os.path.join(data_dir, "train.json"))
        )

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self.read_data(os.path.join(data_dir, "dev.json"))
        )

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self.read_data(os.path.join(data_dir, "test.json"))
        )

    def get_examples(self, data_file, set_type):
        return self._create_examples(self.read_data(data_file))

    def get_labels(self):
        """See base class."""
        return ['1', '2', '3', '4']

    def _build_example_from_named_fields(self, guid, question, options, correct_option):
        context = question
        question_tokens = question.split(" ")
        conj = question_tokens[-1] if len(question_tokens) > 0 else "_"


        option1 =  options[0]
        option2 =  options[1]
        option3 = options[2]
        option4 =  options[3]

        mc_example = MCInputExample(
            example_id=int(guid),
            contexts=[context, context,context,context],
            question=conj,
            endings=[option1, option2, option3, option4],
            label=correct_option,
        )

        return mc_example

    def _create_examples(self, records):
        examples = []
        for index, record in records.iterrows():
            question = record["question"]
            guid = record['id']
            options = [record["opa"], record["opb"], record["opc"], record["opd"]]
            correct_option = record["cop"]

            #mc_example = self._build_example_from_named_fields(
            #    index, question, options, correct_option#record["id"], question, explanation, options, correct_option
            #)
            options_str = ", ".join(f"{index}: {value}" for index, value in enumerate(options))

            mc_example = InputExample(
                guid=index, text_a=question, text_b=options_str, label=str(correct_option))

            examples.append(mc_example)

        return examples

    def read_data(self,path):
        with open(path, 'r') as file:
            data = [json.loads(line) for line in file]
        df = pd.DataFrame(data)
        return df