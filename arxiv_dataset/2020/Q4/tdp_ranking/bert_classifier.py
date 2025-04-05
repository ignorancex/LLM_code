import json

from bert_as_classifier.model import data_to_inputs_and_gold, get_model_inputs
from bert_as_classifier.shared.classifier import BertBaseClassifier
from shared.bert_layer import create_tokenizer_from_hub_module


class BertClassifier(BertBaseClassifier):

    def data_to_inputs_and_gold(self, gold_data, silver_data, dev_data, labeled):
        bert_tokenizer = create_tokenizer_from_hub_module()
        gold_inputs_labels = data_to_inputs_and_gold(gold_data, bert_tokenizer, labeled, self.TE_label_vocab,
                                                     self.TE_label_set, self.edge_label_set, self.max_sequence_length,
                                                     self.disable_handcrafted_features)
        silver_inputs_labels = None
        if silver_data:
            silver_inputs_labels = data_to_inputs_and_gold(silver_data, bert_tokenizer, labeled, self.TE_label_vocab,
                                                           self.TE_label_set, self.edge_label_set,
                                                           self.max_sequence_length, self.disable_handcrafted_features)
        dev_inputs_labels = data_to_inputs_and_gold(dev_data, bert_tokenizer, labeled, self.TE_label_vocab,
                                                    self.TE_label_set, self.edge_label_set, self.max_sequence_length,
                                                    self.disable_handcrafted_features)

        return gold_inputs_labels, silver_inputs_labels, dev_inputs_labels

    def data_to_model_inputs(self, sentence_list, child_parent_candidates):
        bert_tokenizer = create_tokenizer_from_hub_module()
        model_inputs = get_model_inputs(bert_tokenizer, sentence_list, child_parent_candidates,
                                        self.TE_label_set, self.TE_label_vocab, self.max_sequence_length,
                                        self.disable_handcrafted_features)

        return model_inputs

    @classmethod
    def load(cls, model_file):
        with open(model_file + '.config', 'r', encoding='utf-8') as config_file:
            config = json.load(config_file)

            classifier = cls(config['TE_label_set'], config['edge_label_set'], config['max_sequence_length'],
                             config['max_candidate_count'], config['disable_handcrafted_features'])

            classifier.load_model(model_file)
            return classifier
