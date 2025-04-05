import argparse
import codecs
import os

from baseline_classifier import BaselineClassifier
from data_preparation import make_test_data
from data_structures import EDGE_LABEL_LIST_TIMEML


def get_arg_parser():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--test_file", help="test data to be parsed")

    arg_parser.add_argument("--model_file", help="the model to use")
    arg_parser.add_argument("--parsed_file", help="where to output the parsed results")

    arg_parser.add_argument("--TE_label_set",
                            help="which timex/event label set to use when saving the parsed data: "
                                 "none, timex_event, time_ml, or full",
                            choices=['none', 'timex_event', 'time_ml', 'full'], default='full')

    arg_parser.add_argument("--classifier", help="which classifier to use",
                            choices=["bilstm", "glove_bilstm", "bert_bilstm", "bert_as_classifier",
                                     "bert_as_classifier_alt", "baseline"])
    arg_parser.add_argument("--labeled", help="parse with edge labels",
                            action="store_true", default=False)

    arg_parser.add_argument("--default_label",
                            help="default edge label to use for baseline parser",
                            choices=EDGE_LABEL_LIST_TIMEML)

    return arg_parser


def output_parse(edge_list, snt_list, output_file):
    with codecs.open(output_file, 'a', 'utf-8') as f:
        text = '\n'.join([' '.join(snt) for snt in snt_list])
        edge_text = '\n'.join(edge_list)
        f.write(
            'SNT_LIST\n' + text + '\n' + 'EDGE_LIST\n' + edge_text + '\n\n')


def is_cyclic(predicted_tuple, offspring_dict):
    candidate = predicted_tuple[0].ID
    child = predicted_tuple[1].ID

    if child not in offspring_dict:
        return False
    elif candidate not in offspring_dict[child]:
        return False
    else:
        return True


def update_dict_w_new_edge(offspring_dict, parent_dict, predicted_tuple):
    parent = predicted_tuple[0].ID
    child = predicted_tuple[1].ID

    parent_dict[child] = parent
    offspring_dict[parent] = offspring_dict.get(parent, {})
    offspring_dict[parent].update({child: None})
    offspring_dict[parent].update(offspring_dict.get(child, {}))

    while parent in parent_dict:
        parent = parent_dict[parent]
        offspring_dict[parent] = offspring_dict.get(parent, {})
        offspring_dict[parent].update({child: None})
        offspring_dict[parent].update(offspring_dict.get(child, {}))


def decode(test_data, classifier, classifier_name, output_file, labeled, TE_label_set):
    offspring_dict = {}
    parent_dict = {}
    for i, (snt_list, test_instance_list) in enumerate(test_data):
        # print('Parsing doc {} ...'.format(i))

        if classifier_name in ['bilstm', 'bert_bilstm', 'bert_as_classifier', 'bert_as_classifier_alt']:
            scores_list = classifier.predict(snt_list, test_instance_list, labeled)
        else:
            scores_list = []
            for instance in test_instance_list:
                scores_list.append(classifier.predict(snt_list, test_instance_list, instance, labeled))

        edge_list = []
        cyclic_count = 0
        for scores in scores_list:
            # Make sure the new edge doesn't add a cycle in the final tree
            # Scores are sorted descending so choose highest one that works
            j = 0
            while j < len(scores):
                candidate_tuple = scores[j][0]
                if is_cyclic(candidate_tuple, offspring_dict):
                    cyclic_count += 1
                else:
                    break
                j += 1

            assert j < len(scores), "No acyclic edges!!!"

            predicted_tuple = scores[j][0]

            # Update offspring_dict and parent_dict to include the newly added edge
            update_dict_w_new_edge(
                offspring_dict, parent_dict, predicted_tuple)

            if TE_label_set == 'timex_event':
                label = predicted_tuple[1].TE_label
            elif TE_label_set in ['full', 'time_ml']:
                label = predicted_tuple[1].full_label
            else:
                label = 'LABEL'

            edge = '\t'.join([predicted_tuple[1].ID, label, predicted_tuple[0].ID, predicted_tuple[2]])
            edge_list.append(edge)

        if cyclic_count > 0:
            print('Skipped {} cyclic edge prediction(s) in document {}'.format(cyclic_count, i))
        output_parse(edge_list, snt_list, output_file)


if __name__ == '__main__':
    arg_parser = get_arg_parser()
    args = arg_parser.parse_args()

    try:
        os.remove(args.parsed_file)
    except OSError:
        pass

    # Make test data
    test_data = make_test_data(args.test_file, pad_candidates=(args.classifier != 'baseline'))

    # Initialize classifier
    if args.classifier == 'baseline':
        default_label = args.default_label
        classifier = BaselineClassifier(args.default_label)

    elif args.classifier == 'bilstm':
        from bilstm_classifier import BilstmClassifier

        classifier = BilstmClassifier.load(args.model_file)

    elif args.classifier == 'glove_bilstm':
        from glove_bilstm_classifier import GloVeBilstmClassifier

        classifier = GloVeBilstmClassifier.load(args.model_file)

    elif args.classifier == 'bert_bilstm':
        from bert_bilstm_classifier import BertBilstmClassifier

        classifier = BertBilstmClassifier.load(args.model_file)

    elif args.classifier == 'bert_as_classifier':
        from bert_classifier import BertClassifier

        classifier = BertClassifier.load(args.model_file)

    elif args.classifier == 'bert_as_classifier_alt':
        from bert_alt_classifier import BertAltClassifier

        classifier = BertAltClassifier.load(args.model_file)

    else:
        raise ValueError('Invalid classifier argument')

    # Parse
    decode(test_data, classifier, args.classifier, args.parsed_file, args.labeled, args.TE_label_set)
