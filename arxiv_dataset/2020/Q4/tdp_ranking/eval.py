import argparse
import codecs
from collections import Counter

# In eval, we don't have access to the node, just the node ID, so we can't use is_DCT on the node.
# In the training data / TimeML corpus it's always 0_3_3
DCT_node_id = '0_3_3'


def is_DCT(node_id):
    return node_id == DCT_node_id


def get_arg_parser():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--gold_file", help="gold file")
    arg_parser.add_argument("--parsed_file", help="parsed file")
    arg_parser.add_argument("--labeled", help="evaluate with edge labels",
                            action="store_true", default=False)

    return arg_parser


def read_in_tuples(filename):
    lines = codecs.open(filename, 'r', 'utf-8').readlines()

    return lines_to_tuples(lines)


def lines_to_tuples(lines):
    edge_tuples = []
    mode = None
    for line in lines:
        line = line.strip()
        if line == '':
            continue
        elif line.endswith('LIST'):
            mode = line.strip().split(':')[-1]
            if mode == 'SNT_LIST':
                edge_tuples.append([])
        elif mode == 'EDGE_LIST':
            child, child_label, parent, link_label = line.strip().split()
            edge_tuples[-1].append((child, child_label, parent, link_label))
    return edge_tuples


def get_labeled_tuple_set(tups):
    # Omit child_label
    tup_set = set([(tup[0], tup[2], tup[3]) for tup in tups])
    return tup_set


def get_unlabeled_tuple_set(tups):
    # Omit child label and link label
    tup_set = set([(tup[0], tup[2]) for tup in tups])
    return tup_set


def evaluate(gold_tuples, parsed_tuples, get_tuple_set):
    counts = []
    scores = []

    timex_counts = []
    DCT_counts = []
    timex_child_counts = []
    event_child_counts = []
    label_stats = []

    for i, (doc_gold_tuples, doc_parsed_tuples) in enumerate(zip(gold_tuples, parsed_tuples)):
        gold_tup_set = get_tuple_set(doc_gold_tuples)
        parsed_tup_set = get_tuple_set(doc_parsed_tuples)

        true_positive = len(gold_tup_set.intersection(parsed_tup_set))
        false_positive = len(parsed_tup_set.difference(gold_tup_set))
        false_negative = len(gold_tup_set.difference(parsed_tup_set))

        print('test doc {}: true_p = {}, false_p = {}, false_n = {}'.format(
            i, true_positive, false_positive, false_negative))
        p = true_positive / (true_positive + false_positive)
        r = true_positive / (true_positive + false_negative)
        f = 2 * p * r / (p + r) if p + r != 0 else 0

        counts.append((true_positive, false_positive, false_negative))
        scores.append((p, r, f))

        # Look in particular at TIMEX
        timex_counts.append(count_timex(doc_gold_tuples, doc_parsed_tuples, get_tuple_set))

        # Children of DCT
        DCT_counts.append(count_DCT_children(doc_gold_tuples, doc_parsed_tuples, get_tuple_set))

        # Children of other TIMEXs
        timex_child_counts.append(count_timex_children(doc_gold_tuples, doc_parsed_tuples, get_tuple_set))

        # Children of events
        event_child_counts.append(count_event_children(doc_gold_tuples, doc_parsed_tuples, get_tuple_set))

        # Label stats - enable for additional statistics about which labels were used (also enable below)
        # label_stats.append(get_label_stats(doc_gold_tuples, doc_parsed_tuples))

    # macro average
    p = sum([score[0] for score in scores]) / len(scores)
    r = sum([score[1] for score in scores]) / len(scores)
    f = sum([score[2] for score in scores]) / len(scores)

    print('macro average: p = {:.3f}, r = {:.3f}, f = {:.3f}'.format(p, r, f))
    # print('macro average: f = {:.3f}'.format(f), end='; ')

    # micro average - n.B. this is the same as accuracy
    true_p = sum([count[0] for count in counts])
    false_p = sum([count[1] for count in counts])
    false_n = sum([count[2] for count in counts])

    p = true_p / (true_p + false_p)
    r = true_p / (true_p + false_n)
    f = 2 * p * r / (p + r) if p + r != 0 else 0

    print('micro average: p = {:.3f}, r = {:.3f}, f = {:.3f}'.format(p, r, f))
    # print('micro average: f = {:.3f}\n'.format(f))

    print_timex(timex_counts)
    print_DCT_children(DCT_counts)
    print_timex_children(timex_child_counts)
    print_event_children(event_child_counts)
    # print_label_stats(label_stats)  # To enable see above
    print('')


def count_timex(doc_gold_tuples, doc_parsed_tuples, get_tuple_set):
    # tup[1] gets the child label
    timex_indices = [i for i, tup in enumerate(doc_gold_tuples) if 'TIMEX' in tup[1]]
    gold_timexs = get_tuple_set(doc_gold_tuples[i] for i in timex_indices)
    parsed_timexs = get_tuple_set(doc_parsed_tuples[i] for i in timex_indices)
    correct_timex_count = len(gold_timexs.intersection(parsed_timexs))
    return correct_timex_count, len(gold_timexs)


def print_timex(timex_counts):
    total_correct_timex = sum(timex_count[0] for timex_count in timex_counts)
    total_timex = sum(timex_count[1] for timex_count in timex_counts)
    print('TIMEX correct: {:.3f}'.format(total_correct_timex / total_timex))


def count_correct_children(selected_nodes, doc_gold_tuples, doc_parsed_tuples, get_tuple_set):
    selected_child_indices = [i for i, tup in enumerate(doc_gold_tuples) if tup[2] in selected_nodes]
    gold_selected_children = get_tuple_set(doc_gold_tuples[i] for i in selected_child_indices)
    parsed_selected_children = get_tuple_set(doc_parsed_tuples[i] for i in selected_child_indices)
    correct_selected_child_count = len(gold_selected_children.intersection(parsed_selected_children))
    return correct_selected_child_count, len(gold_selected_children)


def percent_correct_children(selected_child_counts):
    total_correct_selected = sum(selected_count[0] for selected_count in selected_child_counts)
    total_selected = sum(selected_count[1] for selected_count in selected_child_counts)
    return '{:.3f}'.format(total_correct_selected / total_selected) if total_selected > 0 else 'n/a'


def count_DCT_children(doc_gold_tuples, doc_parsed_tuples, get_tuple_set):
    DCT_nodes = [DCT_node_id]
    return count_correct_children(DCT_nodes, doc_gold_tuples, doc_parsed_tuples, get_tuple_set)


def print_DCT_children(DCT_counts):
    print('Children of DCT correct: {}'.format(percent_correct_children(DCT_counts)))


def count_timex_children(doc_gold_tuples, doc_parsed_tuples, get_tuple_set):
    timex_nodes = [tup[0] for tup in doc_gold_tuples if 'TIMEX' in tup[1] and not is_DCT(tup[0])]
    return count_correct_children(timex_nodes, doc_gold_tuples, doc_parsed_tuples, get_tuple_set)


def print_timex_children(timex_child_counts):
    print('Children of other TIMEX correct: {}'.format(percent_correct_children(timex_child_counts)))


def count_event_children(doc_gold_tuples, doc_parsed_tuples, get_tuple_set):
    event_nodes = [tup[0] for tup in doc_gold_tuples if 'TIMEX' not in tup[1]]
    return count_correct_children(event_nodes, doc_gold_tuples, doc_parsed_tuples, get_tuple_set)


def print_event_children(event_child_counts):
    print('Children of events correct: {}'.format(percent_correct_children(event_child_counts)))


def get_label_stats(doc_gold_tuples, doc_parsed_tuples):
    gold_label_counts = Counter(edge_label for child, child_label, parent, edge_label in doc_gold_tuples)
    parsed_label_counts = Counter(edge_label for child, child_label, parent, edge_label in doc_parsed_tuples)

    return gold_label_counts, parsed_label_counts


def print_label_stats(label_stats):
    total_gold_label_counts = Counter()
    total_parsed_label_counts = Counter()

    for gold_label_counter, parsed_label_counter in label_stats:
        total_gold_label_counts.update(gold_label_counter)
        total_parsed_label_counts.update(parsed_label_counter)
    total_gold_labels = sum(total_gold_label_counts.values())
    total_parsed_labels = sum(total_parsed_label_counts.values())

    print('Gold label stats:')
    for label, count in total_gold_label_counts.items():
        print('{}: {:.2f}%'.format(label, count / total_gold_labels * 100))
    print('Parsed label stats:')
    for label, count in total_parsed_label_counts.items():
        print('{}: {:.2f}%'.format(label, count / total_parsed_labels * 100))


if __name__ == '__main__':
    arg_parser = get_arg_parser()
    args = arg_parser.parse_args()

    gold_tuples = read_in_tuples(args.gold_file)
    parsed_tuples = read_in_tuples(args.parsed_file)

    if args.labeled:
        evaluate(gold_tuples, parsed_tuples, get_labeled_tuple_set)
    else:
        evaluate(gold_tuples, parsed_tuples, get_unlabeled_tuple_set)
