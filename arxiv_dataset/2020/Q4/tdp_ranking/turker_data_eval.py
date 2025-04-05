"""
A collection of useful functions for comparing Yuchen's expert-annotated Timebank-Dense data to the larger,
Turker-annotated Timebank data, as well as functions for calculating metrics such as max node distance on the gold data.
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from data_preparation import create_snt_edge_lists, make_training_data
from eval import evaluate, get_labeled_tuple_set, lines_to_tuples, get_unlabeled_tuple_set

YUCHEN_PATH = 'data/timebank-dense.yuchen-tdt.all'
TURKER_PATH = 'data/timebank.turker-tdt.all'


def determine_overlap(yuchen_filenames, turker_filenames):
    overlap = set(turker_filenames).intersection(yuchen_filenames)
    distinct_count = len(turker_filenames) + len(yuchen_filenames) - len(overlap)

    return overlap, distinct_count


def get_filenames(data_path):
    data = open(data_path, 'r', encoding='utf-8').read()
    doc_list = data.strip().split('filename:')[1:]
    filenames = [get_filename(doc) for doc in doc_list]
    return filenames


def get_filename(doc):
    first_line = doc.split('\n')[0]
    return first_line.split(':')[0]


def get_lines_for_filenames(data, ordered_filenames):
    filename_str = 'filename:'

    doc_list = data.strip().split(filename_str)
    data_filenames = [doc.partition(':')[0] for doc in doc_list]

    matching_docs = []
    for filename in ordered_filenames:
        for i in range(len(data_filenames)):
            if data_filenames[i] == filename:
                matching_docs.append(doc_list[i])

    joined_docs = filename_str + filename_str.join(matching_docs)
    return joined_docs.split('\n')


def lines_to_file_tuples(lines):
    edge_tuples = dict()
    mode = None
    filename = None

    for line in lines:
        line = line.strip()
        if line.startswith('filename'):
            filename = line.split(':')[1]

        if line == '':
            continue
        elif line.endswith('LIST'):
            mode = line.strip().split(':')[-1]
            if mode == 'SNT_LIST':
                edge_tuples[filename] = []
        elif mode == 'EDGE_LIST':
            child, child_label, parent, link_label = line.strip().split()
            edge_tuples[filename].append((child, child_label, parent, link_label))
    return edge_tuples


def get_words_for_node(file_lines, node):
    snt, start, end = [int(ch) for ch in node.split('_')]

    snt_list, _ = create_snt_edge_lists(file_lines)
    return ' '.join(snt_list[snt][start:end + 1])


def print_overlap(yuchen_filenames, turker_filenames):
    overlap, distinct_count = determine_overlap(yuchen_filenames, turker_filenames)

    print('Yuchen annotated:', YUCHEN_PATH)
    print('Turker annotated:', TURKER_PATH)
    print('Overlap: {} / {}'.format(len(overlap), distinct_count), end='\n\n')
    # print('\n'.join(overlap), end='\n\n')
    print_doc_numbers(yuchen_filenames, turker_filenames)


def print_doc_numbers(yuchen_filenames, turker_filenames):
    for i, y_filename in enumerate(yuchen_filenames):
        turker_i = turker_filenames.index(y_filename)
        if turker_i != -1:
            print('Yuchen {}, Turker {}, filename {}'.format(i, turker_i, y_filename))
    print()


def compare_node_count(yuchen_filenames):
    yuchen_tuples = lines_to_file_tuples(open(YUCHEN_PATH, 'r', encoding='utf-8').readlines())
    turker_lines = get_lines_for_filenames(open(TURKER_PATH, 'r', encoding='utf-8').read(), yuchen_filenames)
    turker_tuples = lines_to_file_tuples(turker_lines)

    differing_files = set()

    for filename in yuchen_tuples.keys():
        yuchen_edge_count = len(yuchen_tuples[filename])
        turker_edge_count = len(turker_tuples[filename])
        edges_diff = turker_edge_count - yuchen_edge_count
        if edges_diff != 0:
            yuchen_timex = {child for child, child_label, parent, link_label in yuchen_tuples[filename] if
                            child_label == 'TIMEX'}
            turker_timex = {child for child, child_label, parent, link_label in turker_tuples[filename] if
                            child_label == 'TIMEX'}
            yuchen_events = {child for child, child_label, parent, link_label in yuchen_tuples[filename] if
                             child_label != 'TIMEX'}
            turker_events = {child for child, child_label, parent, link_label in turker_tuples[filename] if
                             child_label != 'TIMEX'}

            timex_diff = yuchen_timex.symmetric_difference(turker_timex)
            events_diff = yuchen_events.symmetric_difference(turker_events)

            if len(timex_diff) > 0 or len(events_diff) > 0:
                print('{}:'.format(filename))
                file_lines = get_lines_for_filenames('\n'.join(turker_lines), [filename])

                if len(timex_diff) > 0:
                    print('Yuchen TIMEX: {}\nTurker TIMEX: {}'
                          .format(len(yuchen_timex), len(turker_timex)))

                    timex_as_words = [get_words_for_node(file_lines, node) for node in
                                      timex_diff]
                    print('Differing TIMEX: ', timex_as_words)

                if len(events_diff) > 0:
                    print('Yuchen events: {}\nTurker events: {}'
                          .format(len(yuchen_events), len(turker_events)))

                    events_diff = yuchen_events.symmetric_difference(turker_events)
                    events_as_words = [get_words_for_node(file_lines, node) for node in
                                       events_diff]
                    print('Differing events: ', events_as_words)

            differing_files.add(filename)

    print('Total differing files: {} / {}'.format(len(differing_files), len(yuchen_tuples)), end='\n\n')

    return differing_files


def evaluate_turker_against_yuchen(matching_filenames):
    yuchen_tuples = lines_to_tuples(get_lines_for_filenames(open(YUCHEN_PATH, 'r', encoding='utf-8').read(),
                                                            matching_filenames))
    turker_tuples = lines_to_tuples(get_lines_for_filenames(open(TURKER_PATH, 'r', encoding='utf-8').read(),
                                                            matching_filenames))

    print('Treating Yuchen\'s data as gold standard, evaluating only on files with matching nodes...')
    print('Labeled:')
    evaluate(yuchen_tuples, turker_tuples, get_labeled_tuple_set)
    print('\nUnlabeled:')
    evaluate(yuchen_tuples, turker_tuples, get_unlabeled_tuple_set)


def max_node_distance():
    yuchen_node_distances, yuchen_DCT_count, yuchen_root_count = get_node_distances(YUCHEN_PATH)
    turker_node_distances, turker_DCT_count, turker_root_count = get_node_distances(TURKER_PATH)

    print('Yuchen: {} children of DCT, {} children of root, '
          'max forward node distance: {}, max backward node distance: {}'
          .format(yuchen_DCT_count, yuchen_root_count, max(yuchen_node_distances), abs(min(yuchen_node_distances))))
    print('Turker: {} children of DCT, {} children of root, '
          'max forward node distance: {}, max backward node distance: {}'
          .format(turker_DCT_count, turker_root_count, max(turker_node_distances), abs(min(turker_node_distances))))

    fig, axes = plt.subplots(1, 2, tight_layout=True)
    bin_width = 1
    axes[0].hist(yuchen_node_distances, bins=get_bins(yuchen_node_distances, bin_width))
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].set_xlabel('Yuchen node distance (except root, DCT)')
    axes[0].set_ylabel('Count')
    axes[1].hist(turker_node_distances, bins=get_bins(turker_node_distances, bin_width))
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].set_xlabel('Turker node distance (except root, DCT)')
    plt.show()


def get_bins(node_distances, bin_width):
    return range(min(node_distances), max(node_distances) + bin_width, bin_width)


def get_node_distances(file_path):
    node_distances = []
    DCT_count = 0
    root_count = 0

    data, vocab = make_training_data(file_path)
    for document in data:
        sentence_list, child_parent_candidates = document
        for candidates_for_child in child_parent_candidates:
            for candidate in candidates_for_child:
                parent, child, label = candidate
                if label != 'NO_EDGE':
                    if parent.ID == '-1_-1_-1':
                        root_count += 1
                    elif parent.is_DCT:
                        DCT_count += 1
                    else:
                        # Parent is not DCT or root
                        node_distance = parent.node_index_in_doc - child.node_index_in_doc
                        node_distances.append(node_distance)
    return node_distances, DCT_count, root_count


if __name__ == '__main__':
    yuchen_filenames = get_filenames(YUCHEN_PATH)
    turker_filenames = get_filenames(TURKER_PATH)

    print_overlap(yuchen_filenames, turker_filenames)

    differing_files = compare_node_count(yuchen_filenames)

    matching_files = [filename for filename in yuchen_filenames if filename not in differing_files]
    evaluate_turker_against_yuchen(matching_files)

    # max_node_distance()
