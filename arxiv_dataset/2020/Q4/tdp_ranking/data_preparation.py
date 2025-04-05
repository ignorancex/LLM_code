import codecs

from data_structures import Node, get_padding_node, get_root_node, UNK_word, PAD_word

max_nodes_before = 10
max_nodes_after = 3
max_padded_candidate_length = max_nodes_before + max_nodes_after + 2  # Always include root and DCT as well (so +2)


def create_snt_edge_lists(doc):
    snt_list = []
    edge_list = []
    mode = None
    for line in doc:
        if line.endswith('LIST'):
            mode = line.strip().split(':')[-1]
        elif mode == 'SNT_LIST':
            snt_list.append(line.strip().split())
        elif mode == 'EDGE_LIST':
            edge_list.append(line.strip().split())

    return snt_list, edge_list


def create_node_list(snt_list, edge_list):
    node_list = []
    for i, edge in enumerate(edge_list):
        child, c_label, _, _ = edge  # parent and edge label aren't used here
        c_snt, c_start, c_end = [int(ch) for ch in child.split('_')]

        if not c_label.lower().startswith('timex'):
            prefix = 'EVENT-' if c_label.isupper() else 'Event-'
            c_label = prefix + c_label

        c_node = Node(c_snt, c_start, c_end, i,
                      get_word_index_in_doc(snt_list, c_snt, c_start),
                      get_word_index_in_doc(snt_list, c_snt, c_end),
                      ' '.join(snt_list[c_snt][c_start:c_end + 1]),
                      c_label)

        node_list.append(c_node)

    return node_list


def get_word_index_in_doc(snt_list, snt_index_in_doc, word_index_in_snt):
    index = 0
    for i, snt in enumerate(snt_list):
        if i < snt_index_in_doc:
            index += len(snt)
        else:
            break

    return index + word_index_in_snt


def check_example_contains_gold_parent(example):
    for tup in example:
        if tup[2] != 'NO_EDGE':
            return True

    return False


def make_one_doc_training_data(doc, vocab, pad_candidates):
    """
    return: training_example_list
    [[(p_node, c_node, 'NO_EDGE'), (p_node, c_node, 'before'), ...],
        [(...), (...), ...],
        ...]
    """

    doc = doc.strip().split('\n')

    # create snt_list, edge_list
    snt_list, edge_list = create_snt_edge_lists(doc)

    # add words to vocab
    for snt in snt_list:
        for word in snt:
            if word not in vocab:
                vocab[word] = vocab.get(word, 0) + 1

    # create node_list
    node_list = create_node_list(snt_list, edge_list)

    # create training example list 
    training_example_list = []
    root_node = get_root_node()
    padding_node = get_padding_node()

    for i, edge in enumerate(edge_list):
        _, _, parent, label = edge
        child_node = node_list[i]

        if pad_candidates:
            example = choose_candidates_padded(node_list, root_node, padding_node, child_node, parent, label)
        else:
            example = choose_candidates(node_list, root_node, child_node, parent, label)

        if check_example_contains_gold_parent(example):
            training_example_list.append(example)
        else:
            # Either child == parent in the annotation (can happen for Turker annotation) which is invalid
            # Or gold parent is not in the window if using pad_candidates=True
            doc_name = doc[0].split(':')[1]
            if pad_candidates:
                print('WARNING: Gold parent not included for edge {} in document {}'.format(edge, doc_name))
            else:
                raise ValueError('No gold parent for edge {} in document {}'.format(edge, doc_name))

    return [snt_list, training_example_list]


def choose_candidates(node_list, root_node, child, parent_ID, label):
    candidates = []

    # Always consider root
    candidates.append(get_candidate(child, root_node, parent_ID, label))

    for candidate_node in node_list:
        if candidate_node.ID == child.ID:
            continue
        elif candidate_node.snt_index_in_doc - child.snt_index_in_doc > 2:
            # Only consider from beginning of text to two sentences afterwards
            break
        else:
            candidates.append(get_candidate(child, candidate_node, parent_ID, label))

    return candidates


def choose_candidates_padded(node_list, root_node, padding_node, child, parent_ID, label):
    candidates = []

    # Always consider root
    candidates.append(get_candidate(child, root_node, parent_ID, label))

    for candidate_node in node_list:
        node_distance = candidate_node.node_index_in_doc - child.node_index_in_doc
        if not candidate_node.is_DCT and node_distance < -max_nodes_before:
            # Only consider DCT and a fixed number nodes before child
            continue
        elif candidate_node.ID == child.ID:
            # Child cannot be parent of itself
            continue
        elif node_distance > max_nodes_after:
            # Stop after a fixed number of nodes after child
            break
        else:
            candidates.append(get_candidate(child, candidate_node, parent_ID, label))

    if len(candidates) < max_padded_candidate_length:
        padding_length = max_padded_candidate_length - len(candidates)
        for i in range(padding_length):
            candidates.append(get_candidate(child, padding_node, parent_ID, label))

    return candidates


def get_candidate(child, candidate_node, parent_ID, label):
    if label:
        # Training on labels, so either add label or 'NO_EDGE'
        if candidate_node.ID == parent_ID:
            return candidate_node, child, label
        else:
            return candidate_node, child, 'NO_EDGE'
    else:
        # Predicting labels with model, so add None here
        return candidate_node, child, None


def make_training_data(train_file, pad_candidates=False):
    """ Given a file of multiple documents in ConLL-similar format,
    produce a list of training docs, each training doc is 
    (1) a list of sentences in that document; and 
    (2) a list of (parent_candidate, child_node, edge_label/no_edge) tuples 
    in that document; 
    and the vocabulary of this training data set.
    """

    data = codecs.open(train_file, 'r', 'utf-8').read()
    doc_list = data.strip().split('\n\nfilename')

    training_data = []
    count_vocab = {}

    for doc in doc_list:
        try:
            training_data.append(make_one_doc_training_data(doc, count_vocab, pad_candidates))
        except ValueError as e:
            print('WARNING: {}, skipping document'.format(e))
            pass

    index = 2
    vocab = {}
    for word in count_vocab:
        if count_vocab[word] > 0:
            vocab[word] = index
            index += 1

    vocab.update({PAD_word: 0})
    vocab.update({UNK_word: 1})

    return training_data, vocab


def merge_vocab(vocab1, vocab2):
    words = set(vocab1.keys()).union(vocab2.keys())
    words.remove(PAD_word)  # Add PAD manually later with index 0
    words.remove(UNK_word)  # Add UNK manually later with index 1

    vocab = {}
    for index, word in enumerate(words, start=2):
        vocab[word] = index

    vocab.update({PAD_word: 0})
    vocab.update({UNK_word: 1})
    return vocab


def make_one_doc_test_data(doc, pad_candidates):
    doc = doc.strip().split('\n')

    # create snt_list, edge_list
    snt_list, edge_list = create_snt_edge_lists(doc)

    # create node_list
    node_list = create_node_list(snt_list, edge_list)

    # create test instance list
    test_instance_list = []
    root_node = get_root_node()
    padding_node = get_padding_node()

    for c_node in node_list:
        if pad_candidates:
            instance = choose_candidates_padded(node_list, root_node, padding_node, c_node, None, None)
        else:
            instance = choose_candidates(node_list, root_node, c_node, None, None)

        test_instance_list.append(instance)

    return [snt_list, test_instance_list]


def make_test_data(test_file, pad_candidates=False):
    data = codecs.open(test_file, 'r', 'utf-8').read()
    doc_list = data.strip().split('\n\nfilename')

    test_data = []

    for doc in doc_list:
        test_data.append(make_one_doc_test_data(doc, pad_candidates))

    return test_data
