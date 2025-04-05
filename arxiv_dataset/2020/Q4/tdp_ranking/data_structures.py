UNK_word = '<UNK>'
UNK_label = '<UNK>'
PAD_word = 'padding'
PAD_label = '<PAD>'
ROOT_word = 'root'
ROOT_label = 'ROOT'
DCT_word = 'DCT'


class Node:
    def __init__(self, snt_index_in_doc=-1, start_word_index_in_snt=-1, end_word_index_in_snt=-1, node_index_in_doc=-1,
                 start_word_index_in_doc=-1, end_word_index_in_doc=-1, words=ROOT_word, label=ROOT_label):
        self.snt_index_in_doc = snt_index_in_doc
        self.start_word_index_in_snt = start_word_index_in_snt
        self.end_word_index_in_snt = end_word_index_in_snt
        self.node_index_in_doc = node_index_in_doc
        self.start_word_index_in_doc = start_word_index_in_doc
        self.end_word_index_in_doc = end_word_index_in_doc

        self.words = words
        self.full_label = label  # full label
        self.TE_label = label.split('-')[0].upper()  # timex/event label

        self.ID = '_'.join([str(snt_index_in_doc),
                            str(start_word_index_in_snt), str(end_word_index_in_snt)])

        self.is_DCT = self.words == DCT_word

    def __str__(self):
        return '\t'.join([self.ID, self.words, self.full_label])

    def __eq__(self, other):
        if isinstance(other, Node):
            # Note that this does not check whether the two nodes are from the same document
            return self.ID == other.ID
        return False

    def __hash__(self):
        return hash(self.ID)


def get_root_node():
    return Node()


def get_padding_node():
    return Node(-2, -2, -2, -2, -2, -2, PAD_word, PAD_label)


def is_padding_node(node):
    return node.node_index_in_doc == -2


def is_root_node(node):
    return node.node_index_in_doc == -1


LABEL_VOCAB_FULL = {
    'ROOT': 0,
    'Timex-AbsoluteConcrete': 1,
    'Timex-RelativeVague': 2,
    'Timex-RelativeConcrete': 3,
    'Event-Event': 4,
    'Event-State': 5,
    'Event-ModalizedEvent': 6,
    'Event-OngoingEvent': 7,
    'Event-GenericHabitual': 8,
    'Event-GenericState': 9,
    'Event-Habitual': 10,
    'Event-CompletedEvent': 11,
    UNK_label: 12,
    PAD_label: 13,
}

LABEL_VOCAB_TIMEML = {
    'ROOT': 0,
    'TIMEX': 1,
    'EVENT-I_ACTION': 2,
    'EVENT-OCCURRENCE': 3,
    'EVENT-REPORTING': 4,
    'EVENT-PERCEPTION': 5,
    'EVENT-ASPECTUAL': 6,
    'EVENT-I_STATE': 7,
    'EVENT-STATE': 8,
    UNK_label: 9,
    PAD_label: 10,
}

LABEL_VOCAB_TIMEX_EVENT = {
    'ROOT': 0,
    'TIMEX': 1,
    'EVENT': 2,
    UNK_label: 3,
    PAD_label: 4,
}

EDGE_LABEL_LIST_FULL = [
    'ROOT',
    'DCT',
    'PRESENT_REF',
    'PAST_REF',
    'FUTURE_REF',
    'ATEMPORAL',
    'before',
    'after',
    'overlap',
    'includes',
    'Depend-on',
]

EDGE_LABEL_LIST_TIMEML = [
    'before',
    'after',
    'overlap',
    'includes',
    'Depend-on',
]

EDGE_LABEL_LIST_UNLABELED = [
    'EDGE'
]
