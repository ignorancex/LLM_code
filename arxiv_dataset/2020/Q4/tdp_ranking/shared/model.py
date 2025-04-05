def get_gold_candidate_one_hot(candidates_for_child, labeled, edge_label_set):
    """
    Of all the candidate parents for the child, only one of them has a label which is not 'NO_EDGE'.

    The model outputs a score for each candidate parent and edge label, indexed by parent and edge label.
    Assuming that is flattened down to list all the labels for parent 1, then all the labels for parent 2, etc.,
    return the index of the one parent and label combination which is correct i.e. not 'NO_EDGE'.
    """

    out_list = []
    for candidate in candidates_for_child:
        _, _, gold_label = candidate
        if labeled:
            for label in edge_label_set:
                if label == gold_label:
                    out_list.append(1)
                else:
                    out_list.append(0)
        elif not labeled:
            if gold_label != 'NO_EDGE':
                out_list.append(1)
            else:
                out_list.append(0)

    return out_list
