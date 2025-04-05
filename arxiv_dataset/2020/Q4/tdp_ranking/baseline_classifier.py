from data_structures import get_root_node


class BaselineClassifier:
    def __init__(self, default_label):
        self.default_label = default_label
        self.timex_edge_label = 'Depend-on'
        self.unlabeled_label = 'EDGE'

    def predict(self, snt_list, example_list, example, labeled):
        timex_edge_label = self.timex_edge_label if labeled else self.unlabeled_label
        default_label = self.default_label if labeled else self.unlabeled_label

        child = example[0][1]  # All tuples share same child, so grab it from first tuple

        if child.TE_label.upper() == 'TIMEX':
            # Guess that timex are always children of root
            return [((get_root_node(), child, timex_edge_label), 1.0)]

        # Return multiple predictions with decreasing scores in case of cycles between timex in sentence and prev node
        predictions = []

        # Look for timex in same sentence:
        for parent, child, _ in example:
            predicted = (parent, child, default_label)
            if child.snt_index_in_doc == parent.snt_index_in_doc and parent.TE_label.upper() == 'TIMEX':
                predictions.append((predicted, 1.0))
                break

        # Return parent which is previous node to child (for the first real sentence, previous node is DCT)
        for parent, child, _ in example:
            predicted = (parent, child, default_label)
            if child.node_index_in_doc - parent.node_index_in_doc == 1:
                predictions.append((predicted, 0.75))
                break

        # DCT is not available, give up and return root
        predictions.append(((get_root_node(), child, default_label), 0.25))
        return predictions

