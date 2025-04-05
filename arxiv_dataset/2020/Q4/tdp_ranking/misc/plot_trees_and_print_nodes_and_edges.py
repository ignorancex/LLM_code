import argparse
import pygraphviz as pgv

from data_preparation import create_snt_edge_lists, create_node_list
from data_structures import get_root_node


def get_arg_parser():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--data_file", help="annotated data")
    arg_parser.add_argument("--output_dir", help="directory to write the tree PNGs to")
    arg_parser.add_argument("--TE_label_set",
                            help="which timex/event label set to use: none, timex_event, or full",
                            choices=['none', 'timex_event', 'time_ml', 'full'], default='timex_event')

    return arg_parser


if __name__ == '__main__':
    arg_parser = get_arg_parser()
    args = arg_parser.parse_args()

    # Read as a list of documents
    data = open(args.data_file, 'r', encoding='utf-8').read()
    doc_list = data.strip().split('\n\nfilename')

    for i, document in enumerate(doc_list):
        doc_lines = document.strip().split('\n')

        # create node_list
        snt_list, edge_list = create_snt_edge_lists(doc_lines)
        node_list = create_node_list(snt_list, edge_list)
        root_node = get_root_node()
        node_list.insert(0, root_node)

        graph = pgv.AGraph(directed=True)
        for node in node_list:
            label = node.TE_label if args.TE_label_set else node.full_label
            graph.add_node(node.ID, label=node.words + r'\n' + label)
            print("node\t" + str(node.ID) + "\t" + str(node.words))

        doc_valid = True

        for edge in edge_list:
            child, c_label, parent, edge_label = edge
            if child != parent:
                graph.add_edge(parent, child, label=edge_label)
                print ("edge\t" + parent + "\t" + edge_label + "\t" + child)
            else:
                doc_valid = False
                print('Invalid tree in document {}, skipping'.format(i))
                break

        if doc_valid:
            graph.layout(prog='dot')
            filename = args.data_file.split('/')[-1]
            graph.draw('{}/{}_doc{}.png'.format(args.output_dir, filename, i))
