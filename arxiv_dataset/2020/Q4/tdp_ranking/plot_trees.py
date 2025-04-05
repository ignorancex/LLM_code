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


def main():
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

        doc_valid = True

        for edge in edge_list:
            child, c_label, parent, edge_label = edge
            if child != parent:
                graph.add_edge(parent, child, label=edge_label)
            else:
                doc_valid = False
                print('Invalid tree in document {}, skipping'.format(i))
                break

        if doc_valid:
            graph.layout(prog='dot')
            filename = args.data_file.split('/')[-1]
            graph.draw('{}/{}_doc{}.png'.format(args.output_dir, filename, i))


def plot_paper_tree():
    """Plot the example tree for the published paper"""
    graph = pgv.AGraph(directed=True)

    # Nodes
    graph.add_node('root', label='root')
    graph.add_node('DCT', label='DCT' + r'\n' + 'TIMEX')
    graph.add_node('Friday', label='Friday' + r'\n' + 'TIMEX')
    for word in ['share', 'ruled', 'called', 'saying', 'create', 'signed']:
        graph.add_node(word, label=word + r'\n' + 'EVENT')

    # Parent, child, edge_label
    graph.add_edge('root', 'DCT', label='depends on')
    graph.add_edge('root', 'Friday', label='depends on')
    graph.add_edge('DCT', 'share', label='overlap')
    graph.add_edge('DCT', 'ruled', label='before')
    graph.add_edge('DCT', 'called', label='before')
    graph.add_edge('called', 'saying', label='overlap')
    graph.add_edge('saying', 'create', label='after')
    graph.add_edge('Friday', 'signed', label='overlap')

    graph.layout(prog='dot')
    graph.draw('extracted_outputs/paper_parsetree.png')


if __name__ == '__main__':
    main()
    # plot_paper_tree()
