def is_in_component(scc, graph_embedding, largest_scc):
    # is scc an IN component to largest_scc?
    out_nodes = set()
    for src in scc:
        out_nodes.update(graph_embedding[src])
    if len(out_nodes.intersection(set(largest_scc))) > 0:
        return True
    else:
        return False


def is_out_component(scc, graph_embedding, largest_scc):
    # is scc an OUT component to largest_scc?
    out_nodes = set()
    for src in largest_scc:
        out_nodes.update(graph_embedding[src])
    if len(out_nodes.intersection(set(scc))) > 0:
        return True
    else:
        return False
