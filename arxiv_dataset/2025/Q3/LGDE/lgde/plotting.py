"""Code for plotting."""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def _plot_semantic_network(
    nx_graph,
    word_to_ind,
    seed_dict,
    discovered_dict,
    doc_freq,
    n_top=15,
    lcc_only=False,
    node_size_factor=0.1,
    alpha=0.7,
    edge_width=0.15,
    plot_with_other_words=False,
):
    """Plot semantic network."""

    # scale node size
    node_size = node_size_factor * np.sqrt(np.asarray(doc_freq))
    nx.set_node_attributes(
        nx_graph, {i: node_size[i] for i in range(len(doc_freq))}, "node_size"
    )

    # store n_top words from seed and discovered keywords
    top_seed = _compute_top_frequent_kws(seed_dict, doc_freq, word_to_ind, n_top)
    nx.set_node_attributes(nx_graph, top_seed, "top_seed")

    top_discovered = _compute_top_frequent_kws(
        discovered_dict, doc_freq, word_to_ind, n_top
    )
    nx.set_node_attributes(nx_graph, top_discovered, "top_discovered")

    other_kws = list(
        (set(list(word_to_ind.keys())) - set(seed_dict)) - set(discovered_dict)
    )
    top_other = _compute_top_frequent_kws(other_kws, doc_freq, word_to_ind, n_top)
    nx.set_node_attributes(nx_graph, top_other, "top_other")

    # word category for node colour
    word_category = np.zeros(nx_graph.number_of_nodes())
    word_category[[word_to_ind[kw] for kw in seed_dict]] = 1
    word_category[[word_to_ind[kw] for kw in discovered_dict]] = 2
    node_color = []
    for i in word_category:
        if i == 0:
            node_color.append("lightblue")
        elif i == 1:
            node_color.append("red")
        elif i == 2:
            node_color.append("orange")
    nx.set_node_attributes(
        nx_graph,
        {i: node_color[i] for i in range(len(node_color))},
        "node_color",
    )

    if lcc_only:
        # reduce to largest connected component
        lcc = max(nx.connected_components(nx_graph), key=len)
        nx_graph = nx_graph.subgraph(lcc)

    # plot
    fig, ax = plt.subplots(1, figsize=(15, 15))

    nx.draw(
        nx_graph,
        ax=ax,
        pos=nx.get_node_attributes(nx_graph, "pos"),
        font_size=15,
        node_size=list(nx.get_node_attributes(nx_graph, "node_size").values()),
        node_color=list(nx.get_node_attributes(nx_graph, "node_color").values()),
        edge_color="lightgrey",
        arrows=False,
        width=edge_width,
        alpha=alpha,
    )

    nx.draw_networkx_labels(
        nx_graph,
        ax=ax,
        pos=nx.get_node_attributes(nx_graph, "pos"),
        labels=nx.get_node_attributes(nx_graph, "top_seed"),
        font_color="darkred",
    )

    nx.draw_networkx_labels(
        nx_graph,
        ax=ax,
        pos=nx.get_node_attributes(nx_graph, "pos"),
        labels=nx.get_node_attributes(nx_graph, "top_discovered"),
        font_color="darkorange",
    )

    if plot_with_other_words:

        nx.draw_networkx_labels(
            nx_graph,
            ax=ax,
            pos=nx.get_node_attributes(nx_graph, "pos"),
            labels=nx.get_node_attributes(nx_graph, "top_other"),
            font_color="darkslateblue",
        )

    return fig


def _compute_top_frequent_kws(kw_list, doc_freq, word_to_ind, n_top=15):
    """Compute top frequent keywords from list."""
    kw_list_freq = np.array([doc_freq[word_to_ind[kw]] for kw in kw_list])
    return {
        word_to_ind[kw_list[i]]: kw_list[i] for i in np.argsort(-kw_list_freq)[:n_top]
    }


def _plot_semantic_communities(
    nx_graph,
    seed_dict,
    semantic_communities,
    word_ind,
    node_size=1,
    n_plots=10,
    figsize=(5, 5),
    path=None,
):
    """Plot semantic communities."""
    n_communities = len(seed_dict)
    semantic_communities = list(semantic_communities.values())

    for j in range(n_communities):
        if j > n_plots - 1:
            return
        else:
            # obtain nodes for community j
            community_node_ids = np.asarray(
                list(map(word_ind.get, semantic_communities[j]))
            )

            if len(community_node_ids) <= 1:
                print(
                    f"The keyword '{seed_dict[j]}' is a singleton semantic community."
                )

            elif len(community_node_ids) > 1:
                # define community as subgraph and compute layout
                community = nx_graph.subgraph(community_node_ids)
                pos = nx.layout.spring_layout(community, seed=2)

                # define labels
                community_seed = {}
                community_disc = {}
                for i, word in enumerate(semantic_communities[j]):
                    if word in seed_dict:
                        community_seed[community_node_ids[i]] = word
                    else:
                        community_disc[community_node_ids[i]] = word

                # define colour
                node_color = []
                for node in community:
                    if node in list(map(word_ind.get, seed_dict)):
                        node_color.append("red")
                    else:
                        node_color.append("orange")

                # plot
                fig, ax = plt.subplots(1, figsize=figsize)
                nx.draw(
                    community,
                    ax=ax,
                    pos=pos,
                    font_size=20,
                    node_size=node_size,
                    node_color=node_color,
                    arrows=False,
                    width=0.5,
                )

                # plot edge weights
                weights = nx.get_edge_attributes(community, "weight")
                nx.draw_networkx_edge_labels(
                    community, ax=ax, pos=pos, edge_labels=weights
                )

                # plot node labels
                nx.draw_networkx_labels(
                    community,
                    ax=ax,
                    pos=pos,
                    labels=community_seed,
                    font_color="darkred",
                )

                nx.draw_networkx_labels(
                    community,
                    ax=ax,
                    pos=pos,
                    labels=community_disc,
                    font_color="darkorange",
                )

                ax.margins(x=0.4)
                ax.axis("off")
                ax.set(title=f'Community of "{seed_dict[j]}"')

                plt.show()

                if path:
                    fig.savefig(
                        path + f"COMM_{seed_dict[j]}.pdf",
                        dpi=fig.dpi,
                        bbox_inches="tight",
                    )
