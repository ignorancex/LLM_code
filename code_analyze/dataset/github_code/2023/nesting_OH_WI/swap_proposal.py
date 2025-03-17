import random 

def swap(partition):
    """Swap two random nodes, chosen with replacement, between their districts.

    Does NOT validate whether or not this is a valid partition, that is done by the constraints function.

    :param partition: The current partition to propose a swap from.
    :return: a proposed next `~gerrychain.Partition`
    """

    # choose two random nodes with replacement
    node_1, node_2 = random.choices(tuple(partition.graph), k=2)

    # find what districts nodes belong to
    district_1 = partition.assignment[node_1]
    district_2 = partition.assignment[node_2]

    # the new partition swaps the two nodes
    return partition.flip({node_1: district_2, node_2: district_1})