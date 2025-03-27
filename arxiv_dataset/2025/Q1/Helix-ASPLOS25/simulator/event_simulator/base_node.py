# 2023.12.12 Yixuan Mei

from enum import Enum


class NodeType(Enum):
    """ Type of current node """
    Source = "NodeType.Source"
    Sink = "NodeType.Sink"
    Compute = "NodeType.Compute"


class BaseNode:
    def __init__(self, node_uid: int, node_type: NodeType) -> None:
        """
        Base class for all nodes (compute, source, sink)

        :param node_uid: unique identifier of current node
        :param node_type: type of current node
        :return: None
        """
        self.node_uid: int = node_uid
        self.node_type: NodeType = node_type
