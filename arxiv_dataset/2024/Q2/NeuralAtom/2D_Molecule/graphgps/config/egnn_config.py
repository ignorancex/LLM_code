from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


def egnn_cfg(cfg):
    """Extending config group of GraphGym's built-in GNN for purposes of our
    CustomGNN network model.
    """

    cfg.egnn = CN()
    cfg.egnn.beta = 0.1
    cfg.egnn.c_max = 1.0
    cfg.egnn.c_min = 0.2
    cfg.egnn.bias_SReLU = -10
    cfg.egnn.dropout = 0.6
    cfg.egnn.output_dropout = 0.2
    cfg.egnn.loss_weight = 5.0


register_config("egnn", egnn_cfg)
