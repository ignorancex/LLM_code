from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


def gvm_cfg(cfg):
    """Extending config group of GraphGym's built-in GNN for purposes of our
    CustomGNN network model.
    """

    cfg.gvm = CN()
    cfg.gvm.avg_nodes = 50
    cfg.gvm.pool_ratio = 0.25
    cfg.gvm.n_pool_heads = 2
    cfg.gvm.mixer_dim = 256
    cfg.gvm.mixer_depth = 1
    cfg.gvm.mixer_token_exp_fac = 0.5
    cfg.gvm.mixer_dropout = 0.0

register_config('gvm_model', gvm_cfg)
