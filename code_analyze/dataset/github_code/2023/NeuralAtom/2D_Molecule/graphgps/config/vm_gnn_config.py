from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


def vmgnn_cfg(cfg):
    """Extending config group of GraphGym's built-in GNN for purposes of our
    VM GNN network model.
    """
    cfg.vm = CN()
    cfg.vm.vn_residual = True  # * Virtual Node residual
    cfg.vm.vn_drop_ratio = 0.5  # * follow CIGA setting
    cfg.vm.sim_method = "dot"  # * 'dot', 'cos'
    cfg.vm.topk_ratio = 0.5  # * ratio according to cfg.train.batch_size

    cfg.vm.mixer = CN()
    cfg.vm.mixer.hidd_dim = 512
    cfg.vm.mixer.depth = 2
    cfg.vm.mixer.expansion_factor = 4.0  # * channel factor
    cfg.vm.mixer.expansion_factor_token = 0.5  # * token factor
    cfg.vm.mixer.drop_ratio = 0.0

    cfg.model.residual = True  # * intra-layer residual


register_config("vmgnn", vmgnn_cfg)
