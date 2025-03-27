from xoa.managers.t_mgr import TrainingJobManager
from xoa.managers.ws_mgr import WebServiceManager
try:
    import optimizers.bandit
    from xoa.managers.s_mgr import HPOJobManager
    from xoa.managers.p_mgr import ParallelHPOManager
except ImportError as ie:
    print("Modules for training node only")