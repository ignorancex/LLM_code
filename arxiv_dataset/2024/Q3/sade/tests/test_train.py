import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import ml_collections
import pytest
import torch

from sade.losses import get_sde_loss_fn
from sade.models import ncsnpp3d
from sade.models.ema import ExponentialMovingAverage
from sade.models.registry import create_model, create_sde
from sade.optim import get_step_fn, optimization_manager
from sade.sde_lib import VESDE


@pytest.fixture
def test_config():
    config = ml_collections.ConfigDict()
    config.device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    config.fp16 = False

    data = config.data = ml_collections.ConfigDict()
    data.cache_rate = 0
    data.dataset = "ABCD"
    data.image_size = (8, 8, 8)
    data.num_channels = 2
    data.spacing_pix_dim = 1.0
    data.dir_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "dummy_data")
    data.splits_dir = data.dir_path

    config.training = ml_collections.ConfigDict()
    config.training.batch_size = 1
    config.training.sde = "vesde"
    config.fp16 = False

    config.eval = ml_collections.ConfigDict()
    config.eval.batch_size = 1

    model = config.model = ml_collections.ConfigDict()
    # for sde
    model.sigma_min = 1e-2
    model.sigma_max = 1.0
    model.num_scales = 3
    # arch specifc
    model.name = "ncsnpp3d"
    model.resblock_type = "biggan"
    model.act = "memswish"
    model.scale_by_sigma = True
    model.ema_rate = 0.9999
    model.nf = 8
    model.norm_num_groups = 2
    model.blocks_down = (1, 2, 1)
    model.blocks_up = (1, 1)
    model.time_embedding_sz = 8
    model.init_scale = 0.0
    model.conv_size = 3
    model.self_attention = False
    model.dropout = 0.0
    model.resblock_pp = True
    model.embedding_type = "fourier"
    model.fourier_scale = 2.0
    model.learnable_embedding = False
    model.dilation = 1
    model.jit = False

    # optimization
    optim = config.optim = ml_collections.ConfigDict()
    optim.weight_decay = 0.0
    optim.optimizer = "Adam"
    optim.scheduler = "skip"
    optim.lr = 1e-3
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = -1
    optim.grad_clip = 10.0

    return config


def test_sde_initialization(test_config):
    test_config.model.sigma_min = 1e-2
    test_config.model.sigma_max = 3.0
    test_config.model.num_scales = 2

    sde = VESDE(
        sigma_min=test_config.model.sigma_min,
        sigma_max=test_config.model.sigma_max,
        N=test_config.model.num_scales,
    )

    assert sde is not None
    assert torch.isclose(sde.discrete_sigmas, torch.tensor([1e-2, 3.0])).all()
    assert torch.isclose(
        sde.noise_schedule_inverse(sde.discrete_sigmas), torch.tensor([0.0, 1.0])
    ).all()

    x = torch.zeros(1, 2, 2, 2)
    _, std = sde.marginal_prob(x, torch.tensor(sde.T))
    assert torch.isclose(std, torch.tensor([3.0])).all()


def test_loss_fn(test_config):
    test_config.model.sigma_min = 1e-2
    test_config.model.sigma_max = 1.0
    test_config.model.num_scales = 1

    score_model = create_model(test_config)
    sde = create_sde(test_config)
    loss_fn = get_sde_loss_fn(
        sde, train=True, reduce_mean=True, likelihood_weighting=False, amp=False
    )
    N, C, H, W, D = 1, test_config.data.num_channels, *test_config.data.image_size
    x = torch.zeros(N, C, H, W, D, device=test_config.device)

    with torch.no_grad():
        loss = loss_fn(score_model, x)

    assert loss.shape == torch.Size([])
    assert not loss.isnan()
    expected_loss = (x + sde.sigma_max).sum()
    assert loss < expected_loss


def test_optimization_fn(test_config):
    test_config.model.num_scales = 1
    score_model = create_model(test_config)
    sde = create_sde(test_config)
    state_dict = {"model": score_model, "step": 0}
    optimize_fn = optimization_manager(state_dict, test_config)
    assert optimize_fn is not None
    assert "optimizer" in state_dict
    assert state_dict["optimizer"] is not None
    optimizer = state_dict["optimizer"]
    assert isinstance(optimizer, torch.optim.Adam)
    loss_fn = get_sde_loss_fn(
        sde, train=True, reduce_mean=False, likelihood_weighting=False, amp=False
    )
    N, C, H, W, D = 1, test_config.data.num_channels, *test_config.data.image_size
    x = torch.zeros(N, C, H, W, D, device=test_config.device)

    torch.manual_seed(42)
    with torch.inference_mode():
        pre_opt_loss = loss_fn(score_model, x).item()
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(score_model, x)
        loss.backward()
        optimize_fn(score_model.parameters(), step=0)
    torch.manual_seed(42)  # trying to get the same results
    with torch.inference_mode():
        post_opt_loss = loss_fn(score_model, x).item()
    assert post_opt_loss < pre_opt_loss


def test_train_step(test_config):
    score_model = create_model(test_config)
    sde = create_sde(test_config)
    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=test_config.model.ema_rate
    )
    state_dict = dict(model=score_model, ema=ema, step=0)
    optimize_fn = optimization_manager(state_dict, test_config)

    train_step_fn = get_step_fn(
        sde,
        train=True,
        optimize_fn=optimize_fn,
        reduce_mean=True,
        use_fp16=False,
    )

    N, C, H, W, D = 1, test_config.data.num_channels, *test_config.data.image_size
    x = torch.ones(N, C, H, W, D, device=test_config.device)

    loss = train_step_fn(state_dict, x)
    assert loss.shape == torch.Size([])
    assert not loss.isnan()
    expected_loss = (x + sde.sigma_max).sum()
    assert loss < expected_loss
