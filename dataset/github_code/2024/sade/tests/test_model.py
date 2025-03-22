import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import ml_collections
import pytest
import torch

from sade.models import registry


@pytest.fixture
def test_config():
    config = ml_collections.ConfigDict()
    config.device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    data = config.data = ml_collections.ConfigDict()
    data.cache_rate = 0
    data.dataset = "ABCD"
    data.image_size = (8, 4, 8)
    data.spacing_pix_dim = 8.0
    data.num_channels = 1
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
    model.nf = 4
    model.norm_num_groups = 2
    model.blocks_down = (1, 2, 1)
    model.blocks_up = (1, 1)
    model.time_embedding_sz = 4
    model.init_scale = 0.0
    model.conv_size = 3
    model.self_attention = False
    model.dropout = 0.0
    model.embedding_type = "fourier"
    model.fourier_scale = 2.0
    model.learnable_embedding = False
    model.dilation = 1
    model.jit = False
    # This should not be here eventually...
    model.resblock_pp = True

    # flow-model
    config.flow = flow = ml_collections.ConfigDict()
    flow.num_blocks = 2
    flow.patch_batch_size = 8
    flow.context_embedding_size = 16
    flow.use_global_context = True
    flow.global_embedding_size = 32

    # Config for patch sizes
    flow.local_patch_config = ml_collections.ConfigDict()
    flow.local_patch_config.kernel_size = 3
    flow.local_patch_config.padding = 1
    flow.local_patch_config.stride = 1

    # Config for larger receptive fields outputting gobal context
    flow.global_patch_config = ml_collections.ConfigDict()
    flow.global_patch_config.kernel_size = 7
    flow.global_patch_config.padding = 2
    flow.global_patch_config.stride = 4

    # MSMA params
    config.msma = msma = ml_collections.ConfigDict()
    msma.min_timestep = 1e-1
    msma.max_timestep = 1.0
    msma.n_timesteps = 3

    return config


def test_model_initialization(test_config):
    test_config.model.blocks_down = (1, 2)
    test_config.model.blocks_up = (1,)
    score_model = registry.create_model(test_config)
    assert score_model is not None


def test_model_output_shape(test_config):
    test_config.data.num_channels = 7
    score_model = registry.create_model(test_config)
    score_model = score_model.eval().requires_grad_(False)
    score_fn = registry.get_model_fn(score_model, train=False)
    N, C, H, W, D = 3, test_config.data.num_channels, *test_config.data.image_size
    x = torch.ones(N, C, H, W, D)
    y = torch.tensor([1] * N)

    with torch.no_grad():
        score = score_fn(x, y)

    assert not torch.isnan(score).any()
    assert score.shape == (N, C, H, W, D)


def test_mixed_precision(test_config):
    score_model = registry.create_model(test_config)
    score_model = score_model.eval().requires_grad_(False)
    score_fn = registry.get_model_fn(score_model, amp=True, train=False)
    N, C, H, W, D = 1, test_config.data.num_channels, *test_config.data.image_size
    x = torch.ones(N, C, H, W, D)
    y = torch.tensor([1] * N)

    with torch.no_grad():
        score = score_fn(x, y)

    if test_config.device == torch.device("cpu"):
        assert score.dtype == torch.float32
    else:
        assert score.dtype == torch.float16


def test_msma_score_fn(test_config):
    test_config.msma.n_timesteps = S = 3

    score_model = registry.create_model(test_config)
    score_model = score_model.eval().requires_grad_(False)
    msma_score_fn = registry.get_msma_score_fn(
        test_config, score_model, return_norm=False, denoise=False
    )
    msma_score_norm_fn = registry.get_msma_score_fn(
        test_config, score_model, return_norm=True, denoise=False
    )

    N, C, H, W, D = 3, test_config.data.num_channels, *test_config.data.image_size
    x = torch.ones(N, C, H, W, D)

    with torch.no_grad():
        score = msma_score_fn(x)
        score_norm = msma_score_norm_fn(x)

    assert not torch.isnan(score).any()
    assert score.shape == (N, S, H, W, D)

    assert not torch.isnan(score_norm).any()
    assert score_norm.shape == (N, S)


def test_flow_model_output_shape(test_config):
    flow_model = registry.create_flow(test_config)
    flow_model = flow_model.eval().requires_grad_(False)
    N, C, H, W, D = 3, test_config.msma.n_timesteps, *test_config.data.image_size
    x = torch.ones(N, C, H, W, D, dtype=torch.float32, device=test_config.device)

    with torch.no_grad():
        z, jac = flow_model(x, fast=True)

    assert not torch.isnan(z).any()
    assert not torch.isnan(jac).any()
    assert z.shape == (H * W * D, N, C)
    assert jac.shape == (H * W * D, N)

    with torch.no_grad():
        log_probs = flow_model.log_density(x)

    assert not torch.isnan(log_probs).any()
    assert log_probs.shape == (N, H, W, D)
