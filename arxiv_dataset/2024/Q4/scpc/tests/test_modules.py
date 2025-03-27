from typing import Type
from tempfile import TemporaryFile

import pytest
import torch
import numpy as np

from scpc.modules import *


@pytest.mark.parametrize(
    "Encoder",
    [
        CausalSelfAttentionEncoder,
        ConvEncoder,
        FeedForwardEncoder,
        IdentityEncoder,
        RecurrentEncoder,
        SelfAttentionEncoder,
        EncoderSequence,
        CausalConvEncoder,
    ],
    ids=[
        "CausalSelfAttentionEncoder",
        "ConvEncoder",
        "FeedForwardEncoder",
        "IdentityEncoder",
        "RecurrentEncoder",
        "SelfAttentionEncoder",
        "EncoderSequence",
        "CausalConvEncoder",
    ],
)
def test_encoder_variable_batches(Encoder: Type[Encoder]):
    torch.manual_seed(0)
    N, Tmin, Tmax, H = 10, 300, 600, 256
    if Encoder is EncoderSequence:
        encoder = EncoderSequence(FeedForwardEncoder(H), FeedForwardEncoder(H))
    elif Encoder is SelfAttentionEncoder:
        encoder = SelfAttentionEncoder(H, pos_period=Tmax)
    else:
        encoder = Encoder(H)
    encoder.eval()
    x, lens = [], []
    x_exp = []
    for _ in range(N):
        lens_n = torch.randint(Tmin, Tmax + 1, (1,))
        x_n = torch.rand(1, lens_n.item(), H)
        x_exp_n, lens__n = encoder(x_n)
        assert x_exp_n.size(-1) == encoder.output_size
        assert not (x_exp_n == 0).all()  # check we're not just padding nothing
        assert lens__n is None
        x.append(x_n.flatten())
        lens.append(lens_n)
        x_exp.append(x_exp_n.squeeze(0))
    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=-1.0)
    x = x.view(N, -1, H)
    lens = torch.cat(lens)
    x_act, lens_act = encoder(x, lens)
    for n, x_exp_n, x_act_n, lens_act_n in zip(range(N), x_exp, x_act, lens_act):
        assert lens_act_n == x_exp_n.size(0), n
        assert (x_act_n[lens_act_n:] == 0).all(), n
        assert torch.allclose(x_exp_n, x_act_n[:lens_act_n], atol=1e-5), n


def test_causal_self_attention_encoder_is_causal():
    torch.manual_seed(1)
    N, T, H = 3, 13, 17
    x = torch.rand(N, T, H)
    encoder = CausalSelfAttentionEncoder(H, 2 * H, T // 2 - 1, num_heads=1)
    encoder.eval()
    x1 = encoder(x)[0][:, : T // 2]
    x2 = encoder(x[:, : T // 2])[0]
    assert x1.shape == x2.shape
    assert torch.allclose(x1, x2, atol=1e-5)
    x1 = x1[:, -2:]
    x2 = encoder(x[:, 1 : T // 2])[0][:, -2:]
    assert not torch.allclose(x1[:, 0], x2[:, 0], atol=1e-5)
    assert torch.allclose(x1[:, 1], x2[:, 1], atol=1e-5)


def test_causal_conv_encoder_is_causal():
    torch.manual_seed(420)
    N, T, H = 5, 12, 20
    x = torch.rand(N, T, H)
    # for 2 layers + kernel T // 4, receptive field is T // 2 - 1
    encoder = CausalConvEncoder(H, 2 * H, T // 4, num_layers=2)
    encoder.eval()
    x1 = encoder(x)[0][:, : T // 2]
    x2 = encoder(x[:, : T // 2])[0]
    assert x1.shape == x2.shape
    assert torch.allclose(x1, x2, atol=1e-5)
    x1 = x1[:, -2:]
    x2 = encoder(x[:, 1 : T // 2])[0][:, -2:]
    assert not torch.allclose(x1[:, 0], x2[:, 0], atol=1e-5)
    assert torch.allclose(x1[:, 1], x2[:, 1], atol=1e-5)


@pytest.mark.parametrize(
    "embedding", [True, False], ids=["w/-embedding", "w/o-embedding"]
)
@pytest.mark.parametrize("penc", ["ff", "recur", "csa"])
@pytest.mark.parametrize("offset", [0, 2], ids=["o0", "o2"])
@pytest.mark.parametrize("gutted", [0, 1], ids=["g0", "g1"])
@pytest.mark.parametrize("averaging_penalty", [0.0], ids=["a0"])
@pytest.mark.parametrize("lengths", [True, False], ids=["w/ lengths", "w/o lengths"])
def test_cpc_prediction_network_matches_manual_comp(
    embedding, penc, offset, gutted, averaging_penalty, lengths
):
    torch.manual_seed(2)
    N, T, Cl, Cc, K, M = 5, 7, 9, 11, 3, 100_000
    lens = torch.randint(K + offset + 1, T + 1, (N,))
    if lengths:
        lens[0] = T
    else:
        lens[:] = T
    sum_lens = lens.sum().item()
    if penc == "ff":
        penc = None
    elif penc == "csa":
        penc = CausalSelfAttentionEncoder(Cc, Cl * (K - gutted), num_heads=1)
    else:
        penc = RecurrentEncoder(Cc, Cl * (K - gutted))
    net = CPCLossNetwork(
        Cl,
        Cc,
        K,
        M,
        penc,
        N if embedding else None,
        0,
        offset,
        gutted,
        averaging_penalty,
    )
    net.eval()

    latent, context = torch.randn(sum_lens, Cl), torch.randn(N, T, Cc)
    speakers = torch.randint(N, (N,))
    samp = torch.randint(sum_lens, (N * (T - 1) * M,))
    if embedding:
        latent_samp = (latent + net.embed(speakers).repeat_interleave(lens, 0))[samp]
    else:
        latent_samp = latent[samp]
    latent_samp = latent_samp.view(N, T - 1, M, Cl)

    loss_exp, latents_ = 0.0, []
    for n in range(N):
        lens_n = lens[n]
        latent_n, latent = latent[:lens_n], latent[lens_n:]
        latents_.append(latent_n)
        latent_samp_n = latent_samp[n]
        assert latent_n.size(0) == lens_n
        if embedding:
            latent_n = latent_n + net.embed(speakers[n : n + 1])
        context_n = context[n, :lens_n]
        Az_n = net.prediction_encoder(context_n.unsqueeze(0))[0].squeeze(0)
        assert Az_n.shape == (lens_n, Cl * (K - gutted))
        Az_n = Az_n.view(lens_n, K - gutted, Cl).transpose(0, 1)
        mse = latent_n[offset:].mean(0).square().mean()
        for k in range(1 + gutted, K + 1):
            Az_n_k = Az_n[k - gutted - 1]  # (lens_n - 1, Cl)
            loss_exp_nk = 0
            for t in range(lens_n - K - offset):
                Az_n_k_t = Az_n_k[t + offset]  # (Cl,)
                latent_n_tpk = latent_n[t + k + offset]  # (Cl,)
                num = Az_n_k_t @ latent_n_tpk
                assert num.numel() == 1
                latent_samp_n_t = latent_samp_n[t]  # (M, Cl)
                denom = (latent_samp_n_t @ Az_n_k_t).logsumexp(0)
                assert denom.numel() == 1
                loss_exp_nk = loss_exp_nk - (num - denom) + averaging_penalty * mse
            loss_exp = loss_exp + loss_exp_nk / ((lens_n - K - offset) * (K - gutted))
    loss_exp = loss_exp / N  # + math.log(M + 1)

    latents = torch.nn.utils.rnn.pad_sequence(latents_, True, -1)
    loss_act = net(latents, context, lens if lengths else None, speakers)
    assert torch.isclose(loss_exp, loss_act, rtol=1e-1)


class DummyEncoder(IdentityEncoder, json_name="dummy"):
    pass


@pytest.mark.parametrize("from_class", ["Encoder", "EncoderSequence", "DummyEncoder"])
def test_checkpoints(from_class):
    torch.manual_seed(3)
    N, Tmax, H = 100, 1000, 8
    encoder1 = EncoderSequence(
        CausalSelfAttentionEncoder(
            H, max_width=5, dim_feedforward=10, num_heads=1, pos_period=1
        ),
        FeedForwardEncoder(H, nonlin_type="sigmoid", bias=False),
        IdentityEncoder(H),
        RecurrentEncoder(H, num_layers=2, recurrent_type="lstm"),
        SelfAttentionEncoder(H),
        EncoderSequence(IdentityEncoder(H)),
        ConvEncoder(H, norm_style="channel"),
        CausalConvEncoder(H, kernel_size=3, num_layers=3),
    )
    f = TemporaryFile()
    encoder1.eval()
    encoder1.save_checkpoint(f)
    f.seek(0)
    if from_class == "Encoder":
        encoder2 = Encoder.from_checkpoint(f)
    elif from_class == "EncoderSequence":
        encoder2 = EncoderSequence.from_checkpoint(f)
    else:
        with pytest.raises(ValueError, match="subclass"):
            DummyEncoder.from_checkpoint(f)
        return
    encoder2.eval()
    x = torch.randn(N, Tmax, H)
    lens = torch.randint(1, Tmax + 1, (N,))
    out1, lens1 = encoder1(x, lens)
    out2, lens2 = encoder2(x, lens)
    assert (lens1 == lens2).all()
    assert torch.allclose(out1, out2)


def test_contiguous_temporal_mask():
    torch.manual_seed(4)
    N, W, T, C, p, std = 10_000, 5, 100, 5, 0.1, 2.0
    x = torch.ones(N, T, C)
    y = ContiguousTemporalMask(W, p, mean=0.0, std=0.0)(x)
    assert torch.isclose(y.mean(), torch.tensor(1 - p), atol=1e-3)
    y = y.transpose(1, 2).flatten(0, 1)
    for nc, y_nc in enumerate(y):
        # the left and rightmost windows can be shorter than W
        uy_nc, counts_nc = torch.cat(
            [torch.zeros(W), y_nc, torch.zeros(W)]
        ).unique_consecutive(return_counts=True)
        for uy_nct, counts_nct in zip(uy_nc, counts_nc):
            if uy_nct == 0.0:
                assert counts_nct >= W, nc
    y = ContiguousTemporalMask(W, 1.0, std=std)(x)
    assert y.shape == x.shape
    assert torch.isclose(y.mean(), torch.ones(1), atol=1e-3)
    assert torch.isclose(y.std(), torch.tensor(std), atol=1e-3)


@pytest.mark.parametrize("offset", [0, 3], ids=["nooffs", "offs"])
@pytest.mark.parametrize("num_speakers", [None, 5], ids=["nospkrs", "spkrs"])
def test_best_rq_loss_network_matches_manual_comp(offset, num_speakers):
    torch.manual_seed(5)
    N, T, W, F, C, offset = 100, 30, 5, 5, 8, 0
    assert T % 2 == 0
    ff = FeedForwardEncoder(C)
    feats = torch.randn(N, T, F)
    mask = torch.randint(0, 2, (N, T)).bool()
    lens = torch.randint(1, T // 2 + 1, (N,))
    context = torch.randn(N, T // 2, C)
    sources = torch.randint(1 if num_speakers is None else num_speakers, (N,))
    best_rq = BestRqLossNetwork(2 * F, ff, num_speakers=num_speakers, offset=offset)
    loss_exp = 0.0
    denom = 0
    for n in range(N):
        feats_n, mask_n, lens_n, context_n = feats[n], mask[n], lens[n], context[n]
        feats_n, mask_n = feats_n.view(T // 2, 2 * F), mask_n.view(T // 2, 2).all(1)
        feats_n = torch.nn.functional.normalize(feats_n @ best_rq.proj_matrix)
        if num_speakers:
            context_n = context_n + best_rq.embed(sources[n : n + 1])
        logp_n = torch.nn.functional.log_softmax(ff(context_n)[0], 1)
        lens_n = lens_n.item()
        for t in range(offset, lens_n):
            feats_nt, mask_nt = feats_n[t], mask_n[t]
            if mask_nt.item():
                denom += 1
                target_nt_idx, target_nt_norm = -1, float("inf")
                for c in range(C):
                    codebook_c = best_rq.codebook[c]
                    target_c_norm = torch.linalg.vector_norm(feats_nt - codebook_c)
                    if target_c_norm < target_nt_norm:
                        target_nt_idx, target_nt_norm = c, target_c_norm
                loss_exp = loss_exp - logp_n[t, target_nt_idx]
    loss_exp = loss_exp / denom
    loss_act = best_rq(feats, mask, context, lens, sources)
    assert torch.isclose(loss_exp, loss_act)


def test_high_pass_filter():
    torch.manual_seed(6)
    N, T, C, W = 5, 10, 3, 5
    x = torch.randn(N, T, C)
    filter_ = np.hanning(W)
    filter_[::2] *= -1
    hfs = HighPassFilter(W)
    assert torch.allclose(torch.tensor(filter_).float(), hfs.filter_.flip(0))
    exp = []
    for x_nc in x.transpose(1, 2).flatten(0, 1):
        x_nc = np.concatenate([np.zeros(W - 1), x_nc.numpy()])
        exp_nc = np.convolve(x_nc, filter_, "valid")
        exp.append(torch.from_numpy(exp_nc))
    exp = torch.stack(exp).view(N, C, T).transpose(1, 2).float()
    act = HighPassFilter(W)(x)
    assert exp.shape == act.shape
    assert torch.allclose(exp, act)
