import functools
from typing import Any, Callable, List, Mapping, Tuple, Union
import dataclasses
import jax
from jax import numpy as np
import numpy as onp
from flax import linen as nn
from flax.core.frozen_dict import freeze, unfreeze
from flax.training import train_state as fts
import optax
import tqdm

from layers import MLP, SNDense, SmallConvNet, HLLTModel
from resnet import ResNet18
from utils import *


DTuple = Tuple[np.ndarray, np.ndarray]
Featurizer = Union[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray, bool], np.ndarray]]


@dataclasses.dataclass
class NNConfig(object):
    model: str = 'mlp-dropout'  # mlp, smallconv, resnet, hllt
    dropout_rate: float = 0.2 
    # for MLP
    layers: List[int] = dataclasses.field(default_factory=lambda: [100, 100, 100])
    activation: str = 'swish'
    use_sn: bool = False
    # for smallconv
    small_convnet_hidden: int = 128
    # optim: str = 'adamw'
    n_iters: int = 10000
    batch_size: int = 256
    lr: float = 1e-3
    wd: float = 1e-4
    es_tol: float = 1.05


@dataclasses.dataclass
class RegressorConfig(object):
    seed: int = 23
    inp_dims: int = -1
    out_dims: int = -1
    combined_nn: bool = True
    method: str = 'nn'
    nc: NNConfig = NNConfig()


class TrainState(fts.TrainState):
    mut: Mapping  # pyright: ignore
    rng: np.ndarray  # pyright: ignore

    def pack(self, params: Mapping) -> Mapping:
        return freeze(unfreeze(self.mut) | {'params': params})


def _get_nn_model(nc: NNConfig, out_dims: int) -> nn.Module:
    if nc.model.startswith('mlp'):
        rate = 0. if nc.model.find('dropout') == -1 else nc.dropout_rate
        return MLP(n_layers=nc.layers+[out_dims], activation=nc.activation, dropout_rate=rate,
                   base_layer=(SNDense if nc.use_sn else nn.Dense))
    elif nc.model == 'smallconv':
        return SmallConvNet(
          out_dims=out_dims, n_hidden=nc.small_convnet_hidden, dropout_rate=nc.dropout_rate)
    elif nc.model == 'resnet':
        return ResNet18(
          num_classes=out_dims, dropout_rate=nc.dropout_rate)  # the last layer is linear
    elif nc.model.startswith('hllt'):
        is_s1, im = nc.model.find('s1') != -1, nc.model.find('im') != -1
        return HLLTModel(
            is_s1=is_s1, im_feature=im, activation=nc.activation,
            out_dims=out_dims, dropout_rate=nc.dropout_rate)
    raise NotImplementedError(nc.model)


def _train_nn_model(c: RegressorConfig, model: nn.Module,
                    train_step, val_step, dtrain, dval,
                    x_init: np.ndarray) -> Tuple[float, TrainState]:
    nc = c.nc
    pkey = jax.random.PRNGKey(c.seed)
    pkey, ckey, ck1 = jax.random.split(pkey, 3)
    mut = model.init({'params': ckey, 'dropout': ck1}, x_init, True)
    mut, params = mut.pop('params')

    tx = optax.inject_hyperparams(optax.adamw)(learning_rate=nc.lr, weight_decay=nc.wd)
    pkey, ckey = jax.random.split(pkey)
    train_state = TrainState.create(
        apply_fn=model.apply, params=params, tx=tx, mut=mut, rng=ckey)

    rng = onp.random.RandomState(c.seed)
    dloader = TensorDataLoader(*dtrain, batch_size=nc.batch_size, shuffle=True, rng=rng)
    dloader_val = TensorDataLoader(*dval, batch_size=nc.batch_size*5)

    bpe = dtrain[0].shape[0] // nc.batch_size + 1
    n_epochs = (nc.n_iters + bpe - 1) // bpe

    cur_best = (1e100, None)
    train_loss_acc = Accumulator()
    val_loss_acc = Accumulator()
    with tqdm.trange(n_epochs) as tr:
        for c_ep in tr:
            ctl = Accumulator()
            for dtup in dloader:
                l, train_state = train_step(train_state, dtup)
                ctl.append(l)
            c_loss = ctl.average()

            vtl = Accumulator()
            for dtup in dloader_val:
                vtl.append(val_step(train_state, dtup))
            v_loss = vtl.average()

            if c_ep % (c_ep//20+1) == 0:
                tr.set_postfix(c_loss=c_loss, v_loss=v_loss)

            if c_ep > 10 and c_loss > train_loss_acc.minimum(-50) * 1.05:
                print('lr decay from', train_state.opt_state.hyperparams['learning_rate'])
                train_state.opt_state.hyperparams['learning_rate'] *= 0.75

            if c_ep > 10 and v_loss > val_loss_acc.minimum(-50) * nc.es_tol:
                print('early stopping')
                break

            if cur_best[0] > v_loss:
                cur_best = (v_loss, train_state)

            train_loss_acc.append(c_loss)
            val_loss_acc.append(v_loss)

    return cur_best


def get_nn_predictor(c: RegressorConfig, dtrain: DTuple, dval: DTuple
                     ) -> Tuple[float, Featurizer]:  # loss, predictor
    Xtr, Ytr = dtrain
    if len(Xtr.shape) == 2 and Xtr.shape[1] < 500:  # tabular data; hack for flattened MNIST
        x_mean, x_sd = Xtr.mean(0), Xtr.std(0)+1e-5
    else:  # image data
        x_mean, x_sd = Xtr.mean(), Xtr.std()+1e-5

    # although it doesn't seem to matter, the approximation result requires minimizing
    # E(\|\hat F(x) - Y\|_2^2), as opposed to the channelwise-normalized squared error which
    # may lead to different early stopping behavior.
    y_mean, y_sd = Ytr.mean(), Ytr.std()+1e-5 

    def proc(tup):
        return ((tup[0]-x_mean)/x_sd, (tup[1]-y_mean)/y_sd)
    dtrain, dval = map(proc, (dtrain, dval))

    nc = c.nc
    model = _get_nn_model(nc, c.out_dims)

    @jax.jit
    def train_step(st: TrainState, dtup: DTuple):
        x, y = dtup

        @functools.partial(jax.value_and_grad, has_aux=True)
        def get_loss_and_grad(params: Any, rngs: Any):
            ypred, n_mut = st.apply_fn(
                st.pack(params), x, train=True, mutable=list(st.mut.keys()), rngs=rngs)
            return ((y - ypred)**2).mean(), n_mut

        if nc.model != 'mlp':
            rng, c_rng = jax.random.split(st.rng)
            rngs = {'dropout': c_rng}
        else:
            rng = st.rng
            rngs = None
        (loss, n_mut), grads = get_loss_and_grad(st.params, rngs)
        nst = st.apply_gradients(grads=grads, mut=n_mut, rng=rng)
        return loss, nst

    @jax.jit
    def val_step(st: TrainState, dtup: DTuple):
        x, y = dtup
        yp = st.apply_fn(st.pack(st.params), x, train=False, mutable=False)
        return ((y-yp)**2).mean()

    x0 = np.ones_like(Xtr[:nc.batch_size])
    best_val_loss, best_state = _train_nn_model(
        c, model, train_step, val_step, dtrain, dval, x0)

    def extractor(inp, centered=True):
        inp = (inp - x_mean) / x_sd
        ret = best_state.apply_fn(
            best_state.pack(best_state.params), inp, train=False, mutable=False)
        if centered:
            return ret * y_sd  # for feature extraction only
        return ret * y_sd + y_mean

    return best_val_loss, extractor


ExoDTuple = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]  # zb, w, x, y
ExoPredictor = Callable[[np.ndarray, np.ndarray], np.ndarray]  # (w * x) -> y

def get_nn_exo_predictor(c: RegressorConfig,
                         inst_fea_extractor: Featurizer,
                         gp_fea_extractor: Featurizer,
                         dtrain: ExoDTuple,
                         dval: ExoDTuple) -> Tuple[float, ExoPredictor]:
    """
    Train models with exogenous covariates.
    @params c: c.out_dims stands for the embedding size.
    @params inst_fea_extractor: the extractor returned by get_nn_predictor. with centered=False,
        its i-th output dimension estimates E( gp_embed(x)[i] | z ).
    """
    Zb_tr, Wtr, Xtr, Ytr = dtrain
    w_mean, w_sd = Wtr.mean(), Wtr.std(axis=(None if Wtr.shape[1]>500 else 0))+1e-5
    y_mean, y_sd = Ytr.mean(), Ytr.std()+1e-5
    def _proc(t): return (t[0], (t[1] - w_mean) / w_sd, t[2], (t[3] - y_mean) / y_sd)
    dtrain, dval = map(_proc, (dtrain, dval))

    nc = c.nc

    # While the scales of {inst,gp}_fea_extractor are consistent, their means are not. 
    # We also scale the embedding so its l2 norm is ~ 1/sqrt{n_gps}, to be consistent with the
    # multi-output MLP, which has l2 norm ~sqrt{n_gps}.
    gpe = batched_apply(gp_fea_extractor, Xtr)
    n_gps = gpe.shape[1]
    gpe_mean, gpe_scale = gpe.mean(0), (gpe.std(0)+1e-5) * n_gps  # unscaled l2 ~ sqrt{n_gps}

    def gp_embed(x):
        return add_intercept((gp_fea_extractor(x) - gpe_mean) / gpe_scale)
    def inst_embed(z):
        return add_intercept((inst_fea_extractor(z, centered=False) - gpe_mean) / gpe_scale)

    print('embedding prediction NMSE', 
          mse(batched_apply(gp_embed, Xtr), batched_apply(inst_embed, Zb_tr)) * n_gps**2)

    w_model = _get_nn_model(nc, n_gps + 1)

    @jax.jit
    def train_step(st: TrainState, dtup: ExoDTuple):
        zb, w, _, y = dtup

        @functools.partial(jax.value_and_grad, has_aux=True)
        def get_loss_and_grad(params: Any, rngs: Any):
            w_emb, n_mut = st.apply_fn(
                st.pack(params), w, train=True, mutable=list(st.mut.keys()), rngs=rngs)
            ypred = (w_emb * inst_embed(zb)).sum(1, keepdims=True)
            return ((y - ypred)**2).mean(), n_mut

        if nc.model != 'mlp':
            rng, c_rng = jax.random.split(st.rng)
            rngs = {'dropout': c_rng}
        else:
            rng = st.rng
            rngs = None
        (loss, n_mut), grads = get_loss_and_grad(st.params, rngs)
        nst = st.apply_gradients(grads=grads, mut=n_mut, rng=rng)
        return loss, nst

    @jax.jit
    def val_step(st: TrainState, dtup: ExoDTuple):
        zb, w, _, y = dtup
        w_emb = st.apply_fn(st.pack(st.params), w, train=False, mutable=False)
        yp = (w_emb * inst_embed(zb)).sum(1, keepdims=True)
        return ((y-yp)**2).mean()

    x0 = np.ones_like(Wtr[:nc.batch_size])
    best_val_loss, best_state = _train_nn_model(
        c, w_model, train_step, val_step, dtrain, dval, x0)

    def predictor(w, x):
        w = (w - w_mean) / w_sd
        w_emb = best_state.apply_fn(
            best_state.pack(best_state.params), w, train=False, mutable=False)
        y_pred = (w_emb * gp_embed(x)).sum(1, keepdims=True)
        return y_pred * y_sd + y_mean

    return best_val_loss, predictor

