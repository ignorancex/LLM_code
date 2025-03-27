import haiku as hk
from distribution import Distribution
from distribution.normal import ConcentrationNormal, MultivariateNormal
from functools import partial
import jax
from typing import Any, NamedTuple
from jax.tree_util import tree_map, register_pytree_node_class
import jax.numpy as jnp
from jax.experimental.host_callback import id_print
from util import vmap_rng, disable_params_grad, clip_gradient, scale_gradient, scale_gradient_norm
from distribution.util import log_prob_div

@register_pytree_node_class
class DictionaryKeyDistribution(Distribution):
    def __init__(self, key, dist, transform=None):
        self.key = key
        self.dist = dist
        self.transform = transform

    @property
    def mode(self):
        return {
            self.key: self.dist.mode
        }


    def sample(self, rng):
        return {
            self.key: self.dist.sample(rng)
        }

    def log_prob(self, x):
        x = x[self.key]
        x = x if self.transform is None else self.transform(x)
        return self.dist.log_prob(x)

    def tree_flatten(self):
        return (self.dist,), self.key

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # aux_data is the key,
        # children is dist
        return cls(aux_data, *children)

class ObservationModel(hk.Module):
    def __init__(self, secondary_samples=0, secondary_beta=1):
        super().__init__()
        self.secondary_samples = secondary_samples
        self.secondary_beta = secondary_beta
        
    def secondary_encode(self, x, is_training, cross_axes):
        pass

    # qzx
    def encode(self, x, is_training, cross_axes):
        raise ValueError("Not implemented")
    
    # pxz
    def decode(self, z, is_training, cross_axes):
        raise ValueError("Not implemented")

class NonlinearObservationModel(ObservationModel):
    def __init__(self, key, encoder, decoder, 
                secondary_encoder, secondary_samples, secondary_beta):
        super().__init__(secondary_samples, secondary_beta)
        self._secondary_encoder = secondary_encoder
        self._key = key
        self._encoder = encoder
        self._decoder = decoder

    def secondary_encode(self, x, is_training, cross_axes):
        val = x[self._key]
        encoded = self._secondary_encoder(val, is_training, cross_axes)
        return encoded

    def encode(self, x, is_training, cross_axes):
        val = x[self._key]
        encoded = self._encoder(val, is_training, cross_axes)
        return encoded
    
    def decode(self, z, is_training, cross_axes):
        dist = self._decoder(z, is_training, cross_axes)
        return DictionaryKeyDistribution(self._key, dist)

class LinearObservationModel(ObservationModel):
    def __init__(self, key, C, Sigma_y, Prior_inv):
        super().__init__(0, 0)
        self._key = key
        self._C = C
        self._Sigma_y = Sigma_y
        self._Sigma_y_inv = jnp.linalg.inv(Sigma_y)
        self._Prior_inv = Prior_inv

    def encode(self, x, is_training, cross_axes):
        y = (self._C @ jnp.expand_dims(x[self._key], -1)).squeeze(-1)
        conc = self._C.T @ self._Sigma_y_inv @ self._C + self._Prior_inv
        inf = (self._C.T @ self._Sigma_y_inv @ jnp.expand_dims(y, -1)).squeeze(-1)
        return ConcentrationNormal(inf, conc)
    
    def decode(self, z, is_training, cross_axes):
        x = (self._C @ jnp.expand_dims(z, -1)).squeeze(-1)
        dist = MultivariateNormal(x, self._Sigma_y)
        return DictionaryKeyDistribution(self._key, dist,
                lambda x: (self._C @ jnp.expand_dims(x,-1)).squeeze(-1))

class LSVAE(hk.Module):
    def __init__(self, obs_models, A, B, Sigma_w,
                Prior, z_samples, beta, clip_factor):
        super().__init__()
        self.obs_models = obs_models
        self.A = A
        self.B = B
        self.Sigma_w = Sigma_w
        self.Sigma_w_inv = jnp.linalg.inv(Sigma_w)
        self.Prior = ConcentrationNormal.convert(Prior)
        self.z_samples = z_samples
        self.beta = beta
        self.clip_factor = clip_factor

        self.AT_Sigma_w_inv = self.A.T @ self.Sigma_w_inv
        self.AT_Sigma_w_inv_A = self.A.T @ self.Sigma_w_inv @ self.A

    def _filter(self, qzx_qz, inputs):
        # qzx_qz is a stack of T * n_meas
        # concentration normal distributions/qz
        first_update = tree_map(lambda x: x[0], qzx_qz)
        # initial p(z_1|X_1) distribution
        initial = ConcentrationNormal.pdf_prod_multi(first_update)
        # for the initial we actually need to multiply by the prior...
        initial = ConcentrationNormal.pdf_prod(initial, self.Prior)
        later_meas = tree_map(lambda x: x[1:], qzx_qz)

        @hk.jit
        def core(in_t, prev_state):
            qzx_qzs, u_prev = in_t

            prev_cov = jnp.linalg.inv(prev_state.conc)
            prev_mu = prev_cov @ jnp.expand_dims(prev_state.inf, -1)
            # propagate the state
            mu = self.A @ prev_mu + self.B @ jnp.expand_dims(u_prev, -1) \
                        if u_prev is not None else \
                    self.A @ prev_mu
            cov =  self.A @ prev_cov @ self.A.T + self.Sigma_w
            conc = jnp.linalg.inv(cov)
            inf = conc @ mu

            # convert to concentration normal distribution
            prop_state = ConcentrationNormal(inf.squeeze(-1), conc)

            # convert measurements to updates
            new_state = ConcentrationNormal.pdf_prod_update(prop_state, qzx_qzs)
            return (prop_state, new_state), new_state

        # recursively compute the future covariances
        (prior, post), _ = hk.dynamic_unroll(core, (later_meas, inputs), initial)
        # combine initial with propagated uncertainties
        # to get all the filtered covariances
        combined = tree_map(lambda x, y: 
                    jnp.concatenate((jnp.expand_dims(x, 0),y)),
                                    initial, post)
        return prior, combined

    def _sample(self, rng, posts, inputs):
        # now we go backwards
        @hk.jit
        def core(in_t, next_z):
            rng, filter_post, u = in_t
            next_z = jnp.expand_dims(next_z, -1)
            u = jnp.expand_dims(u, -1) if u is not None else None
            # p(z_{k + 1}|z_k)
            dyn_dist = ConcentrationNormal(
                (self.AT_Sigma_w_inv @ (next_z - self.B @ u)).squeeze(-1),
                self.AT_Sigma_w_inv_A
            ) if u is not None else \
                ConcentrationNormal(
                    (self.AT_Sigma_w_inv @ next_z).squeeze(-1),
                    self.AT_Sigma_w_inv_A
                )

            # combine dyn_dist with the filter post
            q_z_zp = ConcentrationNormal.pdf_prod(dyn_dist, filter_post)
            # now we sample from q_z_zp with probability
            z = q_z_zp.sample(rng)
            pdf = q_z_zp.log_prob(z)
            return (z, pdf), z

        # split rng
        rng, last_rng = jax.random.split(rng)
        rngs = jax.random.split(rng, inputs.shape[0])

        # sample last z
        last_post = tree_map(lambda x: x[-1], posts)
        prev_posts = tree_map(lambda x: x[:-1], posts)

        z_last, pdf_last = last_post.sample_with_prob(last_rng)
        # reverse dynamic_unroll
        (z, pdf), _ = hk.dynamic_unroll(core, (rngs, prev_posts, inputs), z_last, reverse=True)

        pdf = jnp.sum(pdf, 0) + pdf_last # sum pdf across time dimension
        # return z, [T, N], and associated pdf
        z = jnp.concatenate((z, jnp.expand_dims(z_last, 0)), 0)
        return z, pdf

    def _multi_sample(self, rng, posts, inputs, n):
        rngs = jax.random.split(rng, n)
        zs, pdfs = hk.vmap(self._sample,
            in_axes=(0, None, None)
        )(rngs, posts, inputs)
        return zs, pdfs

    def _prior_sample(self, rng, L):
        z0 = self.Prior.sample(rng)

        @hk.jit
        def core(in_t, prev_z):
            rng = in_t
            noise = jax.random.multivariate_normal(rng,
                jnp.zeros((self.Sigma_w.shape[-1],)), self.Sigma_w)
            pz = jnp.expand_dims(prev_z, -1)
            z = (self.A @ pz).squeeze(-1) + noise
            return z, z
        
        rngs = jax.random.split(rng, L - 1)
        zs, _ = hk.dynamic_unroll(core, rngs, z0)

        zs = jnp.concatenate((jnp.expand_dims(z0, 0), zs), 0)
        return zs

    def _log_pzu(self, zs, us):
        z_first = zs[0]
        z_next = zs[1:]
        z_begin = zs[:-1]
        z_pred = (self.A @ jnp.expand_dims(z_begin, -1) + \
                    self.B @ jnp.expand_dims(us, -1)).squeeze(-1) if us is not None else \
                (self.A @ jnp.expand_dims(z_begin, -1)).squeeze(-1)
        # using z_Sigma find log_pdf between predicted and next
        noise = ConcentrationNormal(jnp.zeros(self.Sigma_w_inv.shape[0]), self.Sigma_w_inv)
        log_pdfs = noise.multi_log_prob(z_next - z_pred)
        log_first = self.Prior.log_prob(z_first)
        return log_pdfs.sum(0) + log_first

    def _secondary_vae(self, rng, model, qzx, meas, is_training, cross_axes):
        # for the secondary vae we take the kl divergence
        # between qzx and pz
        div = ConcentrationNormal.kl_divergence(qzx, self.Prior)
        z_samples = qzx.multi_sample(rng, model.secondary_samples)

        def decode(z):
            d = model.decode(z, is_training, cross_axes + ['z'])
            return d.log_prob(meas)

        pxzs = hk.vmap(
            decode,
            axis_name='z'
        )(z_samples).mean(0)

        elbo = -div + pxzs
        return dict(pxz=pxzs, kl_div=div, elbo=elbo)

    def _compute_idv(self, rng, meas, inputs, is_training, cross_axes):
        # find qzx for all measurement models
        qzx_conc = []
        qzx_inf = []
        for model in self.obs_models:
            qzx = hk.vmap(
                lambda x: model.encode(x, is_training, cross_axes + ['t']),
                axis_name='t'
            )(meas)
            qzx_conc.append(qzx.conc)
            qzx_inf.append(qzx.inf)
        # the encoded qzx for all observation models
        qzx = ConcentrationNormal(
            jnp.stack(qzx_inf, 1),
            jnp.stack(qzx_conc, 1)
        )
        qzx_qz = ConcentrationNormal.pdf_div(qzx, self.Prior)
        priors, posts = self._filter(qzx_qz, inputs)

        s_rng, rng = jax.random.split(rng)
        zs, log_qzxs = self._multi_sample(s_rng, posts, inputs, self.z_samples)

        # compute pzu for our zs
        log_pzus = hk.vmap(
            lambda z_traj: self._log_pzu(z_traj, inputs)
        )(zs)

        # now compute D_KL(qzx|pzu) across the samples
        kl_div = (log_qzxs - log_pzus).mean(0)

        # compute E_qzx[pxz] across the samples
        log_pxzs = []

        secondary_pxzs = []
        secondary_divs = []
        secondary_elbos = []

        secondary_inf = []
        secondary_conc = []

        obs_rngs = jax.random.split(rng, len(self.obs_models))
        for (i, m), rng in zip(enumerate(self.obs_models), obs_rngs):
            if m.secondary_samples:
                # find the secondary encoding
                qhzx = hk.vmap(
                    lambda x: m.secondary_encode(x, is_training, cross_axes + ['t']),
                    axis_name='t'
                )(meas)
                # qhzx = jax.tree_map(lambda x: x[:, i], qzx)
                qhzx_qz = ConcentrationNormal.pdf_div(qhzx, self.Prior)

                def pxz_per_t(qhzx, qhzx_qz, meas, zs):
                    #z0_dec = m.decode(jax.lax.stop_gradient(zs[0]), 
                    #                is_training, cross_axes + ['t'])
                    #z0_prob = z0_dec.log_prob(meas)

                    # qhzx_qz prob to backprop to zs
                    #qhzx_qz = jax.tree_map(jax.lax.stop_gradient, qhzx_qz)
                    #zs_prob = qhzx_qz.multi_log_prob(zs).mean(0)
                    #return z0_prob + zs_prob
                    # this this have some concentration
                    def pdf(z):
                        return log_prob_div(qhzx, self.Prior, qhzx_qz, z)
                    prob = hk.vmap(pdf)
                    return prob(zs).mean(0)

                log_pxz = hk.vmap(
                    pxz_per_t,
                    axis_name='t', in_axes=(0, 0, 0, 1)
                )(qhzx, qhzx_qz, meas, zs).sum(0)
                log_pxzs.append(log_pxz)
                # add a secondary elbo
                # for each measurement
                sr = hk.vmap(
                    lambda qzx, x: self._secondary_vae(rng, m, qzx, 
                                x, is_training, cross_axes + ['t']),
                    axis_name='t'
                )(qhzx, meas)
                sr = tree_map(jnp.sum, sr) # sum secondary elbos across time
                secondary_elbos.append(m.secondary_beta*sr['elbo'])
                secondary_divs.append(sr['kl_div'])
                secondary_pxzs.append(sr['pxz'])
                secondary_inf.append(qhzx.inf)
                secondary_conc.append(qhzx.conc)
            else:
                def pxz_per_traj(zs):
                    def per_z(x, z):
                        if self.clip_factor > 0:
                            z = scale_gradient_norm(self.clip_factor, z)
                        return m.decode(z, is_training, cross_axes + ['t']).log_prob(x)
                    pdfs = hk.vmap(
                        per_z,
                        axis_name='t'
                    )(meas, zs)
                    return pdfs.sum(0)
                log_pxz = pxz_per_traj(zs[0]) # use only a single sample
                log_pxzs.append(log_pxz)



        if len(secondary_elbos) > 0:
            secondary_qzx = ConcentrationNormal(
                jnp.stack(secondary_inf, 1),
                jnp.stack(secondary_conc, 1)
            )
            secondary_res = {
                'elbo': jnp.stack(secondary_elbos).sum(),
                'kl_div': jnp.stack(secondary_divs).sum(),
                'pxz': jnp.stack(secondary_pxzs).sum()
            }
            secondary_elbo = secondary_res['elbo']
        else:
            secondary_res = None
            secondary_elbo = 0
            secondary_qzx = None

        log_pxz = jnp.stack(log_pxzs).sum(0)

        elbo = -self.beta*kl_div + log_pxz

        return dict(
            stats=dict(
                loss=-elbo - secondary_elbo,
                secondary=secondary_res,
                elbo=elbo,
                kl_div=kl_div,
                log_pxz=log_pxz,
                log_pzu=log_pzus.mean(0),
                log_qzx=log_qzxs.mean(0)
            ),
            data=dict(
                z=zs,
                qzx=qzx,
                secondary_qzx=secondary_qzx,
                prior=priors,
                post=posts,
            )
        )

    def compute(self, rng, meas, inputs, is_training, cross_axes):
        batch_dim = inputs.shape[0]
        res = vmap_rng(
            lambda rng, meas, inputs: 
                self._compute_idv(rng, meas, inputs, 
                        is_training, cross_axes + ['batch']),
            axis_name='batch'
        )(rng, meas, inputs)

        return dict(
            stats=tree_map(lambda x: jnp.mean(x, 0), res['stats']),
            data=res['data']
        )