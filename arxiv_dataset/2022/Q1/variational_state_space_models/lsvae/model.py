from vae_common import ConvImageNormalPosDecoder, ConvImageNormalDecoder, ConvImageConcentrationEncoder
from distribution.normal import MultivariateNormal, ConcentrationNormal
from lsvae import LSVAE, NonlinearObservationModel, LinearObservationModel
import jax.numpy as jnp
import haiku as hk

def build_lsvae(config, i):
    A = jnp.array(config.A)
    B = jnp.array(config.B)
    channels = config.channels
    z_dim = config.z_dim
    Prior = jnp.array(config.Prior)
    Prior_conc = jnp.linalg.inv(Prior)

    if config.fit_dynamics:
        A_var = hk.get_parameter("A", A.shape, jnp.float32, lambda s, d: jnp.eye(A.shape[-1]))
        #A_var = A_var.at[0, 0].set(A[0, 0])
        #A_var = A_var.at[1, 1].set(A[1, 1])
        B_var = hk.get_parameter("B", B.shape, jnp.float32, lambda s, d: B)
    else:
        A_var = A
        B_var = B
    # the encoder needs a reasonable initialization for the covariance
    encoder = ConvImageConcentrationEncoder(z_dim,
                    Prior_conc, config.init_L, config.min_cov)
    decoder = ConvImageNormalDecoder(channels, config.sigma)
    Prior_dist = MultivariateNormal(jnp.zeros(Prior.shape[0]), Prior)
    beta = 1

    obs = NonlinearObservationModel('images', encoder, decoder,
            None, 0, 0)
    models = [obs]
    if config.with_y:
        if A.shape[-1] == 4:
            y_obs = LinearObservationModel('states', jnp.array([[1, 0, 0, 0], [0, 1, 0, 0]]),
                        0.05*jnp.eye(2), Prior_conc)
        else:
            y_obs = LinearObservationModel('states', jnp.array([[1, 0]]),
                        0.05*jnp.eye(1), Prior_conc)
        models.append(y_obs)
    ae = LSVAE(models, A_var, B_var, jnp.array(config.Sigma_w),
                Prior_dist, config.z_samples, beta, config.clip_factor)
    return ae
