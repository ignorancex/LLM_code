"""Multivariate Student T distribution."""
import tensorflow as tf
import tensorflow_probability as tfp

#from psych_metric.supervised_joint_distrib import is_prob_distrib

# NOTE This for use when df, loc, and scale availble, or when fitting with
# gradient descent or Tensorflow Nelder-Mead, or tfp.mcmc.NoUTurnSampler

class MultivariateStudentT(tfp.distributions.MultivariateStudentTLinearOperator):
    """A wrapper for the tensorflow MultivariateStudentTLinearOperator that
    allows it to be used in a similar fashion to other distributions.

    Attributes
    ----------
    df : int
        The degrees of freedom
    loc : np.ndarray(float)
        The means of the different dimensions experessed as a vector of postive
        floats.
    scale : np.ndarray(float)
        The scale matrix of the multivariate student T distribution. This is
        NOT the Sigma matrix, nor the Covariance matrix! Sigma is equal to scale
        @ scale.T, and Covariance matrix only applys when df > 2 and is then
        df / (df - 2) Sigma.

    Properties
    sigma : np.ndarray(float)
        The sigma matrix of the multivariate student T distribution. This is
        NOT the scale matrix! This is equal to scale @ scale.T
    """

    def __init__(self, df, loc, scale, *args, **kwargs):
        if not isinstance(scale, tf.linalg.LinearOperatorLowerTriangular):
            # Ensure that scale is treated as a lower triangular matrix
            scale = tf.linalg.LinearOperatorLowerTriangular(
                scale,
                is_positive_definite=True,
            )

        super(MultivariateStudentT, self).__init__(df, loc, scale, *args, **kwargs)


class MultivariateCauchy(MultivariateStudentT):
    """A wrapper for the tensorflow MultivariateStudentTLinearOperator that
    allows it to be used in a similar fashion to other distributions and
    specifically for a Multivariate Cauchy distribution, where the degrees of
    freedom is always 1.0

    Attributes
    ----------
    df : int
        The degrees of freedom is always 1.0 for a Cauchy distribution. This
        should not be changed. If need a different degree of freedom, then a
        Multivariate Student T distribution is needed.
    loc : np.ndarray(float)
        The means of the different dimensions experessed as a vector of postive
        floats.
    scale : np.ndarray(float)
        The scale matrix of the multivariate student T distribution. This is
        NOT the Sigma matrix, nor the Covariance matrix! Sigma is equal to scale
        @ scale.T, and Covariance matrix only applys when df > 2 and is then
        df / (df - 2) Sigma.

    Properties
    sigma : np.ndarray(float)
        The sigma matrix of the multivariate student T distribution. This is
        NOT the scale matrix! This is equal to scale @ scale.T
    """

    def __init__(self, loc, scale, *args, **kwargs):
        # TODO Should this inherit from the custom MVST class or not? isinstance
        super(MultivariateCauchy, self).__init__(1.0, loc, scale, *args, **kwargs)
