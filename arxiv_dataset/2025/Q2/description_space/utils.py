import numpy as np 
import tensorflow as tf 

@tf.function
def pairwise_l2_distance(pts1, pts2):
  """Computes squared L2 distances between each element of each set of points.
  Args:
    pts1: [N, d] tensor of points.
    pts2: [M, d] tensor of points.
  Returns:
    distance_matrix: [N, M] tensor of distances.
  """
  norm1 = tf.reduce_sum(tf.square(pts1), axis=-1, keepdims=True)
  norm2 = tf.reduce_sum(tf.square(pts2), axis=-1)
  norm2 = tf.expand_dims(norm2, -2)
  distance_matrix = tf.maximum(
      norm1 + norm2 - 2.0 * tf.matmul(pts1, pts2, transpose_b=True), 0.0)
  return distance_matrix


@tf.function
def pairwise_l1_distance(pts1, pts2):
  """Computes L1 distances between each element of each set of points.
  Args:
    pts1: [N, d] tensor of points.
    pts2: [M, d] tensor of points.
  Returns:
    distance_matrix: [N, M] tensor of distances.
  """
  stack_size2 = pts2.shape[0]
  pts1_tiled = tf.tile(tf.expand_dims(pts1, 1), [1, stack_size2, 1])
  distance_matrix = tf.reduce_sum(tf.abs(pts1_tiled-pts2), -1)
  return distance_matrix


@tf.function
def pairwise_linf_distance(pts1, pts2):
  """Computes Chebyshev distances between each element of each set of points.
  The Chebyshev/chessboard distance is the L_infinity distance between two
  points, the maximum difference between any of their dimensions.
  Args:
    pts1: [N, d] tensor of points.
    pts2: [M, d] tensor of points.
  Returns:
    distance_matrix: [N, M] tensor of distances.
  """
  stack_size2 = pts2.shape[0]
  pts1_tiled = tf.tile(tf.expand_dims(pts1, 1), [1, stack_size2, 1])
  distance_matrix = tf.reduce_max(tf.abs(pts1_tiled-pts2), -1)
  return distance_matrix


def get_scaled_similarity(embeddings1,
                          embeddings2,
                          similarity_type,
                          temperature):
  """Returns matrix of similarities between two sets of embeddings.
  Similarity is a scalar relating two embeddings, such that a more similar pair
  of embeddings has a higher value of similarity than a less similar pair.  This
  is intentionally vague to emphasize the freedom in defining measures of
  similarity. For the similarities defined, the distance-related ones range from
  -inf to 0 and cosine similarity ranges from -1 to 1.
  Args:
    embeddings1: [N, d] float tensor of embeddings.
    embeddings2: [M, d] float tensor of embeddings.
    similarity_type: String with the method of computing similarity between
      embeddings. Implemented:
        l2sq -- Squared L2 (Euclidean) distance
        l2 -- L2 (Euclidean) distance
        l1 -- L1 (Manhattan) distance
        linf -- L_inf (Chebyshev) distance
        cosine -- Cosine similarity, the inner product of the normalized vectors
    temperature: Float value which divides all similarity values, setting a
      scale for the similarity values.  Should be positive.
  Raises:
    ValueError: If the similarity type is not recognized.
  """
  eps = 1e-9
  if similarity_type == 'l2sq':
    similarity = -1.0 * pairwise_l2_distance(embeddings1, embeddings2)
  elif similarity_type == 'l2':
    # Add a small value eps in the square root so that the gradient is always
    # with respect to a nonzero value.
    similarity = -1.0 * tf.sqrt(
        pairwise_l2_distance(embeddings1, embeddings2) + eps)
  elif similarity_type == 'l1':
    similarity = -1.0 * pairwise_l1_distance(embeddings1, embeddings2)
  elif similarity_type == 'linf':
    similarity = -1.0 * pairwise_linf_distance(embeddings1, embeddings2)
  elif similarity_type == 'cosine':
    embeddings1, _ = tf.linalg.normalize(embeddings1, ord=2, axis=-1)
    embeddings2, _ = tf.linalg.normalize(embeddings2, ord=2, axis=-1)
    similarity = tf.matmul(embeddings1, embeddings2, transpose_b=True)
  else:
    raise ValueError('Similarity type not implemented: ', similarity_type)

  similarity /= temperature
  return similarity

def compute_entropy(probability_dist):
  return -np.sum(probability_dist*np.log2(np.where(probability_dist>0, probability_dist, 1)))

def compute_total_correlation(probability_dist):
  ## take each axis as a component, and compute the sum of the marginal entropies minus the joint entropy
  running_val = -compute_entropy(probability_dist)
  for ax_ind in range(probability_dist.ndim):
    sum_axes = np.int32(range(probability_dist.ndim))
    sum_axes = np.delete(sum_axes, ax_ind)
    running_val += compute_entropy(np.sum(probability_dist, axis=tuple(sum_axes)))
  return running_val

def compute_o_info(probability_dist):
  ## Omega = (n-2) * joint entropy + sum_j^n marginal j - joint marginalized over j
  number_vars = probability_dist.ndim
  omega = (number_vars - 2) * compute_entropy(probability_dist)
  for ax_ind in range(number_vars):
    sum_axes = np.int32(range(number_vars))
    sum_axes = np.delete(sum_axes, ax_ind)
    marginal_entropy_j = compute_entropy(np.sum(probability_dist, axis=tuple(sum_axes)))
    joint_entropy_without_j = compute_entropy(np.sum(probability_dist, axis=ax_ind))
    omega += (marginal_entropy_j - joint_entropy_without_j)
  return omega

def compute_o_info_occurrences(occurrences, weights=None):
  ## occurrences is [N, dim]
  ## Omega = (n-2) * joint entropy + sum_j^n marginal j - joint marginalized over j
  number_vars = occurrences.shape[1]
  if weights is None:
    probability_dist = np.unique(occurrences, return_counts=True, axis=0)[1]
  else:
    unique_occurrences = np.unique(occurrences, axis=0)
    probability_dist = []
    for unique_occurrence in unique_occurrences:
      probability_dist.append(np.sum(weights[np.where(np.all(occurrences==unique_occurrence, axis=1))[0]]))
    probability_dist = np.array(probability_dist)
  probability_dist = probability_dist / np.sum(probability_dist)
  omega = (number_vars - 2) * compute_entropy(probability_dist)
  for ax_ind in range(number_vars):
    sum_axes = np.int32(range(number_vars))
    sum_axes = np.delete(sum_axes, ax_ind)

    if weights is None:
      probability_dist_marginal = np.unique(occurrences[:, ax_ind], return_counts=True, axis=0)[1]
    else:
      unique_occurrences = np.unique(occurrences[:, ax_ind], axis=0)
      probability_dist_marginal = []
      for unique_occurrence in unique_occurrences:
        probability_dist_marginal.append(np.sum(weights[np.where(occurrences[:, ax_ind]==unique_occurrence)[0]]))
      probability_dist_marginal = np.array(probability_dist_marginal)
    probability_dist_marginal = probability_dist_marginal / np.sum(probability_dist_marginal)
    marginal_entropy_j = compute_entropy(probability_dist_marginal)

    if weights is None:
      probability_dist_joint = np.unique(occurrences[:, sum_axes], return_counts=True, axis=0)[1]
    else:
      unique_occurrences = np.unique(occurrences[:, sum_axes], axis=0)
      probability_dist_joint = []
      for unique_occurrence in unique_occurrences:
        probability_dist_joint.append(np.sum(weights[np.where(np.all(occurrences[:, sum_axes]==unique_occurrence, axis=1))[0]]))
      probability_dist_joint = np.array(probability_dist_joint)

    probability_dist_joint = probability_dist_joint / np.sum(probability_dist_joint)
    joint_entropy_without_j = compute_entropy(probability_dist_joint)
    omega += (marginal_entropy_j - joint_entropy_without_j)
  return omega

def compute_tc_occurrences(occurrences, weights=None):
  ## occurrences is [N, dim]
  number_vars = occurrences.shape[1]
  if weights is None:
    probability_dist = np.unique(occurrences, return_counts=True, axis=0)[1]
  else:
    unique_occurrences = np.unique(occurrences, axis=0)
    probability_dist = []
    for unique_occurrence in unique_occurrences:
      probability_dist.append(np.sum(weights[np.where(np.all(occurrences==unique_occurrence, axis=1))[0]]))
    probability_dist = np.array(probability_dist)
  probability_dist = probability_dist / np.sum(probability_dist)
  joint_entropy = compute_entropy(probability_dist)
  sum_marginals = 0
  for ax_ind in range(number_vars):
    sum_axes = np.int32(range(number_vars))
    sum_axes = np.delete(sum_axes, ax_ind)

    if weights is None:
      probability_dist_marginal = np.unique(occurrences[:, ax_ind], return_counts=True, axis=0)[1]
    else:
      unique_occurrences = np.unique(occurrences[:, ax_ind], axis=0)
      probability_dist_marginal = []
      for unique_occurrence in unique_occurrences:
        probability_dist_marginal.append(np.sum(weights[np.where(occurrences[:, ax_ind]==unique_occurrence)[0]]))
      probability_dist_marginal = np.array(probability_dist_marginal)
    probability_dist_marginal = probability_dist_marginal / np.sum(probability_dist_marginal)
    marginal_entropy_j = compute_entropy(probability_dist_marginal)
    sum_marginals += marginal_entropy_j
  return sum_marginals - joint_entropy

def compute_info_occurrences(occurrences, weights=None):
  ## occurrences is [N, dim]
  number_vars = occurrences.shape[1]
  info = 0
  for ax_ind in range(number_vars):
    if weights is None:
      probability_dist_marginal = np.unique(occurrences[:, ax_ind], return_counts=True, axis=0)[1]
    else:
      unique_occurrences = np.unique(occurrences[:, ax_ind], axis=0)
      probability_dist_marginal = []
      for unique_occurrence in unique_occurrences:
        probability_dist_marginal.append(np.sum(weights[np.where(occurrences[:, ax_ind]==unique_occurrence)[0]]))
      probability_dist_marginal = np.array(probability_dist_marginal)
    probability_dist_marginal = probability_dist_marginal / np.sum(probability_dist_marginal)
    marginal_entropy_j = compute_entropy(probability_dist_marginal)
    info += marginal_entropy_j
  return info

class PositionalEncoding(tf.keras.layers.Layer):
  """Simple positional encoding layer, that appends to an input sinusoids of multiple frequencies.
  """
  def __init__(self, frequencies):
      super(PositionalEncoding, self).__init__()
      self.frequencies = frequencies

  def build(self, input_shape):
      return

  def call(self, inputs):  # Defines the computation from inputs to outputs
      return tf.concat([inputs] + [tf.math.sin(frequency*inputs) for frequency in self.frequencies], -1)


class DistributedIBNet(tf.keras.Model):
  """Distributed IB implementation where each feature is passed through an MLP
  """
  def __init__(self,
    feature_dimensionalities,
    encoder_architecture,
    integration_network_architecture,
    output_dimensionality,
    use_positional_encoding=True,
    positional_encoding_frequencies=2**np.arange(5),
    activation_fn='relu',
    feature_embedding_dimension=32,
    output_activation_fn=None,
    ):
      super(DistributedIBNet, self).__init__()
      self.feature_dimensionalities = feature_dimensionalities
      feature_encoders = []
      for feature_dimensionality in feature_dimensionalities:
        feature_encoder_layers = [tf.keras.layers.Input((feature_dimensionality,))]
        if use_positional_encoding:
          feature_encoder_layers += [PositionalEncoding(positional_encoding_frequencies)]
        feature_encoder_layers += [tf.keras.layers.Dense(number_units, activation_fn) for number_units in encoder_architecture]
        feature_encoder_layers += [tf.keras.layers.Dense(2*feature_embedding_dimension)]
        feature_encoders.append(tf.keras.Sequential(feature_encoder_layers))
      self.feature_encoders = feature_encoders

      integration_network_layers = [tf.keras.layers.Input((len(feature_dimensionalities)*feature_embedding_dimension,))]
      integration_network_layers += [tf.keras.layers.Dense(number_units, activation_fn) for number_units in integration_network_architecture]
      integration_network_layers += [tf.keras.layers.Dense(output_dimensionality, output_activation_fn)]
      self.integration_network = tf.keras.Sequential(integration_network_layers)

      self.beta = tf.Variable(1., dtype=tf.float32, trainable=False)

  def build(self, input_shape):
    assert input_shape[-1] == np.sum(self.feature_dimensionalities)
    for feature_ind in range(len(self.feature_dimensionalities)):
      self.feature_encoders[feature_ind].build(input_shape[:-1]+[self.feature_dimensionalities[feature_ind]])

    self.integration_network.build()
    return

  def call(self, inputs, training=None):  # Defines the computation from inputs to outputs
    features_split = tf.split(inputs, self.feature_dimensionalities, axis=-1)

    feature_embeddings, kl_divergence_channels = [[], []]

    for feature_ind in range(len(self.feature_dimensionalities)):
      emb_mus, emb_logvars = tf.split(self.feature_encoders[feature_ind](features_split[feature_ind]), 2, axis=-1)
      if training:
        emb_channeled = tf.random.normal(emb_mus.shape, mean=emb_mus, stddev=tf.exp(emb_logvars/2.))
      else:
        emb_channeled = emb_mus

      feature_embeddings.append(emb_channeled)
      kl_divergence_channel = tf.reduce_mean(tf.reduce_sum(0.5 * (tf.square(emb_mus) + tf.exp(emb_logvars) - emb_logvars - 1.), axis=-1))
      kl_divergence_channels.append(kl_divergence_channel)
      # self.add_metric(kl_divergence_channel, name=f'KL{feature_ind}')
      self.add_loss(kl_divergence_channel)
    # self.add_metric(self.beta, name='beta')
    prediction = self.integration_network(tf.concat(feature_embeddings, -1))
    return prediction

class InfoBottleneckAnnealingCallback(tf.keras.callbacks.Callback):
  def __init__(self,
               beta_start,
               beta_end,
               number_pretraining_steps,
               number_annealing_steps):
    super(InfoBottleneckAnnealingCallback, self).__init__()
    self.beta_start = beta_start
    self.beta_end = beta_end
    self.number_pretraining_steps = number_pretraining_steps
    self.number_annealing_steps = number_annealing_steps
  def on_epoch_begin(self, epoch, logs=None):
    self.model.beta.assign(tf.exp(tf.math.log(self.beta_start)+tf.cast(max(epoch-self.number_pretraining_steps, 0), tf.float32)/self.number_annealing_steps*(tf.math.log(self.beta_end)-tf.math.log(self.beta_start))))

class MutualInfoEstimateCallback(tf.keras.callbacks.Callback):
  def __init__(self,
               feature_index,
               x_vals,
               eval_every,
               x_weights=None):
    super().__init__()
    self.feature_index = feature_index
    self.x_vals = x_vals
    self.eval_every = eval_every
    self.x_weights = x_weights if x_weights is not None else np.ones_like(x_vals)

  def on_epoch_end(self, epoch, logs=None):
    logs = logs or {}
    if (epoch % self.eval_every) == 0:
      emb_mus, emb_logvars = tf.split(self.model.feature_encoders[self.feature_index](self.x_vals), 2, axis=-1)
      kl_mat = kl_divergence_mat(emb_mus, emb_logvars, emb_mus, emb_logvars)
      bhat_mat = bhattacharyya_dist_mat(emb_mus, emb_logvars, emb_mus, emb_logvars)
      mi_upper = estimate_MI_with_dist_mat(kl_mat, weights=self.x_weights)
      mi_lower = estimate_MI_with_dist_mat(bhat_mat, weights=self.x_weights)
      logs[f'info_upper_bound_{self.feature_index}'] = mi_upper
      logs[f'info_lower_bound_{self.feature_index}'] = mi_lower


#@title info and bhat
def estimate_infonce_sandwich_bounds_tf(encoder, dataset, eval_batch_size=256, num_eval_batches=10):

  @tf.function
  def compute_infos(batch_inps):
    mus, logvars = tf.split(encoder(batch_inps), 2, axis=-1)
    mus = tf.cast(mus, tf.float64)
    logvars = tf.cast(logvars, tf.float64)
    embedding_dimension = tf.shape(mus)[-1]
    stddevs = tf.exp(logvars/2.)
    sampled_y = tf.random.normal(mus.shape, mean=mus, stddev=stddevs, dtype=tf.float64)
    # cov is diagonal so the multivariate pdf is just the product of the univariate pdfs
    y_mu_dists = tf.reshape(sampled_y, [eval_batch_size, 1, embedding_dimension]) - tf.reshape(mus, [1, eval_batch_size, embedding_dimension])  ## we want this to be [N, N], with the first index i for the particular y across the rest of the batch
    y_mu_norm_dists = y_mu_dists / tf.reshape(stddevs, [1, eval_batch_size, embedding_dimension])
    ## Now we want to take the product over the last axis, but we can add these first and then exp
    normalization_factor = (2.*np.pi)**(tf.cast(embedding_dimension, tf.float64)/2.)
    p_y_x_conditional_pdfs = tf.exp(-tf.reduce_sum(y_mu_norm_dists**2, axis=-1)/2. - tf.reshape(tf.reduce_sum(logvars, axis=-1), [1, eval_batch_size])/2.)/normalization_factor

    ## it should be shape [eval_batch_size, eval_batch_size], where the first index is by y, and everything in that row is p(y_i|x_j)
    ## we want the diagonal terms over their rows
    matching_p_yi_xi = tf.linalg.diag_part(p_y_x_conditional_pdfs)
    avg_pyi_xj = tf.reduce_mean(p_y_x_conditional_pdfs, axis=1)
    infonce = tf.reduce_mean(tf.math.log(matching_p_yi_xi/tf.reduce_mean(p_y_x_conditional_pdfs, axis=1)))
    # Now set that diag to zero
    p_y_x_conditional_pdfs *= (1. - tf.eye(eval_batch_size, dtype=tf.float64))
    loo = tf.reduce_mean(tf.math.log(matching_p_yi_xi/tf.reduce_mean(p_y_x_conditional_pdfs, axis=1)))
    return infonce, loo

  ## num_eval_batchs * eval_batch_size can be larger than the dataset -- we gain from re-sampling y even if we have seen the x before
  running_infonce, running_loo, running_nwj, running_kl = [[], [], [], []]
  for batch_inps in dataset.repeat().shuffle(25_000).batch(eval_batch_size).take(num_eval_batches):
    infonce, loo = compute_infos(batch_inps)
    running_infonce.append(infonce)
    running_loo.append(loo)

  return np.mean(running_infonce), np.mean(running_loo)


def bhattacharyya_dist_mat(mus1, logvars1, mus2, logvars2):
  """Computes Bhattacharyya distances between multivariate Gaussians.
  Args:
    mus1: [N, d] float array of the means of the Gaussians.
    logvars1: [N, d] float array of the log variances of the Gaussians (so we're assuming diagonal
    covariance matrices; these are the logs of the diagonal).
    mus2: [M, d] float array of the means of the Gaussians.
    logvars2: [M, d] float array of the log variances of the Gaussians.
  Returns:
    [N, M] array of distances.
  """
  N = mus1.shape[0]
  M = mus2.shape[0]
  embedding_dimension = mus1.shape[1]
  assert (mus2.shape[1] == embedding_dimension)

  ## Manually broadcast in case either M or N is 1
  mus1 = np.tile(mus1[:, np.newaxis], [1, M, 1])
  logvars1 = np.tile(logvars1[:, np.newaxis], [1, M, 1])
  mus2 = np.tile(mus2[np.newaxis], [N, 1, 1])
  logvars2 = np.tile(logvars2[np.newaxis], [N, 1, 1])
  difference_mus = mus1 - mus2  # [N, M, embedding_dimension]; we want [N, M, embedding_dimension, 1]
  difference_mus = difference_mus[..., np.newaxis]
  difference_mus_T = np.transpose(difference_mus, [0, 1, 3, 2])

  sigma_diag = 0.5 * (np.exp(logvars1) + np.exp(logvars2))  ## [N, M, embedding_dimension], but we want a diag mat [N, M, embedding_dimension, embedding_dimension]
  sigma_mat = np.apply_along_axis(np.diag, -1, sigma_diag)
  sigma_mat_inv = np.apply_along_axis(np.diag, -1, 1./sigma_diag)

  determinant_sigma = np.prod(sigma_diag, axis=-1)
  determinant_sigma1 = np.exp(np.sum(logvars1, axis=-1))
  determinant_sigma2 = np.exp(np.sum(logvars2, axis=-1))
  term1 = 0.125 * (difference_mus_T @ sigma_mat_inv @ difference_mus).reshape([N, M])
  term2 = 0.5 * np.log(determinant_sigma / np.sqrt(determinant_sigma1 * determinant_sigma2))
  return term1+term2
