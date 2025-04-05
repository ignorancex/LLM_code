# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 09:51:07 2020

@author: Andrei
"""

from scipy import sparse
import numpy as np
from time import time
import textwrap
import os
import random

try:
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()
except ImportError:
    import tensorflow as tf
          
__PROVE_VER__  = '0.2.0.2'


###############################################################################
###############################################################################
# Begin ProVe    
###############################################################################
###############################################################################

from libraries_pub.generic_obj import LummetryObject

class ProVe(LummetryObject):
  def __init__(self, 
               log,
               default_embeds_file=None,
               name='prove_embs',
               random_seed=None,
               **kwargs
               ):    
    self.embeds = None
    self.graph = None
    self.sess = None
    self.best_score = None
    self._display_every_epochs = 1
    self.version = __PROVE_VER__
    self.log = log
    self._P("v{} initializing...".format(self.version))
    self.name = name
    self.default_embeds_file = default_embeds_file
    self.errors = []
    super().__init__()
    self.random_seed = random_seed
    return
  
  
  def starup(self):
    super().startup()
    if self.random_seed is not None:
      self._set_random_seed()
    self._maybe_load()
    return
    
    
  
  def _Pr(self, s, itr):
    if (itr % self._display_every_epochs) != 0:
      return
    if type(s) != str:
      s = str(s)
    ss = "ProVe iter {:06d}: ".format(itr) + s
    self.log.Pr(ss)
    return
  
  
  def _P(self, s, timer=None):
    if type(s) != str:
      s = str(s)
    ss = "ProVe: " + s
    elapsed = 0
    if timer == 'start':
      self._p_timer = time()
    elif timer == 'stop':
      elapsed = time() - self._p_timer
    if elapsed > 0:
      ss = ss + " ({:.1f}s)".format(elapsed)
    self.log.P(ss)
    return
  
  def _Ps(self, s, timer=None):
    if type(s) != str:
      s = str(s)
    elapsed = 0
    if timer == 'start':
      self._p_timer = time()
    elif timer == 'stop':
      elapsed = time() - self._p_timer
    if elapsed > 0:
      s = s + " ({:.1f}s)".format(elapsed)
    ss = textwrap.indent(s, "    ")
    self.log.P(ss)
    return
  
  
  def _set_random_seed(self):
    self._Ps("Setting all random seeds to {}".format(self.random_seed))
    seed = self.random_seed
    np.random.seed(seed)
    random.seed(seed)
    if tf.__version__[0] == '2':
      tf.random.set_seed(seed)
    else:
      tf.set_random_seed(seed)
    return

      
  def _maybe_load(self):
    if self.default_embeds_file is not None and os.path.isfile(self.default_embeds_file):
      self.embeds = np.load(self.default_embeds_file)
      self.embed_size = self.embeds.shape[-1]
      self.name = os.path.splitext(os.path.split(self.default_embeds_file)[-1])[0]
      self._Ps("Model '{}' {} loaded from {}.".format(
          self.name, self.embeds.shape, self.default_embeds_file))
    return
  
  def fit(self,       
          mco,
          vocab=None,
          initial_embedding_dict=None,
          embed_size=128, 
          max_cooccure=250, # our own setting - must be grid-searched
          epochs=10000,
          mu=0.0, # set to 0.1 as per Dingwall et al when using init embeds
          alpha=0.75, # as per Pennington et al.
          lr=0.05, # as per Dingwall et al
          save_epochs=100,
          max_fail_saves=5,
          save_opt_hist=True,
          in_gpu=True,
          display_every_epochs=10,
          tol=1e-4,
          tb_log_dir=None,
          tb_log_subdir=None,
          interactive_session=False,
          validation_data=None # early stopping
          ):
    
    
    if not self.DEBUG:
      tf.logging.set_verbosity(tf.logging.ERROR)
      
    if initial_embedding_dict is not None and mu == 0:
      raise ValueError("Mu must be set to a value higher than zero to ensure that previous embeds are close to new ones")
      
    self._epochs = epochs
    self._save_epochs = save_epochs
    self._save_folder = self.log.get_models_folder()
    self._save_opt_hist = save_opt_hist
    self._max_cooccure = max_cooccure
    self._tol = tol
    self._display_every_epochs = display_every_epochs
    self._tb_log_dir = tb_log_dir
    self._tb_log_subdir = tb_log_subdir
    self._interactive_session = interactive_session
    self._last_saved_file = None
    self._max_fails = max_fail_saves
    
    self._val_data = validation_data
    
    if self._val_data is not None:
      assert len(self._val_data[0]) == len(self._val_data[1])

    
    self._mu = mu
    self._alpha = alpha
    self._lr = lr

    self.embed_size = embed_size
    self.in_gpu = in_gpu

    self._P("Fitting model '{}' on MCO {}".format(self.name, mco.shape))
    self._Ps("Embedding size: {}".format(self.embed_size))
    self._Ps("Training for {} epochs".format(self._epochs))
    self._Ps("Saving in '{}'".format(self._save_folder))
    self._Ps("Full in-GPU: {}".format(self.in_gpu))

    
    if isinstance(mco, (sparse.csr_matrix, sparse.dok_matrix, sparse.coo_matrix)):
      np_mco = mco.toarray()
    else:
      if type(mco) != np.ndarray:
        raise ValueError("Input MCO must be either sparse maxtrix or ndarray")
      np_mco = mco

    self._check_dimensions(np_mco, vocab, initial_embedding_dict)
    
    weights, log_coincidence = self._initialize(np_mco)
        
    self.embeds = self._fit(
        weights=weights, 
        log_coincidence=log_coincidence,
        vocab=vocab,
        initial_embedding_dict=initial_embedding_dict
        )
      
    return self.embeds
  
  
  def _validate(self):
    if self._val_data is None:
      return 0
    org = self._val_data[0]
    pos = self._val_data[1]
    embs = self._get_embeds()
    score = 0
    for i in range(len(org)):
      idx_org = org[i]
      idx_pos = pos[i]
      dists = cosine_dists_by_idx(idx_org, embs)
      # filter self
      dists[idx_org] = np.inf 
      best = np.argmin(dists)
      if best == idx_pos:
        score += 1
    return score
    
          
  def _fit(self, weights, log_coincidence,
           vocab=None,
           initial_embedding_dict=None,
           ):
    """
    Main trining loop based on:
      - Pennington et al https://nlp.stanford.edu/pubs/glove.pdf
      - Dingwall et al https://arxiv.org/abs/1803.09901
    """


    self._create_graph_and_session(
        weights=weights, 
        log_coincidence=log_coincidence,
        vocab=vocab,
        initial_embedding_dict=initial_embedding_dict,
        )


    t0 = time()
    self._last_timings = deque(maxlen=1000)
    self.val_scores = []
    self.best_score = 0
    self._fails = 0
    for i in range(1, self._epochs+1):
      t1 = time()
      if not self.in_gpu:
        feed_dict = {
            self.weights: weights,
            self.log_coincidence: log_coincidence
            }
      else:
        feed_dict = None
        
      _, loss, stats = self.sess.run(
          [self.optimizer, self.cost, self._merged_logs],
          feed_dict=feed_dict,
          options=self.run_config,
          )

      # Keep track of losses
      if self._tb_log_dir and i % 10 == 0:
        self._log_writer.add_summary(stats)
      self.errors.append(loss)
      t2 = time()
      t_l = t2 - t1
      self._last_timings.append(t_l)
      t_lap = np.mean(self._last_timings)
      t_elapsed = t2 - t0
      t_total = t_lap * self._epochs
      t_remain = t_total - t_elapsed
      if loss < self._tol:
        # Quit early if tolerance is met
        self._Pr("stopping with loss < self.tol", i)
        break
      else:
        self._Pr("loss: {:.3f}, avgtime: {:.2f} s/itr, remain: {:.2f} hrs (elapsed: {:.2f} hrs out of total {:.2f} hrs)\t".format(                    
            loss,
            t_lap,
            t_remain / 3600,
            t_elapsed / 3600,
            t_total / 3600), i)
      
      if (i % self._save_epochs) == 0:
        score = self._validate()
        self.val_scores.append(score)
        if score >= self.best_score:
          self._fails = 0
          print()
          self._Ps("Iter {} score: {}".format(i, score))
          self.best_score = score
          self._save_status(i)
          if self._save_opt_hist:
            self._save_optimization_history()
        else:
          self._fails += 1
        if self._fails > self._max_fails:
          print()
          self._P("Stopping training at iteration {}.".format(i))
          break
                
            
    #endfor iters
#    self._cleanup()
#    self.save(self.name + str(int(i / self._save_epochs)))
    self.name = self._last_name
    return self._get_embeds()  


  
  def _create_graph_and_session(self,weights, log_coincidence,
                                vocab=None,
                                initial_embedding_dict=None,
                                ):
    if hasattr(self, 'sess') and self.sess is not None:
      self.sess.close()
      self.sess = None
    
    self._P("Creating TF1 graph & session...", timer='start')
    
    if not self._interactive_session:
      self.graph = tf.Graph()
    else:
      tf.reset_default_graph()
      self.graph = tf.get_default_graph()

    # Build the computation graph.
    with self.graph.as_default():
      self._Ps("Praparing graph...")
      self._build_graph(vocab, initial_embedding_dict)
      
      # Optimizer set-up:
      self._Ps("Praparing cost/train ops...")
      if self.in_gpu:
          self.cost = self._get_cost_function(weights, log_coincidence)
      else:
          self.cost = self._get_cost_function_with_placeholders()    
          
      self.optimizer = self._get_train_func()

      tf_var_init = tf.global_variables_initializer()

      # Set up logging for Tensorboard
      if self._tb_log_dir:
        n_subdirs = len(os.listdir(self._tb_log_dir))
        subdir = self._tb_log_subdir or str(n_subdirs + 1)
        directory = os.path.join(self._tb_log_dir, subdir)
        self._log_writer = tf.summary.FileWriter(directory, flush_secs=1)
  
      self._merged_logs = tf.summary.merge_all()
          

    if self._interactive_session:
      self._Ps("Creating interactive session...")
      self.sess = tf.InteractiveSession()
    else:
      self._Ps("Creating normal session...")
      self.sess = tf.Session(graph=self.graph)
    

    # Run initialization
    self._Ps("Initializing variables...")
      
    self.sess.run(tf_var_init)
    
    self.run_config = tf.RunOptions(report_tensor_allocations_upon_oom=True)

    self._Ps("Done graph and session prep", timer='stop')        
    return


  def _build_graph(self, vocab, initial_embedding_dict):
    """
    Builds the computatation graph based on:
      - Pennington et al https://nlp.stanford.edu/pubs/glove.pdf
      - Dingwall et al https://arxiv.org/abs/1803.09901
    
    Parameters
    ------------
    vocab : Iterable
    initial_embedding_dict : dict
    """
    # Constants
    self.ones = tf.ones([self.n_words, 1])

    # Parameters:
    if initial_embedding_dict is None:
      # Ordinary GloVe
      self.W = self._weight_init(self.n_words, self.embed_size, 'W')
      self.C = self._weight_init(self.n_words, self.embed_size, 'C')
    else:
      # This is the case where we have values to use as a
      # "warm start":
      self.n = len(next(iter(initial_embedding_dict.values())))
      W = randmatrix(len(vocab), self.embed_size)
      C = randmatrix(len(vocab), self.embed_size)
      self.original_embedding = np.zeros((len(vocab), self.embed_size))
      self.has_embedding = np.zeros(len(vocab))
      for i, w in enumerate(vocab):
        if w in initial_embedding_dict:
          self.has_embedding[i] = 1.0
          embedding = np.array(initial_embedding_dict[w])
          self.original_embedding[i] = embedding
          # Divide the original embedding into W and C,
          # plus some noise to break the symmetry that would
          # otherwise cause both gradient updates to be
          # identical.
          W[i] = 0.5 * embedding + noise(self.embed_size)
          C[i] = 0.5 * embedding + noise(self.embed_size)
      self.W = tf.Variable(W, name='W', dtype=tf.float32)
      self.C = tf.Variable(C, name='C', dtype=tf.float32)
      self.original_embedding = tf.constant(self.original_embedding,
                                            dtype=tf.float32)
      self.has_embedding = tf.constant(self.has_embedding,
                                       dtype=tf.float32)
      # This is for testing. It differs from
      # `self.original_embedding` only in that it includes the
      # random noise we added above to break the symmetry.
      self.G_start = W + C

    self.bw = self._weight_init(self.n_words, 1, 'bw')
    self.bc = self._weight_init(self.n_words, 1, 'bc')

    self.model = tf.tensordot(self.W, tf.transpose(self.C), axes=1) + \
        tf.tensordot(self.bw, tf.transpose(self.ones), axes=1) + \
        tf.tensordot(self.ones, tf.transpose(self.bc), axes=1)


  def _get_cost_function(self, weights, log_coincidence):
    """Compute the cost of the Mittens objective function using pre-defined
    non-trainable Variables loaded with weights and log(mco)

    If self.mittens = 0, this is the same as the cost of GloVe.
    """
    self.weights = tf.Variable(weights,
                               dtype=tf.float32,
                               trainable=False)
    self.log_coincidence = tf.Variable(log_coincidence,
                                       dtype=tf.float32,
                                       trainable=False)
    self.diffs = tf.subtract(self.model, self.log_coincidence)
    cost = tf.reduce_sum(
        0.5 * tf.multiply(self.weights, tf.square(self.diffs)))
    if self._mu > 0:
        self.mittens = tf.constant(self._mu, tf.float32)
        cost += self.mittens * tf.reduce_sum(
            tf.multiply(
                self.has_embedding,
                self._tf_squared_euclidean(
                    tf.add(self.W, self.C),
                    self.original_embedding)))
    tf.summary.scalar("cost", cost)
    return cost


  def _get_cost_function_with_placeholders(self):
    """Compute the cost of the Mittens objective function using
    placeholders for weights and log(mco)

    If self.mittens = 0, this is the same as the cost of GloVe.
    """
    self.weights = tf.placeholder(
        tf.float32, shape=[self.n_words, self.n_words])
    self.log_coincidence = tf.placeholder(
        tf.float32, shape=[self.n_words, self.n_words])
    
    self.diffs = tf.subtract(self.model, self.log_coincidence)
    cost = tf.reduce_sum(
        0.5 * tf.multiply(self.weights, tf.square(self.diffs)))
    if self._mittens > 0:
        self.mittens = tf.constant(self._mittens, tf.float32)
        cost += self.mittens * tf.reduce_sum(
            tf.multiply(
                self.has_embedding,
                self._tf_squared_euclidean(
                    tf.add(self.W, self.C),
                    self.original_embedding)))
    tf.summary.scalar("cost", cost)
    return cost  

        
  @staticmethod
  def _tf_squared_euclidean(X, Y):
    """Squared Euclidean distance between the rows of `X` and `Y`.
    """
    return tf.reduce_sum(tf.pow(tf.subtract(X, Y), 2), axis=1)

  def _get_train_func(self):
    """Uses Adagrad to optimize the GloVe/Mittens objective,
    as specified in the GloVe paper.
    """
    optim = tf.train.AdagradOptimizer(self._lr)
    if self.DEBUG:
        gradients = optim.compute_gradients(self.cost)
        if self.log_dir:
            for name, (g, v) in zip(['W', 'C', 'bw', 'bc'], gradients):
                tf.summary.histogram("{}_grad".format(name), g)
                tf.summary.histogram("{}_vals".format(name), v)
        tf_op = optim.apply_gradients(gradients)
    else:
        tf_op = optim.minimize(self.cost,
                               global_step=tf.train.get_or_create_global_step())
    return tf_op


  def _weight_init(self, m, n, name):
    """
    Uses the Xavier Glorot method for initializing weights. This is
    built in to TensorFlow as `tf.contrib.layers.xavier_initializer`,
    but it's nice to see all the details.
    """
    x = np.sqrt(6.0/(m+n))
    with tf.name_scope(name) as scope:
        return tf.Variable(
            tf.random_uniform(
                [m, n], minval=-x, maxval=x), name=name)
      
        
  def _check_dimensions(self, X, vocab, initial_embedding_dict):
    if vocab:
      assert X.shape[0] == len(vocab), \
          "Vocab has {} tokens, but expected {} " \
          "(since X has shape {}).".format(
              len(vocab), X.shape[0], X.shape)
    if initial_embedding_dict:
      embeddings = initial_embedding_dict.values()
      sample_len = len(random.choice(list(embeddings)))
      assert sample_len == self.n, \
          "Initial embedding contains {}-dimensional embeddings," \
          " but {}-dimensional were expected.".\
              format(sample_len, self.n)


  def _initialize(self, coincidence):
    self._Ps("Initializing weights and log(mco) from MCO: {}".format(coincidence.shape))
    self.n_words = coincidence.shape[0]
    bounded = np.minimum(coincidence, self._max_cooccure)
    weights = (bounded / float(self._max_cooccure)) ** self._alpha
    log_coincidence = log_of_array_ignoring_zeros(coincidence)
    self._Ps("Done initializing weights and log(mco).")
    return weights, log_coincidence
  
    
  def _get_embeds(self):
    return self.sess.run(tf.add(self.W, self.C))          
        

  def save(self, filename):    
    fn = os.path.join(self._save_folder, filename)
    embeds = self._get_embeds()
    try:
      np.save(fn, embeds)
      self._last_name = filename
      self._Ps("Embeddings file '{}' saved.".format(fn))
      res = fn + '.npy'
    except:
      res = None
    return res
    
  
  def _cleanup(self)  :
    if self._last_saved_file is not None:
      try:
        os.remove(self._last_saved_file)
      except:
        print()
        self._P("Could not remove '{}'".format(self._last_saved_file))
    return
  
  
  def _save_status(self, itr):
    self._cleanup()
    self._Ps("Saving status at iter {}".format(itr))
    fn = '{}_{:03d}'.format(self.name, int(itr / self._save_epochs))
    self._last_saved_file = self.save(fn)
    return
  
  
  def _save_optimization_history(self, skip=5):
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    _ = plt.figure()
    ax = plt.gca()
    ax.plot(np.arange(skip, len(self.errors)), self.errors[skip:])
    ax.set_title('Mittens loss history (skipped first {} iters)'.format(skip))
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')
#        ax.set_xscale('log')
    plt.savefig(os.path.join(self._save_folder, '{}_loss.png'.format(self.name)))
    plt.close()
 
###############################################################################
# END ProVe
###############################################################################


def log_of_array_ignoring_zeros(M):
  """Returns an array containing the logs of the nonzero
  elements of M. Zeros are left alone since log(0) isn't
  defined.

  Parameters
  ----------
  M : array-like

  Returns
  -------
  array-like
      Shape matches `M`

  """
  log_M = M.copy()
  mask = log_M > 0
  log_M[mask] = np.log(log_M[mask])
  return log_M      
      

def randmatrix(m, n, random_seed=None):
  """Creates an m x n matrix of random values drawn using
  the Xavier Glorot method."""
  val = np.sqrt(6.0 / (m + n))
  np.random.seed(random_seed)
  return np.random.uniform(-val, val, size=(m, n))

def noise(n, scale=0.01):
  """Sample zero-mean Gaussian-distributed noise.

  Parameters
  ----------
  n : int
      Number of samples to take

  scale : float
      Standard deviation of the noise.

  Returns
  -------
  np.array of size (n, )
  """
  return np.random.normal(0, scale, size=n)



def cosine_dists_by_idx(idx, embeds):
  v = embeds[idx]
  dists = np.maximum(0, 1 - embeds.dot(v) / (np.linalg.norm(v) * np.linalg.norm(embeds, axis=1)))
  return dists