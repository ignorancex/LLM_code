import numpy as np 
import tensorflow as tf
import os

def prepare_data(dataset_name, **dataset_kwargs):
  if dataset_name == 'spins':
    number_components = dataset_kwargs.get('number_components', 5)
    number_states = 2
    temperature = dataset_kwargs.get('temperature', 1.)
    randomized_spin_glass = dataset_kwargs.get('randomized_spin_glass', False)
    if randomized_spin_glass:

      Jij = np.random.randint(-1, 2, size=(number_components, number_components))
      Jij = np.triu(Jij)
      Jij = Jij + np.transpose(Jij)

      
    else:
      Jij = np.float32([
          [ 0, 0, 0, -1, -1],
          [ 0, 0, -1, 1, 0],
          [ 0, -1, 0, -1, 0],
          [-1, 1, -1, 0, 0],
          [-1, 0, 0, 0, 0]])

      number_components = Jij.shape[0]


    data_values = np.meshgrid(*[[0, 1]]*number_components, indexing='ij')
    data_values = np.stack(data_values, -1)
    data_values = np.reshape(data_values, [-1, number_components])
    outer_prod = np.einsum('ni,nj->nij', data_values*2-1, data_values*2-1)

    energies = np.sum(Jij*outer_prod, axis=(-1, -2)).reshape([1, -1]) / 2.
    data_logits = -energies/temperature

    joint_distribution = np.reshape(tf.nn.softmax(data_logits), [number_states]*number_components)
    marginal_distributions = [np.ones(number_states, dtype=np.float32)/number_states for _ in range(number_components)]

      
    # data_values_one_hot = tf.one_hot(data_values, number_states, dtype=tf.int32)

  elif dataset_name == 'sudoku':
    number_components = 16
    number_states = 4
    valid_boards_raw_fname = 'sudoku_4x4_valid_boards.txt'
    valid_boards = []
    with open(valid_boards_raw_fname, 'r') as f:
      for line in f.readlines():
        vals = line.strip().split()
        board = [[int(v) for v in val] for val in vals]
        valid_boards.append(np.array(board).reshape([-1]))
    valid_boards = np.stack(valid_boards, 0)
    # data_values = tf.one_hot(valid_boards-1, number_states, dtype=tf.int32)
    data_values = np.int32(valid_boards-1)
    data_logits = tf.zeros([1, valid_boards.shape[0]])
    marginal_distributions = [np.ones(number_states, dtype=np.float32)/number_states for _ in range(number_components)]

  elif dataset_name == 'ngrams':
    number_components = dataset_kwargs.get('number_components', 4)
    total_word_length = dataset_kwargs.get('ngram_word_length', 4)
    ngram_start_index = dataset_kwargs.get('ngram_start_index', 0)
    if ngram_start_index+number_components > total_word_length:
      raise ValueError(f'Cannot grab {number_components}-grams from {total_word_length}-letter words by starting at index {ngram_start_index}.')
    number_states = 26
    ngram_dataset_filename = dataset_kwargs.get('ngram_dataset_filename', 'enwiki-2023-04-13.txt')
    words, counts = [[], []]
    with open(ngram_dataset_filename, 'r') as f:
      lines = f.readlines()
      for line in lines:
        words.append(line.split()[0])
        counts.append(int(line.split()[1]))
    
    n_letter_words = [word for word, count in zip(words, counts) if len(word) == number_components]
    n_letter_counts = [count for word, count in zip(words, counts) if len(word) == number_components]

    ## they get pretty ridiculous so stop at 10k
    stop_ind = 10_000
    n_letter_words = n_letter_words[:stop_ind]
    n_letter_counts = n_letter_counts[:stop_ind]

    ## turn them into arrays of inds for the letters, and exclude those with invalid letters
    n_letter_word_vecs = []
    kept_counts = []

    for word_ind, word in enumerate(n_letter_words):
      chars_word = [ord(char)-ord('a') for char in word]
      if np.all(np.logical_and(np.array(chars_word) <= 25, np.array(chars_word)>= 0)): ## throw out any words that have a character/symbol outside of a-z
        char_fragment = chars_word[ngram_start_index:ngram_start_index+number_components]
        if char_fragment in n_letter_word_vecs:
          kept_counts[n_letter_word_vecs.index(char_fragment)] += n_letter_counts[word_ind]
        else:
          n_letter_word_vecs.append(char_fragment)
          kept_counts.append(n_letter_counts[word_ind])

    # data_values = tf.one_hot(np.array(n_letter_word_vecs), number_states, dtype=tf.int32)
    data_values = np.int32(n_letter_word_vecs)
    kept_counts = np.array(kept_counts)
    kept_counts = kept_counts / kept_counts.sum()
    data_logits = np.log(kept_counts).reshape([1, -1])

    marginal_distributions = []
    for position_ind in range(number_components):
      letter_freqs = np.bincount(data_values[:, position_ind], weights=kept_counts)
      letter_freqs = np.float32(letter_freqs/np.sum(letter_freqs))
      marginal_distributions.append(letter_freqs)

  else:
    raise NotImplementedError(f'dataset {dataset_name} not implemented')
  number_to_yield_at_once = 1024
  def yield_sample():
    for _ in range(2):
      yield tf.gather(data_values, tf.random.categorical(data_logits, number_to_yield_at_once))[0]

  dataset_train = tf.data.Dataset.from_generator(yield_sample, 
                                                 output_signature=tf.TensorSpec(shape=(number_to_yield_at_once, number_components), dtype=tf.int32)).repeat().unbatch()
  return dataset_train, number_components, number_states, marginal_distributions
