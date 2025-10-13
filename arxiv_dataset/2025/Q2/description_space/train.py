import numpy as np 
import tensorflow as tf 
import os, time
import data, utils
import argparse

tfkl = tf.keras.layers 
default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']



def get_args():
  parser = argparse.ArgumentParser(description="")
  parser.add_argument("--quantity_name",
    choices=['oinfo', 'tc'])
  parser.add_argument("--dataset_name",
    choices=['spins', 'sudoku', 'ngrams'])


  parser.add_argument("--optimizer_latent_codes", type=str, default='sgd',
    help='The latent codes constitute the description; there is one latent code per state per component, and these select the information to include in the description.')
  parser.add_argument("--learning_rate_latent_codes", type=float, default=1e-2)
  parser.add_argument("--bottleneck_dimensionality", type=int, default=4,
    help='This is the dimensionality of the latent code.')

  parser.add_argument("--optimizer_nce_networks", type=str, default='adam',
    help='These are the InfoNCE networks that extremize the remaining terms in the extremized quantity.')
  parser.add_argument("--learning_rate_nce_networks", type=float, default=3e-4)

  parser.add_argument("--batch_size", type=int, default=256)
  parser.add_argument("--infonce_dimensionality", type=int, default=32)
  parser.add_argument("--infonce_similarity", type=str, default='l2sq')
  parser.add_argument("--infonce_temperature", type=float, default=1.)


  parser.add_argument("--encoder_units", type=int, default=128,
    help='This, the number of layers, and the activation_fn specify the MLP architecture for every InfoNCE network.')
  parser.add_argument("--encoder_number_layers", type=int, default=2)
  parser.add_argument("--activation_fn", type=str, default='leaky_relu')
  
  parser.add_argument('--base_outdir', type=str, default='./outputs')
  parser.add_argument('--run_descriptor', type=str, default='base')

  parser.add_argument('--number_training_steps', type=int, default=3*10**4)
  parser.add_argument('--number_infonce_refinement_steps', type=int, default=5000,
    help='Update the InfoNCE networks for additional steps after the latent codes are fixed.')

  parser.add_argument('--min_max', type=int, default=2,
    help='Whether to minimize the quantity (0), maximize it (1), or do both sequentially (2).')

  parser.add_argument("--information_in_start", type=float, default=1.,
    help='The desired amount of information for the description to have about the components.')
  parser.add_argument("--information_in_end", type=float, default=1.,
    help='If different than information_in_start, ramp linearly to this during training.')
  parser.add_argument("--information_in_coefficient_start", type=float, default=1.,
    help='The coefficient on the loss term penalizing the abs difference between the desired and actual information in.')
  parser.add_argument("--information_in_coefficient_end", type=float, default=1.,
    help='If different than information_in_coefficient_start, ramp exponentially to this over the second half of training.')


  parser.add_argument('--number_components', type=int, default=4,
    help='The number of spins or the number of letters in the n-gram.  Irrelevant for Sudoku.')
  parser.add_argument('--spin_temperature', type=float, default=0.625,
    help='The temperature of the spin system.  If large, then all spins become random, and if small, only the lowest energy states have probability mass.')
  parser.add_argument('--randomized_spin_glass', type=bool, default=False,
    help='Whether to use the spin glass from the paper (Fig 1) (True), or generate a new one.')
  parser.add_argument('--ngram_dataset_filename', type=str, default='enwiki-2023-04-13.txt')
  parser.add_argument('--ngram_word_length', type=int, default=4,
    help='The length of words to grab the n-gram from.  In the paper, we used 4-grams from 4-letter and 8-letter words.')
  parser.add_argument('--ngram_start_index', type=int, default=0,
    help='The starting point for the n-gram inside a word.  For the last 4 letters of an 8-letter word, this index would be 4.')

  args = parser.parse_args()
  return args

def main():
  args = get_args()

  outdir = os.path.join(args.base_outdir, args.dataset_name, f'{args.quantity_name}, {args.run_descriptor}')
  while os.path.exists(outdir):
    outdir += '0'
  os.makedirs(outdir)
  print('*'*100)
  print(f'Created {outdir}; saving everything there.')
  print('*'*100)

  quantity_name = args.quantity_name

  infonce_dimensionality = args.infonce_dimensionality
  encoder_arch_spec = [args.encoder_units]*args.encoder_number_layers
  activation_fn = args.activation_fn
  bottleneck_dimensionality = args.bottleneck_dimensionality

  batch_size = args.batch_size

  info_in_var = tf.Variable(args.information_in_start, trainable=False)
  info_coeff = tf.Variable(args.information_in_coefficient_start, trainable=False)

  number_training_steps = args.number_training_steps


  dataset_name = args.dataset_name

  dataset_kwargs = dict(
    number_components=args.number_components,
    temperature=args.spin_temperature,
    randomized_spin_glass=args.randomized_spin_glass,
    ngram_dataset_filename=args.ngram_dataset_filename,
    ngram_word_length=args.ngram_word_length,
    ngram_start_index=args.ngram_start_index,
    )

  dataset_train, number_components, number_states, marginal_distributions = data.prepare_data(dataset_name, **dataset_kwargs)

  ############################################################################################################
  if 'oinfo' in quantity_name:
    terms = [[t] for t in range(number_components)]
    weights = [1] * number_components

    terms += [np.delete(np.arange(number_components), t) for t in range(number_components)]
    weights += [-1] * number_components

    terms += [np.arange(number_components)]
    weights += [(number_components-2)]
  elif 'tc' in quantity_name:
    terms = [[t] for t in range(number_components)]
    weights = [1] * number_components
    terms += [np.arange(number_components)]
    weights += [-1.]
  else:
    raise NotImplementedError('macro quantity not implemented, but you can manually set the terms and weights')

  minimization_quantity_spec_terms = terms

  info_ins, quantity_outs = [[], []]

  to_run = ['min', 'max']
  if args.min_max < 2:
    to_run = [to_run[args.min_max]]
  for min_max in to_run:

    if 'max' in min_max:
      minimization_quantity_spec_weights = np.float32(weights.copy())*-1.
    else:
      minimization_quantity_spec_weights = np.float32(weights.copy())
    ############################################################################################################

    encoders_embs, encoders_raw = [[], []]
    trainable_variables_pro, trainable_variables_con = [[], []]

    component_latent_vecs = []
    for component_id in range(number_components):
      latent_vector = tf.Variable(tf.random.normal((number_states, 2*bottleneck_dimensionality), mean=-1, stddev=0.1), trainable=True)
      component_latent_vecs.append(latent_vector)

    for quantity_term, quantity_weight in zip(minimization_quantity_spec_terms, minimization_quantity_spec_weights):
      if len(quantity_term) == 1:
        continue
      print(quantity_term, quantity_weight)
      encoder_raw = tf.keras.Sequential([tf.keras.Input(number_states*len(quantity_term))] + \
                                    [tfkl.Dense(number_units, activation_fn) for number_units in encoder_arch_spec] + \
                                    [tfkl.Dense(infonce_dimensionality)])
      encoders_raw.append(encoder_raw)
      encoder_embs = tf.keras.Sequential([tf.keras.Input(bottleneck_dimensionality*len(quantity_term))] + \
                                    [tfkl.Dense(number_units, activation_fn) for number_units in encoder_arch_spec] + \
                                    [tfkl.Dense(infonce_dimensionality)])
      encoders_embs.append(encoder_embs)
      
      if quantity_weight < 0:  ## maxing out the info term
        print('Added to the pro net')
        trainable_variables_pro += encoder_raw.trainable_variables
        trainable_variables_pro += encoder_embs.trainable_variables
      else:  ## minimizing the info term
        print('Added to the con net')
        trainable_variables_con += encoder_raw.trainable_variables
        trainable_variables_con += encoder_embs.trainable_variables

    opt_pro = tf.keras.optimizers.get(args.optimizer_nce_networks)
    opt_pro.learning_rate = args.learning_rate_nce_networks
    opt_con = tf.keras.optimizers.get(args.optimizer_nce_networks)
    opt_con.learning_rate = args.learning_rate_nce_networks
    opt_latents = tf.keras.optimizers.get(args.optimizer_latent_codes)
    opt_latents.learning_rate = args.learning_rate_latent_codes
    ############################################################################################################
    tf_minimization_quantity_spec_weights = tf.Variable(minimization_quantity_spec_weights, trainable=False)
    @tf.function
    def train_step(batch_data, update_latents):

      component_embs = []; component_infos = []
      batch_data_one_hot = tf.one_hot(batch_data, number_states, dtype=tf.float32)
      with tf.GradientTape(persistent=True) as tape:
        for component_id in range(number_components):
          # estimate I(Ui;Xi) and sample an embedding
          latent_vec_mus, latent_vec_logvars = tf.split(component_latent_vecs[component_id], 2, axis=-1)
          latent_vec_stddevs = tf.exp(latent_vec_logvars/2.)
          emb_mus, emb_logvars = tf.split(tf.gather(component_latent_vecs[component_id], batch_data[:, component_id], axis=0), 2, axis=-1)

          emb_stddevs = tf.exp(emb_logvars/2.)
          sampled_u_values = tf.random.normal(tf.shape(emb_mus), mean=emb_mus,
                                              stddev=tf.exp(emb_logvars/2.))
          # Expand dimensions to broadcast and compute the pairwise distances between
          # the sampled points and the centers of the conditional distributions
          sampled_u_values = tf.reshape(sampled_u_values,
            [batch_size, 1, bottleneck_dimensionality])
          latent_vec_mus = tf.reshape(latent_vec_mus, [1, number_states, bottleneck_dimensionality])
          distances_ui_muj = sampled_u_values - latent_vec_mus

          normalized_distances_ui_muj = distances_ui_muj / tf.reshape(latent_vec_stddevs, [1, number_states, bottleneck_dimensionality])
          p_ui_cond_xj = tf.exp(-tf.reduce_sum(normalized_distances_ui_muj**2, axis=-1)/2. - \
            tf.reshape(tf.reduce_sum(latent_vec_logvars, axis=-1), [1, number_states])/2.)
          normalization_factor = (2.*np.pi)**(tf.cast(bottleneck_dimensionality, tf.float32)/2.)
          p_ui_cond_xj = p_ui_cond_xj / normalization_factor

          ## Don't broadcast
          normalized_distances = (tf.reshape(sampled_u_values, tf.shape(emb_mus))-emb_mus) / emb_stddevs
          p_ui_cond_xi = tf.exp(-tf.reduce_sum(normalized_distances**2, axis=-1)/2. - \
                                tf.reduce_sum(emb_logvars, axis=-1)/2.)
          normalization_factor = (2.*np.pi)**(tf.cast(bottleneck_dimensionality, tf.float32)/2.)
          p_ui_cond_xi = p_ui_cond_xi / normalization_factor


          info = tf.reduce_mean(tf.math.log(p_ui_cond_xi/ \
                                            tf.reduce_sum(tf.reshape(marginal_distributions[component_id], [1, -1]) * p_ui_cond_xj, axis=1)))
          info = info/np.log(2)
          component_infos.append(info)
          component_embs.append(tf.cast(sampled_u_values, tf.float32))

        loss1 = info_coeff * tf.abs(tf.reduce_sum(component_infos) - info_in_var)

        loss2 = tf.zeros(1)

        component_embs = tf.concat(component_embs, 1)  ## [batch_size, number_components, bottleneck_emb_dim]
        quantity_losses = []
        quantity_id = 0
        for quantity_term, quantity_weight in zip(minimization_quantity_spec_terms, minimization_quantity_spec_weights):
          if len(quantity_term) == 1:
            ## we'll get the info directly
            continue
          enc_raw = encoders_raw[quantity_id](tf.reshape(tf.gather(batch_data_one_hot, quantity_term, axis=1), [batch_size, -1]))
          enc_embs = encoders_embs[quantity_id](tf.reshape(tf.gather(component_embs, quantity_term, axis=1), [batch_size, -1]))
          sim_mat = utils.get_scaled_similarity(enc_raw, enc_embs, args.infonce_similarity, args.infonce_temperature)
          loss_infonce = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(tf.range(batch_size),
                                                                                        sim_mat,
                                                                                        from_logits=True))
          quantity_losses.append(loss_infonce)
          loss1 -= quantity_weight*loss_infonce
          if quantity_weight > 0:
            loss2 += loss_infonce
          quantity_id += 1

      grads1 = tape.gradient(loss1, trainable_variables_pro)
      grads2 = tape.gradient(loss2, trainable_variables_con)
      grads3 = tape.gradient(loss1, component_latent_vecs)
      del tape
      opt_pro.apply_gradients(zip(grads1, trainable_variables_pro))
      opt_con.apply_gradients(zip(grads2, trainable_variables_con))
      if update_latents:
        opt_latents.apply_gradients(zip(grads3, component_latent_vecs))
      return component_infos + quantity_losses

    ############################################################################################################
    quantity_loss_series = []
    print('*'*100)
    print(f'Starting training, {number_training_steps} steps.')
    
    ct = time.time()
    update_latents = tf.constant(True, dtype=tf.bool)
    for step, batch_data in enumerate(dataset_train.batch(batch_size).take(number_training_steps+args.number_infonce_refinement_steps)):
      info_in_var.assign(args.information_in_start+float(step)/number_training_steps*(args.information_in_end-args.information_in_start))
      info_coeff.assign(np.exp(np.log(args.information_in_coefficient_start)+max(2*step/number_training_steps - 1, 0)*np.log(args.information_in_coefficient_end/args.information_in_coefficient_start)))

      if step == number_training_steps:
        update_latents = tf.constant(False, dtype=tf.bool)
      quantity_losses = train_step(batch_data, update_latents)
      quantity_loss_series.append(tf.stack(quantity_losses).numpy())
    print(f'Finished training, time taken: {(time.time()-ct)/60.:.3f} min.')
    print('*'*100)

    ############################################################################################################
    quantity_loss_series = np.stack(quantity_loss_series, 0)
    quantity_loss_series = np.concatenate([quantity_loss_series[:, :number_components], np.log2(batch_size) - quantity_loss_series[:, number_components:]/np.log(2)], -1)
    quantity_series = np.sum(quantity_loss_series*np.float32(weights)[np.newaxis], -1)
    component_info_series = quantity_loss_series[:, :number_components]
    total_component_info_series = np.sum(component_info_series, -1)

    np.savez(os.path.join(outdir, f'{dataset_name}_{quantity_name}_{min_max}.npz'),
             quantity_loss_series=quantity_loss_series,
             extremized_quantity_series=quantity_series,
             component_info_series=component_info_series,
             total_component_info_series=total_component_info_series,
             minimization_quantity_spec_terms=minimization_quantity_spec_terms,
             minimization_quantity_spec_weights=minimization_quantity_spec_weights,
             description_latent_codes=[code.numpy() for code in component_latent_vecs])

if __name__ == '__main__':
  main()
