# -*- coding: utf-8 -*-
"""
Created on Sat May  2 07:42:43 2020

@author: Andrei
"""

class EmbedsEngine():
  
  def _get_retrofitted_embeds_v2_tf(self, 
                                    dct_edges, 
                                    dct_negative=None,
                                    eager=False, 
                                    use_fit=False,
                                    epochs=99, 
                                    batch_size=16384,
                                    gpu_optim=True,
                                    lr=0.1,
                                    patience=2,
                                    tol=1e-1,
                                    **kwargs):
    """
    this method implements a similar approach to Dingwell et al
    """
    self.P("Starting `_get_retrofitted_embeds_v2_tf`...")
    import tensorflow as tf

    
    vocab_size = self.embeds.shape[0]
    embedding_dim = self.embeds.shape[1]

    data = self._prepare_retrofit_data(
        dct_positive=dct_edges,
        dct_negative=dct_negative,
        split=False,
        )
    
    self.P("  Preparing model...")
    
    nr_inputs = 4
    assert data.shape[-1] == nr_inputs

    embeds_old = tf.keras.layers.Embedding(
        vocab_size, embedding_dim, 
        embeddings_initializer=tf.keras.initializers.Constant(self.embeds),
        trainable=False,
        dtype=tf.float32,
        name='org_emb')
    embeds_new = tf.keras.layers.Embedding(
        vocab_size, embedding_dim, 
        embeddings_initializer=tf.keras.initializers.Constant(self.embeds),
        trainable=True,
        dtype=tf.float32,
        name='new_emb')
    
    src_sel = tf.keras.layers.Lambda(lambda x: x[:,0], name='inp_src')
    dst_sel = tf.keras.layers.Lambda(lambda x: x[:,1], name='inp_dst')
    wgh_p_sel = tf.keras.layers.Lambda(lambda x: x[:,2], name='inp_pres_weight')
    wgh_r_sel = tf.keras.layers.Lambda(lambda x: x[:,3], name='inp_rela_weight')
    
    p_diff = tf.keras.layers.Subtract(name='preserve_diff')
    r_diff = tf.keras.layers.Subtract(name='relation_diff')
    
    p_norm = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(tf.pow(x, 2), axis=1), name='preserve_dst')
    r_norm = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(tf.pow(x, 2), axis=1), name='relation_dst')
    
    p_weighting = tf.keras.layers.Multiply(name='preserve_weight')
    r_weighting = tf.keras.layers.Multiply(name='relation_weight')
    
    final_add = tf.keras.layers.Add(name='preserve_and_relation')
    
        
    def identity_loss(y_true, y_pred):
      return tf.math.maximum(0.0, tf.reduce_sum(y_pred))
      
       
    tf_input = tf.keras.layers.Input((nr_inputs,))
    tf_src = src_sel(tf_input)
    tf_dst = dst_sel(tf_input)
    tf_weight_p = wgh_p_sel(tf_input)
    tf_weight_r = wgh_r_sel(tf_input)
    
    tf_src_orig = embeds_old(tf_src)
    tf_src_new = embeds_new(tf_src)
    tf_dst_new = embeds_new(tf_dst)
    
    tf_preserve_diff = p_diff([tf_src_orig, tf_src_new])
    tf_relation_diff = r_diff([tf_src_new, tf_dst_new])
    
    tf_preserve_nw = p_norm(tf_preserve_diff)
    tf_relate_nw = r_norm(tf_relation_diff)
    
    tf_preserve = p_weighting([tf_preserve_nw, tf_weight_p])
    tf_relate = r_weighting([tf_relate_nw, tf_weight_r])
    
    tf_retro_loss_batch = final_add([tf_preserve, tf_relate])
    
    model = tf.keras.models.Model(tf_input, tf_retro_loss_batch)
    self.P("  Training model for {} epochs, batch={}, lr={:.1e}, tol={:.1e}".format(
        epochs, batch_size, lr, tol))
    opt = tf.keras.optimizers.SGD(lr=lr)
    losses = []
    best_loss = np.inf
    fails = 0
    last_embeds = self.embeds
    best_embeds = None
    if eager:
      def _convert(idx_slices):
        return tf.scatter_nd(tf.expand_dims(idx_slices.indices, 1),
                         idx_slices.values, idx_slices.dense_shape)
      self.P("    Starting eager training loop")
      ds = tf.data.Dataset.from_tensor_slices(data)
      n_batches = data.shape[0] // batch_size + 1
      ds = ds.batch(batch_size).prefetch(1)
      for epoch in range(1, epochs+1):
        epoch_losses = []
        for i, tf_batch in enumerate(ds):
          with tf.GradientTape() as tape:
            tf_s = src_sel(tf_batch)
            tf_d = dst_sel(tf_batch)
            tf_w_p = wgh_p_sel(tf_batch)
            tf_w_r = wgh_r_sel(tf_batch)
            
            tf_s_orig = embeds_old(tf_s)
            tf_s_new = embeds_new(tf_s)
            tf_d_new = embeds_new(tf_d)
            
            tf_p_diff = p_diff([tf_s_orig, tf_s_new])
            tf_r_diff = r_diff([tf_s_new, tf_d_new])    
            
            tf_p_nw = p_norm(tf_p_diff)
            tf_r_nw = r_norm(tf_r_diff)
            
            tf_p = p_weighting([tf_p_nw, tf_w_p]) 
            tf_r = r_weighting([tf_r_nw, tf_w_r]) 
            
            tf_retro = final_add([tf_p, tf_r])
            
            tf_loss = identity_loss(None, tf_retro)
          epoch_losses.append(tf_loss.numpy())
          grads = tape.gradient(tf_loss, model.trainable_weights)
          test = _convert(grads[0])
          opt.apply_gradients(zip(grads, model.trainable_weights))
          self.log.Pr("    Epoch {:03d} - {:.1f}% - loss: {:.2f}".format(
              epoch, i / n_batches * 100, np.mean(epoch_losses)))
        epoch_loss = np.mean(epoch_losses)
        losses.append(epoch_loss)          
        self.P("    Epoch {:03d} - loss: {:.2f}".format(epoch, epoch_loss))
        if epoch_loss < best_loss:
          best_loss = epoch_loss
          fails = 0
        else:
          fails += 1
        if fails >= patience or epoch_loss <= tol:
          self.P("    Stopping traing at epoch {}".format(epoch))
          break          
      self.P("  End eager dubug training")
      # end EAGER DEBUG training        
    else:              
      model.compile(optimizer=opt, loss=identity_loss)      
      tf.keras.utils.plot_model(
          model,
          to_file=os.path.join(self.log.get_models_folder(),'emb_retr_v2_tf.png'),
          show_shapes=True,
          show_layer_names=True,
          expand_nested=True,
          )
      if use_fit:
        model.fit(x=data, y=data, epochs=epochs, batch_size=batch_size)
        best_embeds = embeds_new.get_weights()[0]  
      else:
        ds = tf.data.Dataset.from_tensor_slices(data)
        n_batches = data.shape[0] // batch_size + 1
        ds = ds.batch(batch_size)
        if gpu_optim:
          ds = ds.apply(tf.data.experimental.copy_to_device("/gpu:0"))
          ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
          # ds = ds.
        else:
          ds = ds.prefetch(1)
        for epoch in range(1, epochs+1):
          epoch_losses = []
          t1 = time()
          b_shape = None          
          for i, tf_batch in enumerate(ds):
            if i == 0:
              b_shape = tf_batch.shape
            loss = model.train_on_batch(x=tf_batch, y=tf_batch)
            epoch_losses.append(loss)            
            self.log.Pr("    Epoch {:02d} - {:.1f}% - loss: {:.2f}".format(
                epoch, i / n_batches * 100, np.mean(epoch_losses)))
          t2 = time()
          epoch_loss = np.mean(epoch_losses)
          losses.append(epoch_loss)
          new_embeds = embeds_new.get_weights()[0]          
          if epoch_loss < best_loss:
            best_embeds = new_embeds
            best_loss = epoch_loss
            fails = 0
          else:
            fails += 1
          diff = self._measure_changes(last_embeds, new_embeds)
          self.P("    Epoch {:02d}/{} - loss: {:.2f}, change:{:.3f}, time: {:.1f}s, batch: {}, fails: {}".format(
              epoch, epochs, epoch_loss, diff, t2 - t1, b_shape, fails))
          if fails >= patience or diff <= tol:
            self.P("    Stopping traing at epoch {}".format(epoch))
            break
          last_embeds = new_embeds
        # end batch
      # end epoch
    # end else eager
    
    return best_embeds
  