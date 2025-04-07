from design_baselines.utils import spearman
from design_baselines.utils import disc_noise
from design_baselines.utils import cont_noise
from collections import defaultdict
from tensorflow_probability import distributions as tfpd
import tensorflow_probability as tfp
import tensorflow as tf


class Ensemble(tf.Module):

    def __init__(self,
                 forward_models,
                 forward_model_optim=tf.keras.optimizers.Adam,
                 forward_model_lr=0.001):
        """Build a trainer for an ensemble of probabilistic neural networks
        trained on bootstraps of a dataset

        Args:

        oracles: List[tf.keras.Model]
            a list of keras model that predict distributions over scores
        oracle_optim: __class__
            the optimizer class to use for optimizing the oracle model
        oracle__lr: float
            the learning rate for the oracle model optimizer
        """

        super().__init__()
        self.forward_models = forward_models
        self.bootstraps = len(forward_models)

        # create optimizers for each model in the ensemble
        self.forward_model_optims = [
            forward_model_optim(learning_rate=forward_model_lr)
            for i in range(self.bootstraps)]

    def get_distribution(self,
                         x,
                         **kwargs):
        """Build the mixture distribution implied by the set of oracles
        that are trained in this module

        Args:

        x: tf.Tensor
            a batch of training inputs shaped like [batch_size, channels]

        Returns:

        distribution: tfpd.Distribution
            the mixture of gaussian distributions implied by the oracles
        """

        # get the distribution parameters for all models
        params = defaultdict(list)
        for fm in self.forward_models:
            for key, val in fm.get_params(x, **kwargs).items():
                params[key].append(val)

        # stack the parameters in a new component axis
        for key, val in params.items():
            params[key] = tf.stack(val, axis=-1)

        # build the mixture distribution using the family of component one
        weights = tf.fill([self.bootstraps], 1 / self.bootstraps)
        return tfpd.MixtureSameFamily(tfpd.Categorical(
            probs=weights), self.forward_models[0].distribution(**params))


    @tf.function(experimental_relax_shapes=True)
    def train_step(self,
                   x,
                   y,
                   b,
                   len_x_data,
                   eta_lambda_,
                   epsilon,
                   list_lambda,
                   rho, 
                   r):
        """Perform a training step of gradient descent on an ensemble
        using bootstrap weights for each model in the ensemble

        Args:

        x: tf.Tensor
            a batch of training inputs shaped like [batch_size, channels]
        y: tf.Tensor
            a batch of training labels shaped like [batch_size, 1]
        b: tf.Tensor
            bootstrap indicators shaped like [batch_size, num_oracles]

        Returns:

        statistics: dict
            a dictionary that contains logging information
        """
        list_new_lambda = []
        statistics = dict()
        for i in range(self.bootstraps):
            lambda_ = list_lambda[i]
            fm = self.forward_models[i]
            fm_optim = self.forward_model_optims[i]

            with tf.GradientTape(persistent=True) as tape:
                # calculate the prediction error and accuracy of the model
                d = fm.get_distribution(x, training=True)
                nll = -d.log_prob(y)[:, 0]

                # evaluate how correct the rank fo the model predictions are
                rank_correlation = spearman(y[:, 0], d.mean()[:, 0])

                # build the total loss and weight by the bootstrap
                total_loss = tf.math.divide_no_nan(tf.reduce_sum(
                    b[:, i] * nll), tf.reduce_sum(b[:, i])) 

            original_loss_grads = tape.gradient(total_loss, fm.trainable_variables)

            # Compute gradient \nabla_\beta G_B (\beta)
            with tf.GradientTape() as tape2:
                tape2.watch(fm.trainable_variables)
                forward_model_prediction = fm.get_distribution(x, training=True).mean()
                sum_forward_model_prediction = tf.reduce_sum(forward_model_prediction)/len_x_data
            dG_beta_dbeta = tape2.gradient(sum_forward_model_prediction, fm.trainable_variables)
            
            # Norm of gradient ||\nabla_\beta G_B (\beta)||
            norm_dG_beta_dbeta = tf.linalg.global_norm(dG_beta_dbeta)

            # Perturb surrogate model \beta^
            for i in range(len(fm.trainable_variables)):
                if dG_beta_dbeta[i] is not None:
                    fm.trainable_variables[i].assign_add(r*tf.convert_to_tensor(dG_beta_dbeta[i])/norm_dG_beta_dbeta)
                    # fm.trainable_variables[i].assign_add(r*dG_beta_dbeta[i]/norm_dG_beta_dbeta)
            
            # Compute gradient \nabla_\beta G_B (\beta^)
            with tf.GradientTape() as tape3:
                tape3.watch(fm.trainable_variables)
                perturbed_model_prediction = fm.get_distribution(x, training=True).mean()
                sum_perturbed_model_prediction = tf.reduce_sum(perturbed_model_prediction)/len_x_data
            dG_beta_hat_dbeta = tape3.gradient(sum_perturbed_model_prediction, fm.trainable_variables)
            
            # Reverse surrogate model
            for i in range(len(fm.trainable_variables)):
                if dG_beta_dbeta[i] is not None:
                    fm.trainable_variables[i].assign_sub(r*tf.convert_to_tensor(dG_beta_dbeta[i])/norm_dG_beta_dbeta)
                    # fm.trainable_variables[i].assign_sub(r*dG_beta_dbeta[i]/norm_dG_beta_dbeta)
            
            # Combine the utimate gradient
            for i in range(len(fm.trainable_variables)):
                if dG_beta_dbeta[i] is not None:
                    original_loss_grads[i] = original_loss_grads[i] + lambda_*rho/r*(tf.convert_to_tensor(dG_beta_hat_dbeta[i])-tf.convert_to_tensor(dG_beta_dbeta[i]))
                    # original_loss_grads[i] = original_loss_grads[i] + lambda_*rho/r*(dG_beta_hat_dbeta[i]-dG_beta_dbeta[i])

            fm_optim.apply_gradients(zip(original_loss_grads, fm.trainable_variables))
            new_lambda = lambda_ + eta_lambda_*(rho*norm_dG_beta_dbeta - epsilon)
            list_new_lambda.append(new_lambda)
            statistics[f'oracle_{i}/train/nll'] = nll
            statistics[f'oracle_{i}/train/rank_corr'] = rank_correlation

        return statistics, list_new_lambda
    
    @tf.function(experimental_relax_shapes=True)
    def validate_step(self,
                      x,
                      y):
        """Perform a validation step on an ensemble of models
        without using bootstrapping weights

        Args:

        x: tf.Tensor
            a batch of validation inputs shaped like [batch_size, channels]
        y: tf.Tensor
            a batch of validation labels shaped like [batch_size, 1]

        Returns:

        statistics: dict
            a dictionary that contains logging information
        """

        statistics = dict()

        for i in range(self.bootstraps):
            fm = self.forward_models[i]

            # calculate the prediction error and accuracy of the model
            d = fm.get_distribution(x, training=False)
            nll = -d.log_prob(y)[:, 0]

            # evaluate how correct the rank fo the model predictions are
            rank_correlation = spearman(y[:, 0], d.mean()[:, 0])

            statistics[f'oracle_{i}/validate/nll'] = nll
            statistics[f'oracle_{i}/validate/rank_corr'] = rank_correlation

        return statistics

    def train(self,
              eta_lambda_,
              epsilon,
              list_lambda,
              rho,
              r,
              dataset):
        """Perform training using gradient descent on an ensemble
        using bootstrap weights for each model in the ensemble

        Args:

        dataset: tf.data.Dataset
            the training dataset already batched and prefetched

        Returns:

        loss_dict: dict
            a dictionary mapping names to loss values for logging
        """

        statistics = defaultdict(list)
        for x, y, b in dataset:
            len_x_data = x.shape[0]
            stat, list_new_lambda = self.train_step(x, y, b, len_x_data, eta_lambda_, epsilon, list_lambda, rho, r)
            list_lambda = list_new_lambda
            
            for name, tensor in stat.items():
                statistics[name].append(tensor)
            
        for name in statistics.keys():
            statistics[name] = tf.concat(statistics[name], axis=0)
        return statistics, list_lambda

    def validate(self,
                 dataset):
        """Perform validation on an ensemble of models without
        using bootstrapping weights

        Args:

        dataset: tf.data.Dataset
            the validation dataset already batched and prefetched

        Returns:

        loss_dict: dict
            a dictionary mapping names to loss values for logging
        """

        statistics = defaultdict(list)
        for x, y in dataset:
            for name, tensor in self.validate_step(x, y).items():
                statistics[name].append(tensor)
        for name in statistics.keys():
            statistics[name] = tf.concat(statistics[name], axis=0)
        return statistics

    def launch(self,
               eta_lambda_,
               epsilon,
               lambda_,
               rho,
               r,
               train_data,
               validate_data,
               logger,
               epochs,
               start_epoch=0):
        """Launch training and validation for the model for the specified
        number of epochs, and log statistics

        Args:

        train_data: tf.data.Dataset
            the training dataset already batched and prefetched
        validate_data: tf.data.Dataset
            the validation dataset already batched and prefetched
        logger: Logger
            an instance of the logger used for writing to tensor board
        epochs: int
            the number of epochs through the data sets to take
        """
        list_lambda = [lambda_ for i in range(self.bootstraps)]
        for e in range(start_epoch, start_epoch + epochs):
            stat, list_new_lambda = self.train(eta_lambda_, epsilon, list_lambda, rho, r, train_data)
            list_lambda = list_new_lambda
            for name, loss in stat.items():
                logger.record(name, loss, e)
            for name, loss in self.validate(validate_data).items():
                logger.record(name, loss, e)

    def get_saveables(self):
        """Collects and returns stateful objects that are serializeable
        using the tensorflow checkpoint format

        Returns:

        saveables: dict
            a dict containing stateful objects compatible with checkpoints
        """

        saveables = dict()
        for i in range(self.bootstraps):
            saveables[f'forward_model_{i}'] = self.forward_models[i]
            saveables[f'forward_model_optim_{i}'] = self.forward_model_optims[i]
        return saveables
