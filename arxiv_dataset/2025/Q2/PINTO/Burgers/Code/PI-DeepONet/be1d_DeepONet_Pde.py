import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import wandb
from scipy.io import loadmat
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()
tf.random.set_seed(1234)


class PdeModel:
    def __init__(self, inputs, outputs, get_models, loss_fn, optimizer, metrics,
                 parameters, batches=1, val_batches=50):

        self.inputs = inputs
        self.outputs = outputs
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.batches = batches
        self.parameters = parameters

        # Create efficient data pipelines
        self.inner_data = self.create_data_pipeline(inputs['xin'], inputs['tin'], inputs['ubc_in'],
                                                    batch=batches).cache()
        self.bound_data = self.create_data_pipeline(inputs['xb'], inputs['tb'], outputs['ub'],
                                                    inputs['ubc_b'],
                                                    batch=batches).cache()
        self.init_data = self.create_data_pipeline(inputs['x_init'], inputs['t_init'], outputs['u_init'],
                                                   inputs['ubc_init'],
                                                   batch=batches).cache()
        self.val_data = self.create_data_pipeline(inputs['xval'], inputs['tval'], outputs['uval'],
                                                  inputs['ubc_val'],
                                                  batch=val_batches).cache()

        self.nn_model = get_models['nn_model']

        self.loss_tracker = metrics['loss']
        self.bound_loss_tracker = metrics['bound_loss']
        self.residual_loss_tracker = metrics['residual_loss']
        self.init_loss_tracker = metrics['init_loss']
        self.val_loss_tracker = metrics['val_loss']
        self.val_data_loss_tracker = metrics['val_data_loss']
        self.val_residual_loss_tracker = metrics['val_res_loss']

    @staticmethod
    def create_data_pipeline(*args, batch):
        dataset = tf.data.Dataset.from_tensor_slices(args)
        dataset = dataset.shuffle(buffer_size=len(args[0]))
        dataset = dataset.batch(np.ceil(len(args[0]) / batch))
        return dataset

    @tf.function
    def Pde_residual(self, inner_data, nue, training=True):

        x, t, ubc = inner_data

        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x, t])
            #  forward pass
            u = self.nn_model([x, t, ubc], training=training)

            # first order derivative wrt x / forward pass for first order derivatives
            ux = tape.gradient(u, x)
            ut = tape.gradient(u, t)
        uxx = tape.gradient(ux, x)

        del tape

        # governing equations
        fx = ut + u * ux - (nue / np.pi) * uxx
        residual_loss = tf.square(fx)
        return residual_loss

    @tf.function
    def train_step(self, init_data, bound_data, inner_data, nue):

        xb, tb, ub, ubc = bound_data
        x_init, t_init, u_init, ubc_init = init_data

        with tf.GradientTape(persistent=True) as tape:
            ub_pred = self.nn_model([xb, tb, ubc], training=True)
            ui_pred = self.nn_model([x_init, t_init, ubc_init], training=True)

            residual_loss = tf.reduce_mean(self.Pde_residual(inner_data, nue, training=True))

            bound_loss = self.loss_fn(ub, ub_pred)
            init_loss = self.loss_fn(u_init, ui_pred)

            loss = 100 * bound_loss + residual_loss + 10 * init_loss

        grads = tape.gradient(loss, self.nn_model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.nn_model.trainable_weights))

        del tape

        self.loss_tracker.update_state(loss)
        self.bound_loss_tracker.update_state(bound_loss)
        self.residual_loss_tracker.update_state(residual_loss)
        self.init_loss_tracker.update_state(init_loss)

        return {"loss": self.loss_tracker.result(),
                "init_loss": self.init_loss_tracker.result(),
                "bound_loss": self.bound_loss_tracker.result(),
                "residual_loss": self.residual_loss_tracker.result()}

    @tf.function
    def test_step(self, val_data, nue):

        x, t, u, ubc = val_data
        upred = self.nn_model([x, t, ubc], training=False)
        val_data_loss = self.loss_fn(u, upred)
        val_res_loss = tf.reduce_mean(self.Pde_residual([x, t, ubc], nue, training=False))
        val_loss = val_data_loss + val_res_loss

        self.val_loss_tracker.update_state(val_loss)
        self.val_data_loss_tracker.update_state(val_data_loss)
        self.val_residual_loss_tracker.update_state(val_res_loss)
        return {'val_loss': self.val_loss_tracker.result(), 'val_data_loss': self.val_data_loss_tracker.result(),
                'val_res_loss': self.val_residual_loss_tracker.result()}

    def reset_metrics(self):
        self.loss_tracker.reset_state()
        self.bound_loss_tracker.reset_state()
        self.residual_loss_tracker.reset_state()
        self.init_loss_tracker.reset_state()
        self.val_loss_tracker.reset_state()
        self.val_data_loss_tracker.reset_state()
        self.val_residual_loss_tracker.reset_state()

    def get_model_graph(self, log_dir, wb=False):

        keras.utils.plot_model(self.nn_model, to_file=log_dir + '_nn_model.png',
                               show_shapes=True)

        if wb:
            wandb.log({"nn_model": wandb.Image(log_dir + '_nn_model.png')})

    def run(self, epochs, ddir, log_dir, idx_si, wb=False, verbose_freq=1000, plot_freq=10000,
            validation_freq=500):

        history = {"loss": [], "residual_loss": [], "bound_loss": [], "init_loss": []}
        # val_history = {"val_loss": [], "val_data_loss": [], "val_res_loss": []}
        start_time = time.time()

        self.get_model_graph(log_dir=log_dir, wb=wb)
        nue = self.parameters['nue']
        ts_ind = self.parameters['test_ind']
        x_sen = self.inputs['xbc_in'][0].reshape((1, -1))
        t_sen = self.inputs['tbc_in'][0].reshape((1, -1))
        # id_sen = idx_sensor
        sd = loadmat(ddir)
        sol_data = sd['output']
        xdisc = sd['x'].reshape(-1, 1)
        t_coord = sd['tspan'].reshape(-1)
        sol_i = sd['input']

        for epoch in range(epochs):

            self.reset_metrics()

            for j, (init_data, bound_data, inner_data) in enumerate(zip(
                    self.init_data, self.bound_data, self.inner_data)):
                logs = self.train_step(init_data, bound_data, inner_data, nue)

            if wb:
                wandb.log(logs, step=epoch + 1)

            if (epoch + 1) % validation_freq == 0:
                for j, val_data in enumerate(self.val_data):
                    val_logs = self.test_step(val_data, nue)
                if wb:
                    wandb.log(val_logs, step=epoch + 1)

            tae = time.time() - start_time
            for key, value in logs.items():
                history[key].append(value.numpy())
            # if (epoch + 1) % validation_freq == 0:
            #     for key, value in val_logs.items():
            #         val_history[key].append(value.numpy())
            if (epoch + 1) % verbose_freq == 0:
                print(f'''Epoch:{epoch + 1}/{epochs}''')
                for key, value in logs.items():
                    print(f"{key}: {value:.4f} ", end="")
                # if (epoch + 1) % validation_freq == 0:
                #     for key, value in val_logs.items():
                #         print(f"{key}: {value:.4f} ", end="")
                print(f"Time: {tae / 60:.4f}min")
            if (epoch + 1) % plot_freq == 0:
                for i in ts_ind:
                    u_si = sol_i[i, idx_si].reshape(1, -1)
                    u_sen = u_si
                    self.get_plots(epoch + 1, xdisc, t_coord, sol_data[i].squeeze(), sol_i[i],
                                   u_sen, log_dir=log_dir, ind=i, wb=wb)

        odata = pd.DataFrame(history)
        # val_odata = pd.DataFrame(val_history)
        odata.to_csv(path_or_buf=log_dir + 'history.csv')
        # val_odata.to_csv(path_or_buf=log_dir + 'val_history.csv')

        plt.figure()
        plt.plot(range(1, len(odata) + 1), np.log(odata['loss']))
        plt.xlabel('Epochs')
        plt.ylabel('Log_Loss')
        plt.title('log loss plot')
        plt.savefig(log_dir + '_log_loss_plt.png', dpi=300)
        if wb:
            wandb.log({"loss_plot": wandb.Image(log_dir + '_log_loss_plt.png')}, step=epochs)
        return history

    def predictions(self, inputs):
        u_pred = self.nn_model.predict(inputs, batch_size=32, verbose=False)
        return u_pred

    def get_plots(self, step, xdisc, t_coord, sol_data, sol_i, u_sen, log_dir, ind, wb=False):

        idx0 = np.where(t_coord == 0.)
        idx1 = np.where(t_coord == 0.5)
        idx2 = np.where(t_coord == 1.)

        ubc = tf.repeat(u_sen, repeats=[len(xdisc)], axis=0)

        u_pred1 = self.predictions([xdisc, tf.zeros_like(xdisc), ubc])
        u_true1 = sol_i.reshape(u_pred1.shape)
        u_pred2 = self.predictions([xdisc, 0.5*tf.ones_like(xdisc), ubc])
        u_true2 = sol_data[idx1, :].reshape(u_pred2.shape)
        u_pred3 = self.predictions([xdisc, tf.ones_like(xdisc), ubc])
        u_true3 = sol_data[idx2, :].reshape(u_pred3.shape)

        fig, ax = plt.subplots(3, 1, figsize=(9, 6))
        # fig.tight_layout()

        ax[0].plot(xdisc, u_pred1, label='Pred')
        ax[0].plot(xdisc, u_true1, label='True')
        ax[0].set_title(f"t={0}")
        ax[0].set_ylabel("U")
        ax[0].legend()
        ax[0].grid()

        ax[1].plot(xdisc, u_pred2, label='Pred')
        ax[1].plot(xdisc, u_true2, label='True')
        ax[1].set_title(f"t={1}")
        ax[1].set_ylabel("U")
        ax[1].legend()
        ax[1].grid()

        ax[2].plot(xdisc, u_pred3, label='Pred')
        ax[2].plot(xdisc, u_true3, label='True')
        ax[2].set_title(f"t={1.5}")
        ax[2].set_ylabel("U")
        ax[2].legend()
        ax[2].grid()

        ax[2].set_xlabel("X")
        plt.savefig(log_dir + 'at_' + str(ind) + '_' + str(step) + '_' + '.png', dpi=300)
        plt.close()
        if wb:
            wandb.log(
                {"plot_image_" + str(ind): wandb.Image(log_dir + 'at_' + str(ind) + '_' + str(step) + '_' + '.png')},
                step=step)
