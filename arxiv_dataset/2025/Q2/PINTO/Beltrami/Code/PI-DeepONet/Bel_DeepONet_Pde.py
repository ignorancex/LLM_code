import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import wandb
import warnings

from tensorflow.python.ops.numpy_ops import np_config
from utils import get_fvalues
np_config.enable_numpy_behavior()

tf.random.set_seed(1234)


class PdeModel:
    def __init__(self, inputs, outputs, get_models, loss_fn, optimizer, metrics,
                 parameters, batches=1):

        self.inputs = inputs
        self.outputs = outputs
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.batches = batches
        self.parameters = parameters

        self.train_nue = parameters['train_nue']
        self.test_nue = parameters['test_nue']

        self.inner_data = self.create_data_pipeline(inputs['xin'], inputs['yin'], inputs['tin'],
                                                    inputs['xbc_in'], inputs['ybc_in'], inputs['tbc_in'],
                                                    inputs['ubc_in'], inputs['vbc_in'], inputs['pbc_in'],
                                                    inputs['nue_d'],
                                                    batch=batches).cache()
        self.boundary_data = self.create_data_pipeline(inputs['xb'], inputs['yb'], inputs['tb'],
                                                       inputs['xbc_b'], inputs['ybc_b'], inputs['tbc_b'],
                                                       inputs['ubc_b'], inputs['vbc_b'], inputs['pbc_b'],
                                                       outputs['ub'], outputs['vb'], outputs['pb'],
                                                       batch=batches).cache()

        self.nn_model = get_models['nn_model']

        self.loss_tracker = metrics['loss']
        self.u_loss_tracker = metrics['u_loss']
        self.v_loss_tracker = metrics['v_loss']
        self.p_loss_tracker = metrics['p_loss']
        self.bound_loss_tracker = metrics['bound_loss']
        self.residual_loss_tracker = metrics['residual_loss']

    @staticmethod
    def create_data_pipeline(*args, batch):
        dataset = tf.data.Dataset.from_tensor_slices(args)
        dataset = dataset.shuffle(buffer_size=len(args[0]))
        dataset = dataset.batch(np.ceil(len(args[0]) / batch))
        return dataset

    @tf.function
    def Pde_residual(self, inner_data, training=True):

        x, y, t, xbc, ybc, tbc, ubc, vbc, pbc, nue = inner_data

        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x, y, t])
            #  forward pass
            u, v, p = self.nn_model([x, y, t,
                                     ubc, vbc, pbc], training=training)

            # first order derivative wrt x / forward pass for first order derivatives
            ux = tape.gradient(u, x)
            vx = tape.gradient(v, x)
            px = tape.gradient(p, x)

            # first order derivative wrt y / forward pass for first order derivatives
            uy = tape.gradient(u, y)
            vy = tape.gradient(v, y)
            py = tape.gradient(p, y)

            # first order derivative wrt t / forward pass for first order derivatives
            ut = tape.gradient(u, t)
            vt = tape.gradient(v, t)

        # second order derivatives wrt x
        uxx = tape.gradient(ux, x)
        vxx = tape.gradient(vx, x)

        # second order derivatives wrt y
        uyy = tape.gradient(uy, y)
        vyy = tape.gradient(vy, y)

        del tape

        # momentum equations
        fx = ut + u * ux + v * uy + px - nue * (uxx + uyy)
        fy = vt + u * vx + v * vy + py - nue * (vxx + vyy)

        # continuity equation
        div = ux + vy

        residual_loss = tf.reduce_mean(tf.square(fx)) + tf.reduce_mean(tf.square(fy)) + tf.reduce_mean(tf.square(div))

        return residual_loss

    @tf.function
    def train_step(self, bound_data, inner_data):

        xb, yb, tb, xbc, ybc, tbc, ubc, vbc, pbc, ub, vb, pb = bound_data

        with tf.GradientTape(persistent=True) as tape:
            u_pred, v_pred, p_pred = self.nn_model([xb, yb, tb,
                                                    ubc, vbc, pbc], training=True)

            u_loss = self.loss_fn(ub, u_pred)
            v_loss = self.loss_fn(vb, v_pred)
            p_loss = self.loss_fn(pb, p_pred)
            bound_loss = u_loss + v_loss + p_loss
            residual_loss = self.Pde_residual(inner_data, training=True)
            loss = bound_loss + residual_loss

        grads = tape.gradient(loss, self.nn_model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.nn_model.trainable_weights))

        self.loss_tracker.update_state(loss)
        self.u_loss_tracker.update_state(u_loss)
        self.v_loss_tracker.update_state(v_loss)
        self.p_loss_tracker.update_state(p_loss)
        self.bound_loss_tracker.update_state(bound_loss)
        self.residual_loss_tracker.update_state(residual_loss)

        return {"loss": self.loss_tracker.result(), "bound_loss": self.bound_loss_tracker.result(),
                "residual_loss": self.residual_loss_tracker.result(),
                "u_loss": self.u_loss_tracker.result(), "v_loss": self.v_loss_tracker.result(),
                'p_loss': self.p_loss_tracker.result()}

    def reset_metrics(self):
        self.loss_tracker.reset_state()
        self.bound_loss_tracker.reset_state()
        self.residual_loss_tracker.reset_state()
        self.u_loss_tracker.reset_state()
        self.v_loss_tracker.reset_state()
        self.p_loss_tracker.reset_state()

    def get_model_graph(self, log_dir, wb=False):

        keras.utils.plot_model(self.nn_model, to_file=log_dir + '_nn_model.png',
                               show_shapes=True)

        if wb:
            wandb.log({"nn_model": wandb.Image(log_dir + '_nn_model.png')})

    def run(self, epochs, log_dir, save_path=None, wb=False, verbose_freq=1000, plot_freq=10000):

        history = {"loss": [], "bound_loss": [], "residual_loss": [],
                   "u_loss": [], "v_loss": [], 'p_loss': []}
        start_time = time.time()

        self.get_model_graph(log_dir=log_dir, wb=wb)

        for epoch in range(epochs):
            self.reset_metrics()

            for j, (bound_data, inner_data) in enumerate(zip(self.boundary_data, self.inner_data)):
                logs = self.train_step(bound_data, inner_data)

            if wb:
                wandb.log(logs, step=epoch + 1)

            tae = time.time() - start_time
            for key, value in logs.items():
                history[key].append(value.numpy())
            # if (epoch + 1) % 1000 == 0:
            #     if history['loss'][-1] <= history['loss'][-999]:
                    # with warnings.catch_warnings():
                    #     warnings.simplefilter('ignore')
                    #     self.nn_model.save(save_path, save_format='tf')
            if (epoch + 1) % verbose_freq == 0:
                print(f'''Epoch:{epoch + 1}/{epochs}''')
                for key, value in logs.items():
                    # history[key].append(value.numpy())
                    print(f"{key}: {value:.4f} ", end="")
                print(f"Time: {tae / 60:.4f}min")
            if (epoch + 1) % plot_freq == 0:
                # self.get_plots(epoch + 1, log_dir=log_dir, wb=wb)
                for i in self.test_nue:
                    self.get_plots(epoch + 1, log_dir=log_dir, nue=i, wb=wb)

        odata = pd.DataFrame(history)
        odata.to_csv(path_or_buf=log_dir + 'history.csv')

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
        u_pred, v_pred, p_pred = self.nn_model.predict(inputs, batch_size=32, verbose=False)

        return u_pred, v_pred, p_pred

    def get_repeated_tensors(self, size, nue):
        xbc = np.repeat(self.inputs['xbc_in'][0:1, :], size, axis=0)
        ybc = np.repeat(self.inputs['ybc_in'][0:1, :], size, axis=0)
        tbc = np.repeat(self.inputs['tbc_in'][0:1, :], size, axis=0)
        ubc, vbc, pbc = get_fvalues(xbc, ybc, tbc, nue=nue)

        return xbc, ybc, tbc, ubc, vbc, pbc

    def get_plots(self, step, log_dir, nue, wb=False):

        xdisc = np.linspace(start=-1, stop=1., num=128)
        ydisc = np.linspace(start=-1, stop=1, num=128)

        X, Y = np.meshgrid(xdisc, ydisc)
        grid_loc = np.hstack((X.flatten()[:, None], Y.flatten()[:, None], np.ones_like(X).flatten()[:, None] * 0.5))

        xbc, ybc, tbc, ubc, vbc, pbc = self.get_repeated_tensors(size=len(grid_loc), nue=nue)

        test_data = [grid_loc[:, 0:1], grid_loc[:, 1:2], grid_loc[:, 2:], ubc, vbc, pbc]

        u_test, v_test, p_test = self.predictions(test_data)
        u_test = u_test.reshape(X.shape)
        v_test = v_test.reshape(X.shape)
        p_test = p_test.reshape(X.shape)

        u_true, v_true, p_true = get_fvalues(X=grid_loc[:, 0:1], Y=grid_loc[:, 1:2], T=grid_loc[:, 2:], nue=nue)
        u_true = u_true.reshape(X.shape)
        v_true = v_true.reshape(X.shape)
        p_true = p_true.reshape(X.shape)

        true_mag = (u_true ** 2 + v_true ** 2) ** 0.5
        pred_mag = (u_test ** 2 + v_test ** 2) ** 0.5
        er_mag = true_mag - pred_mag
        p_er = p_true - p_test

        level = np.linspace(true_mag.min(), true_mag.max(), num=7)

        fig, ax = plt.subplots(2, 3, figsize=(10, 6), sharex='col', sharey='row')
        fig.tight_layout()

        pres = ax[0][0].streamplot(X, Y, u_test, v_test, color='k',
                                   linewidth=0.5)
        pre = ax[0][0].contourf(X, Y, pred_mag, level, cmap='cool', extend='both')
        fig.colorbar(pre, ax=ax[0, 0])
        pre.cmap.set_under('yellow')
        pre.cmap.set_over('green')

        refs = ax[0][1].streamplot(X, Y, u_true, v_true, color='k',
                                   linewidth=0.5)
        ref = ax[0][1].contourf(X, Y, true_mag, level, cmap='cool', extend='both')
        fig.colorbar(ref, ax=ax[0, 1])
        er = ax[0, 2].contourf(X, Y, er_mag, cmap=plt.cm.cool)
        fig.colorbar(er, ax=ax[0, 2])

        prep = ax[1][0].contourf(X, Y, p_test, cmap='cool')
        fig.colorbar(prep, ax=ax[1][0])
        refp = ax[1][1].contourf(X, Y, p_true, cmap='cool')
        fig.colorbar(refp, ax=ax[1, 1])
        erp = ax[1][2].contourf(X, Y, p_er, cmap='cool')
        fig.colorbar(erp, ax=ax[1, 2])

        ax[0][0].title.set_text("Pred")
        ax[0][0].set_ylabel("Y")
        ax[1][0].set_xlabel("X")
        ax[0][1].title.set_text("True")
        ax[0][2].title.set_text('Error')
        ax[1][1].set_xlabel("X")
        ax[1][2].set_xlabel("X")
        plt.savefig(log_dir + str(1 / nue) + '_nue_at_' + str(step) + '_' + '.png', dpi=300)
        plt.close()
        if wb:
            wandb.log({"plot_image_" + str(1 / nue): wandb.Image(
                log_dir + str(1 / nue) + '_nue_at_' + str(step) + '_' + '.png')}, step=step)
