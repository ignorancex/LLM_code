import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import wandb
from scipy.io import loadmat
from scipy.interpolate import griddata
from matplotlib import ticker as tick

from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()

tf.random.set_seed(1234)


class PdeModel:
    def __init__(self, inputs, outputs, get_models, loss_fn, optimizer, metrics,
                 parameters, train_vel, test_vel, batches=1):

        self.inputs = inputs
        self.outputs = outputs
        self.loss_fn = loss_fn
        self.train_vel = train_vel
        self.test_vel = test_vel
        self.optimizer = optimizer
        self.batches = batches
        self.test_vel = test_vel
        self.parameters = parameters

        self.inner_data = self.create_data_pipeline(inputs['xin'], inputs['yin'],
                                                    inputs['xbc_in'], inputs['ybc_in'],
                                                    inputs['ubc_in'], inputs['vbc_in'], batch=batches).cache()
        self.boundary_data = self.create_data_pipeline(inputs['xb'], inputs['yb'], outputs['ub'],
                                                       outputs['vb'], inputs['xbc_b'], inputs['ybc_b'],
                                                       inputs['ubc_b'], inputs['vbc_b'], batch=batches).cache()
        self.top_data = self.create_data_pipeline(inputs['x_top'], inputs['y_top'], outputs['u_top'],
                                                  outputs['v_top'], inputs['xbc_top'], inputs['ybc_top'],
                                                  inputs['ubc_top'], inputs['vbc_top'], batch=batches).cache()

        self.nn_model = get_models['nn_model']

        self.loss_tracker = metrics['loss']
        self.u_loss_tracker = metrics['u_loss']
        self.v_loss_tracker = metrics['v_loss']
        self.bound_loss_tracker = metrics['boundary_loss']
        self.residual_loss_tracker = metrics['residual_loss']

    @staticmethod
    def create_data_pipeline(*args, batch):
        dataset = tf.data.Dataset.from_tensor_slices(args)
        dataset = dataset.shuffle(buffer_size=len(args[0]))
        dataset = dataset.batch(np.ceil(len(args[0]) / batch))
        return dataset

    @tf.function
    def Pde_residual(self, inner_data, nue, training=True):

        x, y, xbc, ybc, ubc, vbc = inner_data

        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x, y])
            #  forward pass
            u, v, p = self.nn_model([x, y,
                                     xbc, ybc,
                                     ubc], training=training)

            # first order derivative wrt x / forward pass for first order derivatives
            ux = tape.gradient(u, x)
            vx = tape.gradient(v, x)
            px = tape.gradient(p, x)

            # first order derivative wrt y / forward pass for first order derivatives
            uy = tape.gradient(u, y)
            vy = tape.gradient(v, y)
            py = tape.gradient(p, y)

        # second order derivatives wrt x
        uxx = tape.gradient(ux, x)
        vxx = tape.gradient(vx, x)

        # second order derivatives wrt y
        uyy = tape.gradient(uy, y)
        vyy = tape.gradient(vy, y)

        del tape

        # momentum equations
        fx = u * ux + v * uy + px - nue * (uxx + uyy)
        fy = u * vx + v * vy + py - nue * (vxx + vyy)

        # continuity equation
        div = ux + vy

        residual_loss = tf.reduce_mean(tf.square(fx)) + tf.reduce_mean(tf.square(fy)) + tf.reduce_mean(tf.square(div))

        return residual_loss

    @tf.function
    def train_step(self, bound_data, inner_data, top_data, nue):

        xb, yb, ub, vb, xbc_b, ybc_b, ubc_b, vbc_b = bound_data
        xb_top, yb_top, ub_top, vb_top, xbc_top, ybc_top, ubc_top, vbc_top = top_data

        with tf.GradientTape(persistent=True) as tape:

            u_pred, v_pred, _ = self.nn_model([xb, yb, xbc_b, ybc_b, ubc_b], training=True)
            ut_pred, vt_pred, _ = self.nn_model([xb_top, yb_top, xbc_top, ybc_top, ubc_top], training=True)

            u_top_loss = self.loss_fn(ub_top, ut_pred)
            u_loss = self.loss_fn(ub, u_pred)
            v_loss = self.loss_fn(vb, v_pred) + self.loss_fn(vb_top, vt_pred)

            bound_loss = 100 * u_loss + 100 * v_loss + u_top_loss
            residual_loss = self.Pde_residual(inner_data, nue, training=True)

            loss = bound_loss + residual_loss

        grads = tape.gradient(loss, self.nn_model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.nn_model.trainable_weights))

        self.loss_tracker.update_state(u_loss + u_top_loss + v_loss + residual_loss)
        self.u_loss_tracker.update_state((u_loss + u_top_loss))
        self.v_loss_tracker.update_state(v_loss)
        self.bound_loss_tracker.update_state((u_loss + u_top_loss + v_loss))
        self.residual_loss_tracker.update_state(residual_loss)

        return {"loss": self.loss_tracker.result(), "bound_loss": self.bound_loss_tracker.result(),
                "residual_loss": self.residual_loss_tracker.result(),
                "u_loss": self.u_loss_tracker.result(), "v_loss": self.v_loss_tracker.result()}

    def reset_metrics(self):
        self.loss_tracker.reset_state()
        self.bound_loss_tracker.reset_state()
        self.residual_loss_tracker.reset_state()
        self.u_loss_tracker.reset_state()
        self.v_loss_tracker.reset_state()

    def get_model_graph(self, log_dir, wb=False):

        keras.utils.plot_model(self.nn_model, to_file=log_dir + '_nn_model.png',
                               show_shapes=True)

        if wb:
            wandb.log({"nn_model": wandb.Image(log_dir + '_nn_model.png')})

    def run(self, epochs, log_dir, data_dir, param_dir, wb=False, verbose_freq=1000, plot_freq=10000):

        history = {"loss": [], "bound_loss": [], "residual_loss": [],
                   "u_loss": [], "v_loss": []}
        start_time = time.time()

        self.get_model_graph(log_dir=log_dir, wb=wb)

        for epoch in range(epochs):
            self.reset_metrics()

            for j, (bound_data, inner_data, top_data) in enumerate(zip(self.boundary_data, self.inner_data, self.top_data)):

                logs = self.train_step(bound_data, inner_data, top_data, self.parameters['nue'])

            if wb:
                wandb.log(logs, step=epoch + 1)

            tae = time.time() - start_time
            for key, value in logs.items():
                history[key].append(value.numpy())
            if (epoch + 1) % verbose_freq == 0:
                print(f'''Epoch:{epoch + 1}/{epochs}''')
                for key, value in logs.items():
                    print(f"{key}: {value:.4f} ", end="")
                print(f"Time: {tae / 60:.4f}min")
            if (epoch + 1) % plot_freq == 0:
                for i in self.test_vel:
                    self.get_plots(epoch + 1, bound_val=i,
                                   log_dir=log_dir, data_dir=data_dir[str(i)], param_dir=param_dir, wb=wb)

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

    def get_plots(self, step, bound_val, log_dir, data_dir, param_dir, wb=False):

        param_data = loadmat(param_dir)
        true_data = loadmat(data_dir)

        x_grid, y_grid = param_data['XP'].T.shape
        xmesh = np.linspace(start=0, stop=param_data['app']['a'].item()[0, 0], num=x_grid)
        ymesh = np.linspace(start=0., stop=param_data['app']['b'].item()[0, 0], num=y_grid)

        X, Y = np.meshgrid(xmesh, ymesh)
        side_bc = np.zeros((y_grid - 2, 1))
        bbc = np.zeros((1, x_grid))
        tbc = np.ones_like(bbc) * bound_val
        tbc[0, 0] = 0
        tbc[0, -1] = 0

        xbc = np.repeat(self.inputs['xbc_in'][0].reshape((1, -1)), len(X.reshape((-1, 1))), axis=0)
        ybc = np.repeat(self.inputs['ybc_in'][0].reshape((1, -1)), len(X.reshape((-1, 1))), axis=0)
        ubc = np.ones_like(xbc) * bound_val
        # vbc = np.zeros_like(xbc)
        test_data = [X.reshape((-1, 1)), Y.reshape((-1, 1)), xbc, ybc, ubc]

        u_test, v_test, p_test = self.predictions(test_data)
        u_test = u_test.reshape(X.shape)
        v_test = v_test.reshape(X.shape)

        u_interpolated_data = griddata(
            np.hstack(
                (param_data['XU'].reshape((-1, 1), order='F'), param_data['YU'].reshape((-1, 1), order='F'))),
            true_data['u'][4:], np.hstack((X[1:-1, 1:-1].reshape((-1, 1)), Y[1:-1, 1:-1].reshape((-1, 1)))))
        v_interpolated_data = griddata(
            np.hstack((param_data['XV'].reshape((-1, 1), order='F'), param_data['YV'].reshape((-1, 1), order='F'))),
            true_data['v'][4:], np.hstack((X[1:-1, 1:-1].reshape((-1, 1)), Y[1:-1, 1:-1].reshape((-1, 1)))))

        u_padded = np.concatenate((
            bbc, np.concatenate((side_bc, u_interpolated_data.reshape((x_grid - 2, y_grid - 2)), side_bc),
                                axis=1), tbc), axis=0)
        v_padded = np.concatenate(
            (bbc, np.concatenate((side_bc, v_interpolated_data.reshape((x_grid - 2, y_grid - 2)),
                                  side_bc), axis=1), bbc), axis=0)

        true_mag = (u_padded ** 2 + v_padded ** 2) ** 0.5
        pred_mag = (u_test ** 2 + v_test ** 2) ** 0.5
        err_mag = true_mag - pred_mag

        fig, ax = plt.subplots(1, 3, sharex='col', sharey='row', figsize=(9, 3),
                               gridspec_kw={'wspace': 0.3, 'hspace': 0.2,
                                            'width_ratios': [1, 1, 1], 'height_ratios': [1]})
        # fig.tight_layout()

        level = np.linspace(true_mag.min(), true_mag.max(), num=7)
        pres = ax[0].streamplot(X, Y - 1, u_test, v_test, color='k',
                                linewidth=0.5)
        pre = ax[0].contourf(X, Y - 1, pred_mag, level, cmap='cool', extend='both')
        pre.cmap.set_under('yellow')
        pre.cmap.set_over('green')

        trus = ax[1].streamplot(X, Y - 1, u_padded, v_padded, color='k',
                                linewidth=0.5)
        ref = ax[1].contourf(X, Y - 1, true_mag, level, cmap='cool', extend='both')

        pcbar = fig.colorbar(pre, ax=ax[0])
        pcbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
        pcbar.set_label('|V| (non-dim)', fontsize=10, fontweight='bold')
        tcbar = fig.colorbar(ref, ax=ax[1])
        tcbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
        tcbar.set_label('|V| (non-dim)', fontsize=10, fontweight='bold')

        ax[0].set_title('Predictions', fontsize=10, fontweight='bold')
        ax[0].set_xlim(0, 1)
        ax[0].set_ylim(-1, 0)
        ax[0].set_ylabel('Z', fontsize=10, fontweight='bold')

        ref = ax[2].contourf(X, Y - 1, err_mag, cmap='cool', extend='both')

        fig.colorbar(ref, ax=ax[2])
        ax[1].set_title('Numerical Solutions', fontsize=10, fontweight='bold')
        ax[1].set_xlim(0, 1)
        ax[1].set_ylim(-1, 0)
        ax[2].set_title('Error', fontsize=10, fontweight='bold')

        ax[1].set_xlabel('X', fontsize=10, fontweight='bold')
        ax[2].set_xlabel('X', fontsize=10, fontweight='bold')
        plt.savefig(log_dir + 'at_' + str(step) + '_' + str(bound_val) + '.png', dpi=300)
        plt.close()
        if wb:
            wandb.log({"plot_image_" + str(bound_val): wandb.Image(
                log_dir + 'at_' + str(step) + '_' + str(bound_val) + '.png')}, step=step)
