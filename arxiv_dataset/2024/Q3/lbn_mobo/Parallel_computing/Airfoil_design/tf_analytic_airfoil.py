import numpy as np
from bezier_gan import BezierGAN

import sys
sys.path.append('..')
# from utils import ElapsedTimer, create_dir, safe_remove

class airfoil_fun:
    def __init__(self):
        latent_dim = 5
        noise_dim = 10
        bezier_degree = 31
        lambda0 = 5.0
        lambda1 = 0.2
        bounds = (0., 1.)
        n_points = 192

        save_dir = './trained_gan/{}_{}'.format(lambda0, lambda1)
        self.airfoilmodel = BezierGAN(latent_dim, noise_dim, n_points, bezier_degree, bounds, lambda0, lambda1)
        self.airfoilmodel.restore(directory=save_dir)

    def airfoil_analytic(self, x_in, return_values_of=['F', 'dF', 'hF']):
        out = {}
        model = self.airfoilmodel
        # noise_np = np.random.normal(scale=0.5, size=(N, model.noise_dim))
        x = x_in
        if x_in.size == x_in.shape[0]:
            x = x_in.reshape(1, x_in.shape[0])

        noise_np = np.zeros((x.shape[0], model.noise_dim))
        if 'F' in return_values_of:
            tmp =model.sess.run([model.y_test], feed_dict={model.c: x, model.z: noise_np})[0]
            out['F'] = tmp
            if x.shape[0] == 1:
                out['F'] = tmp[0]

        if 'dF' in return_values_of:
            out['dF'] = model.sess.run([model.jac_out], feed_dict={model.c: x, model.z: noise_np})[0]

        if 'hF' in return_values_of:
            out['hF'] = model.sess.run([model.hess_out], feed_dict={model.c: x, model.z: noise_np})[0]

        if 'airfoil' in return_values_of:
            out['airfoil'] = model.sess.run([model.x_fake_test], feed_dict={model.c: x, model.z: noise_np})[0]

        # if just a single value do not return a tuple
        if len(return_values_of) == 1:
            return out[return_values_of[0]]
        else:
            return [out[val] for val in return_values_of]



# model = airfoil_fun()
# a,b,c = model.airfoil_analytic(np.random.uniform(low=0, high=1, size=(1, 5)), return_values_of=['F', 'dF', 'hF'])

