import numpy as np
import theano
import theano.tensor as T
import theano.sparse


x = T.matrix()
y = T.matrix()


def outer(x, y):
    return x[:, :, np.newaxis] * y[:, np.newaxis, :]


def cast32(x):
    return T.cast(x, 'float32')


class CFRBM:
    def __init__(self, num_visible_x, num_visible_o, num_visible_s, num_visible_a, num_visible_g, num_hidden, initial_v_x=None,
                initial_v_o=None, initial_v_s=None, initial_v_a=None, initial_v_g=None, initial_weights_x=None, initial_weights_o=None, 
                initial_weights_s=None, initial_weights_a=None, initial_weights_g=None, debug=False):
        self.dim_x = (num_visible_x, num_hidden)
        self.dim_o = (num_visible_o, num_hidden)
        self.dim_s = (num_visible_s, num_hidden)
        self.dim_a = (num_visible_a, num_hidden)
        self.dim_g = (num_visible_g, num_hidden)
        self.num_visible_x = num_visible_x
        self.num_visible_o = num_visible_o
        self.num_visible_s = num_visible_s
        self.num_visible_a = num_visible_a
        self.num_visible_g = num_visible_g
        self.num_hidden = num_hidden

        if initial_weights_x:
            initial_weights_x = np.load('{}.W_x.npy'.format(initial_weights_x))
            initial_weights_o = np.load('{}.W_o.npy'.format(initial_weights_o))
            initial_weights_s = np.load('{}.W_s.npy'.format(initial_weights_s))
            initial_weights_a = np.load('{}.W_a.npy'.format(initial_weights_a))
            initial_weights_g = np.load('{}.W_g.npy'.format(initial_weights_g))
            initial_hbias = np.load('{}.h.npy'.format(initial_weights_x))
            initial_vbias_x = np.load('{}.b_x.npy'.format(initial_weights_x))
            initial_vbias_o = np.load('{}.b_o.npy'.format(initial_weights_o))
            initial_vbias_s = np.load('{}.b_s.npy'.format(initial_weights_s))
            initial_vbias_a = np.load('{}.b_a.npy'.format(initial_weights_a))
            initial_vbias_g = np.load('{}.b_g.npy'.format(initial_weights_g))
        else:
            initial_weights_x = np.array(np.random.normal(0, 0.1, size=self.dim_x),
                                       dtype=np.float32)
            initial_weights_o = np.array(np.random.normal(0, 0.1, size=self.dim_o),
                                       dtype=np.float32)
            initial_weights_s = np.array(np.random.normal(0, 0.1, size=self.dim_s),
                                       dtype=np.float32)
            initial_weights_a = np.array(np.random.normal(0, 0.1, size=self.dim_a),
                                       dtype=np.float32)
            initial_weights_g = np.array(np.random.normal(0, 0.1, size=self.dim_g),
                                       dtype=np.float32)
            initial_hbias = np.zeros(num_hidden, dtype=np.float32)

            if initial_v_x:
                initial_vbias_x = np.array(initial_v_x, dtype=np.float32)
                initial_vbias_o = np.array(initial_v_o, dtype=np.float32)
                initial_vbias_s = np.array(initial_v_s, dtype=np.float32)
                initial_vbias_a = np.array(initial_v_a, dtype=np.float32)
                initial_vbias_g = np.array(initial_v_g, dtype=np.float32)
            else:
                initial_vbias_x = np.zeros(num_visible_x, dtype=np.float32)
                initial_vbias_o = np.zeros(num_visible_o, dtype=np.float32)
                initial_vbias_s = np.zeros(num_visible_s, dtype=np.float32)
                initial_vbias_a = np.zeros(num_visible_a, dtype=np.float32)
                initial_vbias_g = np.zeros(num_visible_g, dtype=np.float32)

        self.weights_x = theano.shared(value=initial_weights_x,
                                     borrow=True,
                                     name='weights_x')
        self.weights_o = theano.shared(value=initial_weights_o,
                                     borrow=True,
                                     name='weights_o')
        self.weights_s = theano.shared(value=initial_weights_s,
                                     borrow=True,
                                     name='weights_s')
        self.weights_a = theano.shared(value=initial_weights_a,
                                     borrow=True,
                                     name='weights_a')
        self.weights_g = theano.shared(value=initial_weights_g,
                                     borrow=True,
                                     name='weights_g')
        self.vbias_x = theano.shared(value=initial_vbias_x,
                                   borrow=True,
                                   name='vbias_x')
        self.vbias_o = theano.shared(value=initial_vbias_o,
                                    borrow=True,
                                    name='vbias_o')
        self.vbias_s = theano.shared(value=initial_vbias_s,
                                    borrow=True,
                                    name='vbias_s')
        self.vbias_a = theano.shared(value=initial_vbias_a,
                                    borrow=True,
                                    name='vbias_a')
        self.vbias_g = theano.shared(value=initial_vbias_g,
                                    borrow=True,
                                    name='vbias_g')
        self.hbias = theano.shared(value=initial_hbias,
                                   borrow=True,
                                   name='hbias')

        prev_gw_x = np.zeros(shape=self.dim_x, dtype=np.float32)
        self.prev_gw_x = theano.shared(value=prev_gw_x, borrow=True, name='g_w_x')

        prev_gw_o = np.zeros(shape=self.dim_o, dtype=np.float32)
        self.prev_gw_o = theano.shared(value=prev_gw_o, borrow=True, name='g_w_o')

        prev_gw_s = np.zeros(shape=self.dim_s, dtype=np.float32)
        self.prev_gw_s = theano.shared(value=prev_gw_s, borrow=True, name='g_w_s')

        prev_gw_a = np.zeros(shape=self.dim_a, dtype=np.float32)
        self.prev_gw_a = theano.shared(value=prev_gw_a, borrow=True, name='g_w_a')

        prev_gw_g = np.zeros(shape=self.dim_g, dtype=np.float32)
        self.prev_gw_g = theano.shared(value=prev_gw_g, borrow=True, name='g_w_g')

        prev_gh = np.zeros(num_hidden, dtype=np.float32)
        self.prev_gh = theano.shared(value=prev_gh, borrow=True, name='g_h')

        prev_gv_x = np.zeros(num_visible_x, dtype=np.float32)
        self.prev_gv_x = theano.shared(value=prev_gv_x, borrow=True, name='g_v_x')

        prev_gv_o = np.zeros(num_visible_o, dtype=np.float32)
        self.prev_gv_o = theano.shared(value=prev_gv_o, borrow=True, name='g_v_o')

        prev_gv_s = np.zeros(num_visible_s, dtype=np.float32)
        self.prev_gv_s = theano.shared(value=prev_gv_s, borrow=True, name='g_v_s')

        prev_gv_a = np.zeros(num_visible_a, dtype=np.float32)
        self.prev_gv_a = theano.shared(value=prev_gv_a, borrow=True, name='g_v_a')

        prev_gv_g = np.zeros(num_visible_g, dtype=np.float32)
        self.prev_gv_g = theano.shared(value=prev_gv_g, borrow=True, name='g_v_g')

        self.theano_rng = T.shared_randomstreams.RandomStreams(
            np.random.RandomState(17).randint(2**30))

        if debug:
            theano.config.compute_test_value = 'warn'
            theano.config.optimizer = 'None'
            theano.config.exception_verbosity = 'high'

    def prop_up(self, vis, occ, sex, age, genre):
        return T.nnet.sigmoid(T.dot(vis, self.weights_x) + T.dot(occ, self.weights_o) 
            + T.dot(sex, self.weights_s) + T.dot(age, self.weights_a) + T.dot(genre, self.weights_g) + self.hbias)

    def sample_hidden(self, vis, occ, sex, age, genre):
        activations = self.prop_up(vis, occ, sex, age, genre)
        h1_sample = self.theano_rng.binomial(size=activations.shape,
                                             n=1, p=activations,
                                             dtype=theano.config.floatX)
        return h1_sample, activations

    def prop_down_x(self, h):
        return T.nnet.sigmoid(T.dot(h, self.weights_x.T) + self.vbias_x)

    def prop_down_o(self, h):
        return T.nnet.sigmoid(T.dot(h, self.weights_o.T) + self.vbias_o)

    def prop_down_s(self, h):
        return T.nnet.sigmoid(T.dot(h, self.weights_s.T) + self.vbias_s)

    def prop_down_a(self, h):
        return T.nnet.sigmoid(T.dot(h, self.weights_a.T) + self.vbias_a)

    def prop_down_g(self, h):
        return T.nnet.sigmoid(T.dot(h, self.weights_g.T) + self.vbias_g)

    def sample_visible_x(self, h, k=5):
        activations = self.prop_down_x(h)
        k_ones = T.ones(k)

        partitions = \
            activations.reshape((-1, k)).sum(axis=1).reshape((-1, 1)) * k_ones

        activations_x = activations / partitions.reshape(activations.shape)
        v1_sample_x = self.theano_rng.binomial(size=activations.shape,
                                             n=1, p=activations,
                                             dtype=theano.config.floatX)

        return v1_sample_x, activations_x

    def sample_visible_o(self, h, k=1):
        activations = self.prop_down_o(h)
        k_ones = T.ones(k)

        partitions = \
            activations.reshape((-1, k)).sum(axis=1).reshape((-1, 1)) * k_ones

        activations_o = activations / partitions.reshape(activations.shape)
        v1_sample_o = self.theano_rng.binomial(size=activations.shape,
                                             n=1, p=activations,
                                             dtype=theano.config.floatX)

        return v1_sample_o, activations_o

    def sample_visible_s(self, h, k=1):
        activations = self.prop_down_s(h)
        k_ones = T.ones(k)

        partitions = \
            activations.reshape((-1, k)).sum(axis=1).reshape((-1, 1)) * k_ones

        activations_s = activations / partitions.reshape(activations.shape)
        v1_sample_s = self.theano_rng.binomial(size=activations.shape,
                                             n=1, p=activations,
                                             dtype=theano.config.floatX)

        return v1_sample_s, activations_s

    def sample_visible_a(self, h, k=1):
        activations = self.prop_down_a(h)
        k_ones = T.ones(k)

        partitions = \
            activations.reshape((-1, k)).sum(axis=1).reshape((-1, 1)) * k_ones

        activations_a = activations / partitions.reshape(activations.shape)
        v1_sample_a = self.theano_rng.binomial(size=activations.shape,
                                             n=1, p=activations,
                                             dtype=theano.config.floatX)

        return v1_sample_a, activations_a

    def sample_visible_g(self, h, k=1):
        activations = self.prop_down_g(h)
        k_ones = T.ones(k)

        partitions = \
            activations.reshape((-1, k)).sum(axis=1).reshape((-1, 1)) * k_ones

        activations_g = activations / partitions.reshape(activations.shape)
        v1_sample_g = self.theano_rng.binomial(size=activations.shape,
                                             n=1, p=activations,
                                             dtype=theano.config.floatX)

        return v1_sample_g, activations_g

    def contrastive_divergence_1(self, v1_x, v1_o, v1_s, v1_a, v1_g):
        h1, _ = self.sample_hidden(v1_x, v1_o, v1_s, v1_a, v1_g)
        v2_x, v2a_x = self.sample_visible_x(h1)
        v2_o, v2a_o = self.sample_visible_o(h1)
        v2_s, v2a_s = self.sample_visible_s(h1)
        v2_a, v2a_a = self.sample_visible_a(h1)
        v2_g, v2a_g = self.sample_visible_g(h1)
        h2, h2a = self.sample_hidden(v2_x, v2_o, v2_s, v2_a, v2_g)

        return (v1_x, v1_o, v1_s, v1_a, v1_g, h1, v2_x, v2a_x, v2_o, v2a_o, v2_s, v2a_s, v2_a, v2a_a, v2_g, v2a_g, h2, h2a)

    def gradient(self, v1_x, v1_o, v1_s, v1_a, v1_g, h1, v2_x, v2_o, v2_s, v2_a, v2_g, h2p, masks_x, masks_o, masks_s, masks_a, masks_g):
        v1_xh1_mask_x = outer(masks_x, h1)
        v1_oh1_mask_o = outer(masks_o, h1)
        v1_sh1_mask_s = outer(masks_s, h1)
        v1_ah1_mask_a = outer(masks_a, h1)
        v1_ah1_mask_g = outer(masks_g, h1)

        gw_x = ((outer(v1_x, h1) * v1_xh1_mask_x) -
              (outer(v2_x, h2p) * v1_xh1_mask_x)).mean(axis=0)
        gw_o = ((outer(v1_o, h1) * v1_oh1_mask_o) -
              (outer(v2_o, h2p) * v1_oh1_mask_o)).mean(axis=0)
        gw_s = ((outer(v1_s, h1) * v1_sh1_mask_s) -
              (outer(v2_s, h2p) * v1_sh1_mask_s)).mean(axis=0)
        gw_a = ((outer(v1_a, h1) * v1_ah1_mask_a) -
              (outer(v2_a, h2p) * v1_ah1_mask_a)).mean(axis=0)
        gw_g = ((outer(v1_g, h1) * v1_ah1_mask_g) -
              (outer(v2_g, h2p) * v1_ah1_mask_g)).mean(axis=0)
        gv_x = ((v1_x * masks_x) - (v2_x * masks_x)).mean(axis=0)
        gv_o = ((v1_o * masks_o) - (v2_o * masks_o)).mean(axis=0)
        gv_s = ((v1_s * masks_s) - (v2_s * masks_s)).mean(axis=0)
        gv_a = ((v1_a * masks_a) - (v2_a * masks_a)).mean(axis=0)
        gv_g = ((v1_g * masks_g) - (v2_g * masks_g)).mean(axis=0)
        gh = (h1 - h2p).mean(axis=0)

        return (gw_x, gw_o, gw_s, gw_a, gw_g, gv_x, gv_o, gv_s, gv_a, gv_g, gh)

    def cdk_fun(self, vis_x, vis_o, vis_s, vis_a, vis_g, masks_x, masks_o, masks_s, masks_a, masks_g, k=1, w_lr=0.000021, v_lr=0.000025,
                h_lr=0.000025, decay=0.0000, momentum=0.0):
        v1_x, v1_o, v1_s, v1_a, v1_g, h1, v2_x, v2a_x, v2_o, v2a_o, v2_s, v2a_s, v2_a, v2a_a, v2_g, v2a_g, h2, h2a = \
                  self.contrastive_divergence_1(vis_x, vis_o, vis_s, vis_a, vis_g)

        for i in range(k-1):
            v1_x, v1_o, v1_s, v1_a, v1_g, h1, v2_x, v2a_x, v2_o, v2a_o, v2_s, v2a_s, v2_a, v2a_a, v2_g, v2a_g, h2, h2a = \
                  self.contrastive_divergence_1(v2_x, v2_o, v2_s, v2_a, v2_g)

        (W_x, W_o, W_s, W_a, W_g, V_x, V_o, V_s, V_a, V_g, H) = self.gradient(v1_x, v1_o, v1_s, v1_a, v1_g, h1, 
                  v2_x, v2_o, v2_s, v2_a, v2_g, h2a, masks_x, masks_o, masks_s, masks_a, masks_g)

        if decay:
            W_x -= decay * self.weights_x
            W_o -= decay * self.weights_o
            W_s -= decay * self.weights_s
            W_a -= decay * self.weights_a
            W_g -= decay * self.weights_g

        updates = [
            (self.weights_x,
             cast32(self.weights_x + (momentum * self.prev_gw_x) +
                                   (W_x * w_lr))),
            (self.weights_o,
             cast32(self.weights_o + (momentum * self.prev_gw_o) +
                                   (W_o * w_lr))),
            (self.weights_s,
             cast32(self.weights_s + (momentum * self.prev_gw_s) +
                                   (W_s * w_lr))),
            (self.weights_a,
             cast32(self.weights_a + (momentum * self.prev_gw_a) +
                                   (W_a * w_lr))),
            (self.weights_g,
             cast32(self.weights_g + (momentum * self.prev_gw_g) +
                                   (W_g * w_lr))),
            (self.vbias_x,
             cast32(self.vbias_x + (momentum * self.prev_gv_x) +
                                 (V_x * v_lr))),
            (self.vbias_o,
             cast32(self.vbias_o + (momentum * self.prev_gv_o) +
                                 (V_o * v_lr))),
            (self.vbias_s,
             cast32(self.vbias_s + (momentum * self.prev_gv_s) +
                                 (V_s * v_lr))),
            (self.vbias_a,
             cast32(self.vbias_a + (momentum * self.prev_gv_a) +
                                 (V_a * v_lr))),
            (self.vbias_g,
             cast32(self.vbias_g + (momentum * self.prev_gv_g) +
                                 (V_g * v_lr))),
            (self.hbias,
             cast32(self.hbias + (momentum * self.prev_gh) +
                                 (H * h_lr))),
            (self.prev_gw_x, cast32(W_x)),
            (self.prev_gw_o, cast32(W_o)),
            (self.prev_gw_s, cast32(W_s)),
            (self.prev_gw_a, cast32(W_a)),
            (self.prev_gw_g, cast32(W_g)),
            (self.prev_gh, cast32(H)),
            (self.prev_gv_x, cast32(V_x)),
            (self.prev_gv_o, cast32(V_o)),
            (self.prev_gv_s, cast32(V_s)),
            (self.prev_gv_a, cast32(V_a)),
            (self.prev_gv_g, cast32(V_g))
        ]

        return theano.function([vis_x, vis_o, vis_s, vis_a, vis_g, masks_x, masks_o, masks_s, masks_a, masks_g], updates=updates)

    def predict(self, v1_x, v1_o, v1_s, v1_a, v1_g):
        h1, _ = self.sample_hidden(v1_x, v1_o, v1_s, v1_a, v1_g)
        v2_x, v2a_x = self.sample_visible_x(h1)
        #v2_g, v2a_g = self.sample_visible_g(h1)
        return theano.function([v1_x, v1_o, v1_s, v1_a, v1_g], v2a_x)

    #def get_weights(self):
    #    return self.weights_x, self.weights_g, self.vbias_x, self.vbias_g, self.hbias
