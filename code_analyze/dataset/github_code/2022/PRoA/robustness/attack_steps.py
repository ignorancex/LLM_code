from fnmatch import translate
import torch as ch
import itertools
import kornia
import math
from . import spatial
class AttackerStep:
    '''
    Generic class for attacker steps, under perturbation constraints
    specified by an "origin input" and a perturbation magnitude.
    Must implement project, step, and random_perturb
    '''
    def __init__(self, orig_input, eps, step_size, use_grad=True):
        '''
        Initialize the attacker step with a given perturbation magnitude.

        Args:
            eps (float): the perturbation magnitude
            orig_input (ch.tensor): the original input
        '''
        self.orig_input = orig_input
        self.eps = eps
        self.step_size = step_size
        self.use_grad = use_grad

    def project(self, x):
        '''
        Given an input x, project it back into the feasible set

        Args:
            ch.tensor x : the input to project back into the feasible set.

        Returns:
            A `ch.tensor` that is the input projected back into
            the feasible set, that is,
        .. math:: \min_{x' \in S} \|x' - x\|_2
        '''
        raise NotImplementedError

    def step(self, x, g):
        '''
        Given a gradient, make the appropriate step according to the
        perturbation constraint (e.g. dual norm maximization for :math:`\ell_p` norms).

        Parameters:
            g (ch.tensor): the raw gradient

        Returns:
            The new input, a ch.tensor for the next step.
        '''
        raise NotImplementedError

    def random_perturb(self, x):
        '''
        Given a starting input, take a random step within the feasible set
        '''
        raise NotImplementedError

### Instantiations of the AttackerStep class

# L-infinity threat model
class LinfStep(AttackerStep):
    """
    Attack step for :math:`\ell_\infty` threat model. Given :math:`x_0`
    and :math:`\epsilon`, the constraint set is given by:

    .. math:: S = \{x | \|x - x_0\|_\infty \leq \epsilon\}
    """
    def project(self, x):
        """
        """
        diff = x - self.orig_input
        diff = ch.clamp(diff, -self.eps, self.eps)
        return diff + self.orig_input

    def step(self, x, g):
        """
        """
        step = ch.sign(g) * self.step_size
        return x + step

    def random_perturb(self, x):
        """
        """
        return 2 * (ch.rand_like(x) - 0.5) * self.eps

# L2 threat model
class L2Step(AttackerStep):
    """
    Attack step for :math:`\ell_\infty` threat model. Given :math:`x_0`
    and :math:`\epsilon`, the constraint set is given by:

    .. math:: S = \{x | \|x - x_0\|_2 \leq \epsilon\}
    """
    def project(self, x):
        """
        """
        diff = x - self.orig_input
        diff = diff.renorm(p=2, dim=0, maxnorm=self.eps)
        return self.orig_input + diff

    def step(self, x, g):
        """
        """
        # Scale g so that each element of the batch is at least norm 1
        g_norm = ch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
        scaled_g = g / (g_norm + 1e-10)
        return x + scaled_g * self.step_size

    def random_perturb(self, x):
        """
        """
        return (ch.rand_like(x) - 0.5).renorm(p=2, dim=1, maxnorm=self.eps)


NUM_HUE = 5
NUM_SATU = 5
NUM_BRIGHT = 5
NUM_CONT = 5
NUM_GAU = 10


class Spatial:
    def __init__(self, attack_type, rot, trans, scale):
        self.use_grad = False
        self.rot_constraint = float(rot)
        self.trans_constraint = float(trans)
        self.scale_constraint = float(scale)
        # self.trans_constraint = self.rot_constraint/4.5
        # self.scale_constraint = self.rot_constraint/150.
        self.attack_type = attack_type

    def project(self, x):
        return x

    def random_perturb(self, x):
        return x

    def step(self, x, g, correcter=None):
        assert x.shape[2] == x.shape[3]
        max_trans = self.trans_constraint
        max_rot = self.rot_constraint
        max_scale = self.scale_constraint
        bs = x.shape[0]

        device = x.get_device()
        if self.attack_type == 'random':
            rots = spatial.unif((bs,), -max_rot, max_rot)
            txs = spatial.unif((bs, 2), -max_trans, max_trans)
            scales = spatial.unif((bs, 2), 1-max_scale, 1+max_scale)
            transformed = transform(x, rots, txs, scales)
            return transformed

        assert self.attack_type == 'grid'
        assert x.shape[0] == 1

        NUM_ROT = 1 if max_rot==0 else 1000
        NUM_TRANS = 1 if max_trans==0 else 100
        NUM_SCALE = 1 if max_scale==0 else 100

        rots = ch.linspace(-max_rot, max_rot, steps=NUM_ROT)
        trans = ch.linspace(-max_trans, max_trans, steps=NUM_TRANS)
        scales = ch.linspace( 1-max_scale, 1+max_scale, steps=NUM_SCALE)

        tfms = ch.tensor(list(itertools.product(rots, trans, trans, scales, scales))).cuda(device=device)

        all_rots = tfms[:, 0]
        all_trans = tfms[:, 1:3]
        all_scales = tfms[:, 3:]
        ntfm = all_rots.shape[0]
        transformed = transform(x.repeat([ntfm, 1, 1, 1]), all_rots, all_trans, all_scales)

        i = 0
        all_losses = []
        while i < ntfm:
            to_do = transformed[i:i+MAX_BS]
            is_correct = correcter(to_do).int()
            argmin = is_correct.argmin()
            if is_correct[argmin] == 0:
                # print(is_correct[argmin])
                return transformed[i+argmin:i+argmin+1]

            i += MAX_BS
        # print(1)
        return transformed[0:1]

MAX_BS = 250
# x: [bs, 3, w, h]
# rotation: [bs]
# translation: [bs, 2]
# uses bilinear
def transform(x, rotation, translation, scale):
    # assert x.shape[1] == 3

    with ch.no_grad():
        transformed = spatial.transform(x, rotation, translation, scale)
 
    return transformed

def transform_kornia(x, hue, saturation, bright, contrast):
    # assert x.shape[1] == 3

    with ch.no_grad():
        x = kornia.enhance.adjust_hue(x, hue)
        x = kornia.enhance.adjust_saturation(x, saturation)
        x = kornia.enhance.adjust_brightness(x, bright)
        x = kornia.enhance.adjust_contrast(x, contrast)
        transformed = x
    return transformed


def blur_kornia(x, gau_size, gau_sigma1, gau_sigma2):

    with ch.no_grad():
        bs = x.shape[0] 
        if bs == 1:
            transformed = kornia.filters.gaussian_blur2d(x, (gau_size, gau_size), (gau_sigma1, gau_sigma2))
        else:
            transformed = x
            for i in range(bs):
                transformed[i,:,:,:] = kornia.filters.gaussian_blur2d(x[i, :, :,:].unsqueeze(0), (gau_size, gau_size), (gau_sigma1[i], gau_sigma2[i]))
           
    return transformed

# Unconstrained threat model
class UnconstrainedStep(AttackerStep):
    """
    Unconstrained threat model, :math:`S = \mathbb{R}^n`.
    """
    def project(self, x):
        """
        """
        return x

    def step(self, x, g):
        """
        """
        return x + g * self.step_size

    def random_perturb(self, x):
        """
        """
        return (ch.rand_like(x) - 0.5).renorm(p=2, dim=1, maxnorm=1)

class Color:
    def __init__(self, attack_type, hue, satu, bright, cont):
        self.use_grad = False
        self.hue_constraint = float(hue)
        self.satu_constraint = float(satu)
        self.bright_constraint = float(bright)
        self.cont_constraint = float(cont)
        self.attack_type = attack_type

    def project(self, x):
        return x

    def random_perturb(self, x):
        return x

    def step(self, x, g, correcter=None):
        assert x.shape[2] == x.shape[3]
        max_hue = self.hue_constraint
        max_satu = self.satu_constraint
        max_bright= self.bright_constraint
        max_cont= self.cont_constraint
        bs = x.shape[0]

        device = x.get_device()
        if self.attack_type == 'random':
            hues = spatial.unif((bs,), -max_hue, max_hue)
            satus = spatial.unif((bs,), 1-max_satu, 1+max_satu)
            brights = spatial.unif((bs,), -max_bright, max_bright)
            conts = spatial.unif((bs,), 1-max_cont, 1+max_cont)
            transformed = transform_kornia(x, hues, satus, brights, conts)
            return transformed

        assert self.attack_type == 'grid'
        assert x.shape[0] == 1

        NUM_HUE = 1 if max_hue==0 else 1000
        NUM_SATU = 1 if max_satu==0 else 1000
        NUM_BRIGHT = 1 if max_bright==0 else 100
        NUM_CONT = 1 if max_cont==0 else 100

        hues = ch.linspace(-max_hue, max_hue, steps=NUM_HUE)
        satus = ch.linspace(1-max_satu, 1+max_satu, steps=NUM_SATU)
        brights = ch.linspace(-max_bright, max_bright, steps=NUM_BRIGHT)
        conts = ch.linspace(1-max_cont, 1+max_cont, steps=NUM_CONT)
        tfms = ch.tensor(list(itertools.product(hues, satus, brights, conts))).cuda(device=device)

        all_hues = tfms[:, 0]
        all_satus = tfms[:, 1]
        all_brights = tfms[:, 2]
        all_conts = tfms[:, 3]
        ntfm = all_hues.shape[0]
        transformed = transform_kornia(x.repeat([ntfm, 1, 1, 1]), all_hues, all_satus, all_brights, all_conts)

        i = 0
        all_losses = []
        while i < ntfm:
            to_do = transformed[i:i+MAX_BS]
            is_correct = correcter(to_do).int()
            argmin = is_correct.argmin()
            if is_correct[argmin] == 0:
                return transformed[i+argmin:i+argmin+1]

            i += MAX_BS

        return transformed[0:1]

class Blur:
    def __init__(self, attack_type, gau_size, gau_sigma):
        self.use_grad = False
        self.gau_size_constraint = int(gau_size)
        self.gau_sigma_constraint = float(gau_sigma)
        self.attack_type = attack_type

    def project(self, x):
        return x

    def random_perturb(self, x):
        return x

    def step(self, x, g, correcter=None):
        assert x.shape[2] == x.shape[3]
        gau_size= self.gau_size_constraint
        max_gau_sigma= self.gau_sigma_constraint
        bs = x.shape[0]

        device = x.get_device()
        if self.attack_type == 'random':
            transformed = x.expand(bs, -1, -1, -1)
            gau_sigma1 = spatial.unif((bs, ), 0, max_gau_sigma)
            gau_sigma2 = spatial.unif((bs, ), 0, max_gau_sigma)
            transformed = blur_kornia(x, gau_size, gau_sigma1, gau_sigma2)
            return transformed

        assert self.attack_type == 'grid'
        assert x.shape[0] == 1

        NUM_SIGMA1 = 1 if max_gau_sigma==0 else 400
        NUM_SIGMA2 = 1 if max_gau_sigma==0 else 400

        gau_sigma1 = ch.linspace(0, max_gau_sigma, steps=NUM_GAU)
        gau_sigma2 = ch.linspace(0, max_gau_sigma, steps=NUM_GAU)

        tfms = ch.tensor(list(itertools.product(gau_sigma1, gau_sigma2))).cuda(device=device)

        all_sigma1s = tfms[:, 0]
        all_sigma2s = tfms[:, 1]
        ntfm = all_sigma1s.shape[0]
        transformed = blur_kornia(x.repeat([ntfm, 1, 1, 1]), gau_size, all_sigma1s, all_sigma2s)

        i = 0
        all_losses = []
        while i < ntfm:
            to_do = transformed[i:i+MAX_BS]
            is_correct = correcter(to_do).int()
            argmin = is_correct.argmin()
            if is_correct[argmin] == 0:
                return transformed[i+argmin:i+argmin+1]
            i += MAX_BS  
        return transformed[0:1]