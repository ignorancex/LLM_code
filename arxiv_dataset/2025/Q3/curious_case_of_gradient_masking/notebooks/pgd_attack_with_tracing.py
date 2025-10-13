import numpy as np
import torch
from bcai_art.attacks import AttackBase, START_RANDOM, apply_func, get_start_delta, assert_property, project

SMALL_GRAD_EPS=0.01

class PGDAttackWithTracing(AttackBase):
    """
    A projected (multi-step) gradient descent attack.

    | Paper link: https://arxiv.org/abs/1706.06083
    """

    def __init__(self, target,
                 epsilon=0.1,
                 norm=np.inf,
                 alpha=0.01, num_iters=5, restarts=1,
                 start=START_RANDOM,
                 patch_width=0,
                 debug=False):
        """Constructor.

        :param target: an attack target class (or None)
        :param epsilon:  a maximum perturbation size (for a given norm)
        :param norm: a norm
        :param alpha: PGD step size (can be seen as a learning rate)
        :param num_iters: a number of PGD steps
        :param restarts: a number of PGD restarts
        :param start: a type of the starting point
        :param patch_width: a width of the patch for the masked PGD. if specified,
                            a PGD attack is applied only to pixels in a random square patch
                            whose side is specified by this parameter.
        """
        super().__init__(target)

        self.epsilon = epsilon
        self.norm = norm
        self.alpha = alpha
        self.num_iters = num_iters
        self.start = start
        # assert number of restarts is 1 in case of starting from the given data points
        if self.start == 'zero':
            if restarts != 1:
                raise Exception('For the non-ramdom (zero) starting point, the # of restarts must be one!')
        self.restarts = restarts

        self.masked = not (patch_width == 0)
        if debug:
            print("masked PGD: "+ str(self.masked))

        if self.masked:
            self.patch_width = patch_width
            
        self.inf_qty = 0
        self.nan_qty = 0
        self.tot_grad_qty = 0
        self.small_grad_qty = 0
        self.grad_arr_min = []
        self.grad_arr_max = []
        self.grad_arr_med = []
        self.grad_arr_mean = []
        

    def generate(self, train_env, X, y):
        """ 
        Generate and return PGD adversarial perturbations on the samples X, with random restarts.
        Among restarts, we select perturbations with the maximal loss.

        :param train_env: a training environment object
        :type  train_env: `bcai_art.TrainEvalEnviron`
        :param X: Training data.
        :type  X: `torch.tensor`
        :param y: Labels for the training data. Should be non-None even for a targeted attack.
        :type  y: `torch.tensor`

        :return generated perturbations.
        """
        max_loss = torch.zeros(y.shape[0]).to(y.device)
        max_delta = apply_func(torch.zeros_like,X)

        if self.masked:
            masks = get_rect_mask_random((self.patch_width,self.patch_width),X)
            masks = masks.to(train_env.device_name)
            masks = masks.unsqueeze(1)


        for i in range(self.restarts):
            delta = apply_func(get_start_delta, X, self.start, self.epsilon, self.norm,
                                    is_matr=train_env.is_matr(),
                                    requires_grad=False)
            if self.masked:
                delta.data = delta.data * masks
                
        for i in range(self.restarts):
            delta = apply_func(get_start_delta, X, self.start, self.epsilon, self.norm,
                                    is_matr=train_env.is_matr(),
                                    requires_grad=False)
            if self.masked:
                delta.data = delta.data * masks

            for t in range(self.num_iters):
                delta.requires_grad=True
                self.clamp_comp_loss_and_backprop(train_env, X + delta, y)
                delta_grad = delta.grad.detach()
                delta = delta.detach()
                assert_property(delta, "requires_grad", True)
                assert_property(delta_grad, "requires_grad", True)
                
                # tracing part begings
                delta_grad_flat = torch.flatten(delta_grad)
                self.tot_grad_qty += len(delta_grad_flat)
                self.nan_qty += torch.sum(torch.isnan(delta_grad_flat)).cpu()
                self.inf_qty += torch.sum(torch.isinf(delta_grad_flat)).cpu()
                self.small_grad_qty += torch.sum(torch.abs(delta_grad_flat) < SMALL_GRAD_EPS).cpu()
                self.grad_arr_min.append(torch.min(delta_grad_flat).cpu())
                self.grad_arr_max.append(torch.max(delta_grad_flat).cpu())
                self.grad_arr_med.append(torch.median(delta_grad_flat).cpu())
                self.grad_arr_mean.append(torch.mean(delta_grad_flat).cpu())
                # tracing part ends
                
                if self.masked:
                    delta_update = self.alpha * (delta_grad.sign()*masks)
                else:
                    delta_update = self.alpha * delta_grad.sign()

                delta = apply_func(project, delta + delta_update,
                                self.epsilon, self.norm,
                                is_matr=train_env.is_matr())

            all_loss = self.loss(train_env,
                                 X+delta, y,
                                 reduce_by_sum=False)

            # Select maximum-loss perturbations
            max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
            max_loss = torch.max(max_loss, all_loss)

        return max_delta             