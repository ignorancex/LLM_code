import os
import sys
sys.path.append('/Project/Yi/defense')
from CODE.Utils.package import *

class FGSM:
    def __init__(self, model, eps=0.1, device=None):

        self.model = model
        self.eps = eps
        self.device = device

    def generate(self, x):

        x = x.to(self.device)
        x.requires_grad = True
        x_adv = x.clone().detach().requires_grad_(True)
        x_adv = x_adv.to(self.device)
        y_pred = self.model(x)

        y_target = self.get_y_target(x, y_pred)

        loss = nn.functional.cross_entropy(y_pred, y_target, reduction="mean")
        
        self.model.zero_grad()
        loss.backward()
        
        x_adv.data = x.data - self.eps * x.grad.sign()
        
        return x_adv

    def get_y_target(self, x, y_pred):

        with torch.no_grad():
            y_target = torch.zeros_like(y_pred)
            _, c = torch.max(y_pred, dim=1)
            for i in range(len(y_pred)):
                c_s = torch.arange(y_pred.shape[1], device=y_pred.device)
                c_s = c_s[c_s != c[i]]
                new_c = c_s[torch.randint(0, len(c_s), (1,))]
                y_target[i, new_c] = 1.0

        return y_target

class BIM:
    def __init__(self, model, eps_init=0.001, eps=0.1, beta=0.0005, num_iters=1000, device=None):
        """
        Initialize the BIM attack class.
        :param model: The model to attack.
        :param eps: Maximum perturbation amount (clip bound).
        :param beta: Scaling factor for each iteration's step size.
        :param num_iters: Number of iterations to perform.
        """
        self.model = model
        self.eps = eps
        self.beta = beta
        self.num_iters = num_iters
        self.device = device
        self.eps_init = eps_init

    def __init_r__(self, x):
        r_data = (
            torch.randint(2, x.shape, dtype=x.dtype, device=x.device) * 2 - 1
        ) * self.eps_init
        r = r_data.clone().detach().requires_grad_(True)
        r = torch.nn.Parameter(r_data, requires_grad=True).to(self.device)
        return r

    def __get_optimizer__(self, r):
        return optim.Adam([r], lr=0.001, betas=(0.9, 0.999), eps=1e-07, amsgrad=False)

    def generate(self, x):
        """
        Generate adversarial examples.
        :param x: Original input samples.
        :return: Adversarial examples.
        """
        x = x.to(self.device)
        x_adv = x.clone().detach().requires_grad_(True)
        # Forward pass
        y_pred = self.model(x)
        r = self.__init_r__(x)
        optimizer = self.__get_optimizer__(r)

        y_target = self.get_y_target(x, y_pred)
        for _ in range(self.num_iters):

            y_pred_adv = self.model(x+r)
            loss = nn.functional.cross_entropy(y_pred_adv, y_target, reduction="mean")
            print(f'{_} adversarial loss is {loss}')
            optimizer.zero_grad()
            loss.backward()
            # Here, we use the sign of the gradient for the update
            grad_sign = r.grad.sign()
            r.data = r.data - self.beta * grad_sign
            r.data = torch.clamp(r.data, -self.eps, self.eps)
            #print(self.num_iters)
        x_adv = x + r
        
        return x_adv

    def get_y_target(self, x, y_pred):
    
        with torch.no_grad():
            y_target = torch.zeros_like(y_pred)
            _, c = torch.max(y_pred, dim=1)
            for i in range(len(y_pred)):
                c_s = torch.arange(y_pred.shape[1], device=y_pred.device)
                c_s = c_s[c_s != c[i]]
                new_c = c_s[torch.randint(0, len(c_s), (1,))]
                y_target[i, new_c] = 1.0

        return y_target
        
class GM:
    def __init__(self, model, eps_init=0.001, eps=0.1, num_iters=1000, device=None):
        """
        Initialize the GM attack class.
        :param model: The model to attack.
        :param eps: Maximum perturbation amount (clip bound).
        :param beta: Scaling factor for each iteration's step size.
        :param num_iters: Number of iterations to perform.
        """
        self.model = model
        self.eps = eps
        self.num_iters = num_iters
        self.eps_init = eps_init
        self.device = device
    
    def __init_r__(self, x):
        r_data = (
            torch.randint(2, x.shape, dtype=x.dtype, device=x.device) * 2 - 1
        ) * self.eps_init
        r = r_data.clone().detach().requires_grad_(True)
        r = torch.nn.Parameter(r_data, requires_grad=True).to(self.device)
        return r

    def __get_optimizer__(self, r):
        return optim.Adam([r], lr=0.001, betas=(0.9, 0.999), eps=1e-07, amsgrad=False)

    def generate(self, x):
        x = x.to(self.device)  # Move x to the device first
        y_pred = self.model(x)
        r = self.__init_r__(x)
        optimizer = self.__get_optimizer__(r)
        y_target = self.get_y_target(x, y_pred)

        for epoch in range(self.num_iters):
            y_pred_adv = self.model(x+r)
            loss = nn.functional.cross_entropy(y_pred_adv, y_target, reduction="mean")
            print(f'{epoch} adversarial loss is {loss}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            r.data = torch.clamp(r.data, -self.eps, self.eps)

        x_adv = x + r
        #y_adv = self.f(x_adv).argmax(1)

        return x_adv

    def get_y_target(self, x, y_pred):

        with torch.no_grad():
            y_target = torch.zeros_like(y_pred)
            _, c = torch.max(y_pred, dim=1)
            for i in range(len(y_pred)):
                c_s = torch.arange(y_pred.shape[1], device=y_pred.device)
                c_s = c_s[c_s != c[i]]
                new_c = c_s[torch.randint(0, len(c_s), (1,))]
                y_target[i, new_c] = 1.0

        return y_target

class GM_l2:
    def __init__(self, model, eps_init=0.001, eps=0.1, num_iters=1000, device=None, beta=1):
        """
        Initialize the GM attack class.
        :param model: The model to attack.
        :param eps: Maximum perturbation amount (clip bound).
        :param beta: Scaling factor for each iteration's step size.
        :param num_iters: Number of iterations to perform.
        """
        self.model = model
        self.eps = eps
        self.num_iters = num_iters
        self.eps_init = eps_init
        self.device = device
        self.beta = beta

    def __init_r__(self, x):
        r_data = (
            torch.randint(2, x.shape, dtype=x.dtype, device=x.device) * 2 - 1
        ) * self.eps_init
        r = r_data.clone().detach().requires_grad_(True)
        r = torch.nn.Parameter(r_data, requires_grad=True).to(self.device)
        return r

    def __get_optimizer__(self, r):
        return optim.Adam([r], lr=0.001, betas=(0.9, 0.999), eps=1e-07, amsgrad=False)

    def generate(self, x):
        x = x.to(self.device)  # Move x to the device first
        y_pred = self.model(x)
        r = self.__init_r__(x)
        optimizer = self.__get_optimizer__(r)
        y_target = self.get_y_target(x, y_pred)

        for epoch in range(self.num_iters):
            y_pred_adv = self.model(x+r)

            loss_ce = nn.functional.cross_entropy(y_pred_adv, y_target, reduction="mean")

            l2_norm = torch.norm(r, p=2)
            loss_total = loss_ce + self.beta * l2_norm
            optimizer.zero_grad()
            loss_total.backward()
            
            optimizer.step()
            r.data = torch.clamp(r.data, -self.eps, self.eps)

        x_adv = x + r
        #y_adv = self.f(x_adv).argmax(1)

        return x_adv

    def get_y_target(self, x, y_pred):
 
        with torch.no_grad():
            y_target = torch.zeros_like(y_pred)
            _, c = torch.max(y_pred, dim=1)
            for i in range(len(y_pred)):
                c_s = torch.arange(y_pred.shape[1], device=y_pred.device)
                c_s = c_s[c_s != c[i]]
                new_c = c_s[torch.randint(0, len(c_s), (1,))]
                y_target[i, new_c] = 1.0

        return y_target

class PGD:
    def __init__(self, model, eps_init=0.001, eps=0.1, beta=0.0005, num_iters=1000, device=None):
        """
        Initialize the PGD attack class.
        :param model: The model to attack.
        :param eps: Maximum perturbation amount (clip bound).
        :param beta: Scaling factor for each iteration's step size.
        :param num_iters: Number of iterations to perform.
        """
        self.model = model
        self.eps = eps
        self.beta = beta
        self.num_iters = num_iters
        self.device = device
        self.eps_init = eps_init
        
    def __init_r__(self, x):
        r_data = (
            torch.randint(2, x.shape, dtype=x.dtype, device=x.device) * 2 - 1
        ) * self.eps_init
        r = r_data.clone().detach().requires_grad_(True)
        r = torch.nn.Parameter(r_data, requires_grad=True).to(self.device)
        return r

    def __get_optimizer__(self, r):
        return optim.Adam([r], lr=0.001, betas=(0.9, 0.999), eps=1e-07, amsgrad=False)

    def generate(self, x):
        """
        Generate adversarial examples.
        :param x: Original input samples.
        :return: Adversarial examples.
        """
        x = x.to(self.device)
        x_adv = x.clone().detach().requires_grad_(True)
        # Forward pass
        y_pred = self.model(x)
        r = self.__init_r__(x)
        optimizer = self.__get_optimizer__(r)

        y_target = self.get_y_target(x, y_pred)
        for _ in range(self.num_iters):
            if _ > 0:  
                random_noise = (torch.rand_like(x) * 2 - 1) * self.eps_init
                r.data = r.data + random_noise
                r.data = torch.clamp(r.data, -self.eps, self.eps)  

            y_pred_adv = self.model(x+r)
            loss = nn.functional.cross_entropy(y_pred_adv, y_target, reduction="mean")
            print(f'{_} adversarial loss is {loss}')
            optimizer.zero_grad()
            loss.backward()
            # Here, we use the sign of the gradient for the update
            grad_sign = r.grad.sign()
            r.data = r.data - self.beta * grad_sign
            r.data = torch.clamp(r.data, -self.eps, self.eps)

        x_adv = x + r
        return x_adv

    def get_y_target(self, x, y_pred):
 
        with torch.no_grad():
            y_target = torch.zeros_like(y_pred)
            _, c = torch.max(y_pred, dim=1)
            for i in range(len(y_pred)):
                c_s = torch.arange(y_pred.shape[1], device=y_pred.device)
                c_s = c_s[c_s != c[i]]
                new_c = c_s[torch.randint(0, len(c_s), (1,))]
                y_target[i, new_c] = 1.0

        return y_target

class PGD_for_AT:
    def __init__(self, model, eps_init=0.001, eps=0.1, beta=0.0005, num_iters=1000, device=None):
        """
        Initialize the PGD attack class.
        :param model: The model to attack.
        :param eps: Maximum perturbation amount (clip bound).
        :param beta: Scaling factor for each iteration's step size.
        :param num_iters: Number of iterations to perform.
        """
        self.model = model
        self.eps = eps
        self.beta = beta
        self.num_iters = num_iters
        self.device = device
        self.eps_init = eps_init
        
    def __init_r__(self, x):
        r_data = (
            torch.randint(2, x.shape, dtype=x.dtype, device=x.device) * 2 - 1
        ) * self.eps_init
        r = r_data.clone().detach().requires_grad_(True)
        r = torch.nn.Parameter(r_data, requires_grad=True).to(self.device)
        return r

    def __get_optimizer__(self, r):
        return optim.Adam([r], lr=0.001, betas=(0.9, 0.999), eps=1e-07, amsgrad=False)

    def generate(self, x, y):
        """
        Generate adversarial examples.
        :param x: Original input samples.
        :return: Adversarial examples.
        """
        x = x.to(self.device)
        x_adv = x.clone().detach().requires_grad_(True)
       
       
        r = self.__init_r__(x)
        optimizer = self.__get_optimizer__(r)

        for _ in range(self.num_iters):
            optimizer.zero_grad()
            y_pred_adv = self.model(x+r)
            loss = nn.functional.cross_entropy(y_pred_adv, y, reduction="mean")
            
            loss.backward()
            
            # Here, we use the sign of the gradient for the update
            grad_sign = r.grad.sign()
            r.data = r.data + self.beta * grad_sign
            r.data = torch.clamp(r.data, -self.eps, self.eps)

        x_adv = x + r
        return x_adv

class CW:
    def __init__(self, model, eps_init=0.001, eps=0.1, c=1e-5, num_iters=1000, device=None):

        self.model = model
        self.eps = eps
        self.c = c
        self.num_iters = num_iters
        self.device = device
        self.eps_init = eps_init

    def __init_r__(self, x):
        r_data = (
            torch.randint(2, x.shape, dtype=x.dtype, device=x.device) * 2 - 1
        ) * self.eps_init
        r = r_data.clone().detach().requires_grad_(True)
        r = torch.nn.Parameter(r_data, requires_grad=True).to(self.device)
        return r

    def __get_optimizer__(self, r):
        return optim.Adam([r], lr=0.001, betas=(0.9, 0.999), eps=1e-07, amsgrad=False)

    def __CW_loss_fun__(self, x, r, y_target, top1_index):
        y_pred_adv = self.model(x + r)
        y_target_labels = torch.max(y_target, 1)[1]  
        loss = nn.functional.cross_entropy(y_pred_adv, y_target_labels, reduction="none") 

        mask = torch.zeros_like(loss, dtype=torch.bool)
        _, top1_index_adv = torch.max(y_pred_adv, dim=1)

        for i in range(len(y_target)):
            if not top1_index_adv[i].item() == top1_index[i].item(): 
                mask[i] = True
        loss[mask] = 0

        # Combine the attack loss with the L2 regularization
        l2_reg = torch.norm(r, p=2)

        return l2_reg * self.c + loss.mean()

    def generate(self, x):
        """
        Generate adversarial examples.
        :param x: Original input samples.
        :return: Adversarial examples.
        """
        x = x.to(self.device)
        #x_adv = x.clone().detach().requires_grad_(True)
        # Forward pass
        y_pred = self.model(x)
        r = self.__init_r__(x)
        optimizer = self.__get_optimizer__(r)

        y_target, top1_index = self.get_y_target(x, y_pred)
        for _ in range(self.num_iters):

            #y_pred_adv = self.model(x+r)
            loss = self.__CW_loss_fun__(x, r, y_target, top1_index)
            print(f'{_} adversarial loss is {loss}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            r.data = torch.clamp(r.data, -self.eps, self.eps)

        x_adv = x + r
        return x_adv

    def get_y_target(self, x, y_pred):

        with torch.no_grad():
            y_target = torch.zeros_like(y_pred)
            _, c1 = torch.max(y_pred, dim=1)
            for i in range(len(y_pred)):
                c_s = torch.arange(y_pred.shape[1], device=self.device)
                c_s = c_s[c_s != c1[i]]
                new_c = c_s[torch.randint(0, len(c_s), (1,))]
                y_target[i, new_c] = 1.0

        return y_target, c1

class SWAP:
    def __init__(self, model, eps_init=0.001, eps=0.1, gamma=0.01, num_iters=1000, device=None):
  
        self.model = model
        self.eps = eps
        self.gamma = gamma
        self.num_iters = num_iters
        self.device = device
        self.eps_init = eps_init

    def __init_r__(self, x):
        r_data = (
            torch.randint(2, x.shape, dtype=x.dtype, device=x.device) * 2 - 1
        ) * self.eps_init
        r = r_data.clone().detach().requires_grad_(True)
        r = torch.nn.Parameter(r_data, requires_grad=True).to(self.device)
        return r

    def __get_optimizer__(self, r):
        return optim.Adam([r], lr=0.001, betas=(0.9, 0.999), eps=1e-07, amsgrad=False)

    def generate(self, x):
        x = x.to(self.device)  # Move x to the device first
        y_pred = self.model(x)
        r = self.__init_r__(x)
        optimizer = self.__get_optimizer__(r)
        y_target = self.get_y_target(x, y_pred)

        for epoch in range(self.num_iters):
            y_pred_adv = self.model(x+r)
            loss = nn.functional.cross_entropy(y_pred_adv, y_target, reduction="mean")
            print(f'{epoch} adversarial loss is {loss}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            r.data = torch.clamp(r.data, -self.eps, self.eps)

        x_adv = x + r
        #y_adv = self.f(x_adv).argmax(1)

        return x_adv

    def get_y_target(self, x, y_pred):
        with torch.no_grad():
            _, top2_indices = torch.topk(y_pred, 2, dim=1)
            y_target = y_pred.clone()

            for i in range(len(y_pred)):
                c_top2 = top2_indices[i]
                mean_ = (
                    y_pred[i, c_top2[0]] + y_pred[i, c_top2[1]]
                ) / 2 
                y_target[i, c_top2[1]] = mean_ + self.gamma
                y_target[i, c_top2[0]] = mean_ - self.gamma  
        return y_target

class SWAP_l2:
    def __init__(self, model, eps_init=0.001, eps=0.1, gamma=0.01, num_iters=1000, device=None, beta=0.1):
  
        self.model = model
        self.eps = eps
        self.gamma = gamma
        self.num_iters = num_iters
        self.device = device
        self.eps_init = eps_init
        self.beta = beta

    def __init_r__(self, x):
        r_data = (
            torch.randint(2, x.shape, dtype=x.dtype, device=x.device) * 2 - 1
        ) * self.eps_init
        r = r_data.clone().detach().requires_grad_(True)
        r = torch.nn.Parameter(r_data, requires_grad=True).to(self.device)
        return r

    def __get_optimizer__(self, r):
        return optim.Adam([r], lr=0.001, betas=(0.9, 0.999), eps=1e-07, amsgrad=False)


    def generate(self, x):
        x = x.to(self.device)  # Move x to the device first
        y_pred = self.model(x)
        r = self.__init_r__(x)
        optimizer = self.__get_optimizer__(r)
        y_target = self.get_y_target(x, y_pred)

        for epoch in range(self.num_iters):
            y_pred_adv = self.model(x+r)
            loss_ce = nn.functional.cross_entropy(y_pred_adv, y_target, reduction="mean")

            l2_norm = torch.norm(r, p=2)
            loss_total = loss_ce + self.beta * l2_norm
            optimizer.zero_grad()
            loss_total.backward()
            
            optimizer.step()
            r.data = torch.clamp(r.data, -self.eps, self.eps)

        x_adv = x + r
        #y_adv = self.f(x_adv).argmax(1)

        return x_adv

    def get_y_target(self, x, y_pred):
        with torch.no_grad():
            _, top2_indices = torch.topk(y_pred, 2, dim=1)
            y_target = y_pred.clone()

            for i in range(len(y_pred)):
                c_top2 = top2_indices[i]
                mean_ = (
                    y_pred[i, c_top2[0]] + y_pred[i, c_top2[1]]
                ) / 2 
                y_target[i, c_top2[1]] = mean_ + self.gamma
                y_target[i, c_top2[0]] = mean_ - self.gamma 
        return y_target
