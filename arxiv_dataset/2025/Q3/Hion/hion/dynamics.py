from typing import Dict, List, Optional, Tuple
import torch as th
from itertools import chain

"""
Author: Josue N Rivera
"""
        
class Dynamics():

    def __init__(self, 
                 constants:Optional[Dict] = None,
                 state_derivative_orders:List[int] = [1],
                 control_derivative_orders:List[int] = [0],
                 invariant_state_mask:Optional[List[bool]] = None,
                 control_in_ode_mask:Optional[List[bool]] = None,
                 device:Optional[th.device] = None) -> None:
        
        """
        * By definition, only primitive states can be invariant so the mask must be the same size as `state_derivative_orders`
        """

        if device is None:
            device = th.device('cuda' if th.cuda.is_available() else 'cpu')

        self.constants = constants

        self.state_derivative_orders = th.tensor(state_derivative_orders, dtype=th.int, device=device)
        self.control_derivative_orders = th.tensor(control_derivative_orders, dtype=th.int, device=device)
        
        self.first_state_orders = th.cat([th.arange(i+1, dtype=th.int, device=device) for i in self.state_derivative_orders])
        self.first_control_orders = th.cat([th.arange(i+1, dtype=th.int, device=device) for i in self.control_derivative_orders])
        
        self.highest_state_order = max(self.state_derivative_orders)
        self.highest_control_order = max(self.control_derivative_orders)

        self.highest_first_state_order_mask = th.tensor(list(chain([[False]*i+[True] for i in state_derivative_orders])), dtype=th.bool, device=device).view(-1)
        
        self.highest_first_state_order_idxs = self.highest_first_state_order_mask.nonzero().view(-1)

        self.highest_first_control_order_mask = th.tensor(list(chain([[False]*i+[True] for i in control_derivative_orders])), dtype=th.bool, device=device).view(-1)
        
        self.state_primitive_mask = th.tensor(list(chain([[True]+[False]*i for i in state_derivative_orders])), dtype=th.bool, device=device).view(-1)
        self.control_primitive_mask = th.tensor(list(chain([[True]+[False]*i for i in control_derivative_orders])), dtype=th.bool, device=device).view(-1)


        self.primitive_state_n, self.primitive_control_n = (len(state_derivative_orders), len(control_derivative_orders))

        self.first_order_state_n, self.first_order_control_n = (sum(state_derivative_orders) + self.primitive_state_n, sum(control_derivative_orders) + self.primitive_control_n)

        invariant_state_mask = [False]*len(self.primitive_state_n) if invariant_state_mask is None else invariant_state_mask

        invariant_state_mask = [False]*self.primitive_state_n if invariant_state_mask is None else invariant_state_mask
        self.invariant_state_mask = th.tensor(invariant_state_mask, dtype=th.bool, device=device)

        self.first_invariant_state_mask = th.tensor(list(chain([[invariant_state_mask[i]]+[False]*n for i, n in enumerate(state_derivative_orders)])), dtype=th.bool, device=device).view(-1)

        self.invariant_state_idxs = self.invariant_state_mask.nonzero().view(-1)

        self.first_invariant_state_idxs = self.first_invariant_state_mask.nonzero().view(-1)

        self.has_invariant_states = sum(self.invariant_state_mask) > 0

        self.in_state_order_idxs = [(self.state_derivative_orders >= order).nonzero().view(-1) for order in range(self.highest_state_order+1)]

        control_in_ode_mask = [False]*self.primitive_state_n if control_in_ode_mask is None else control_in_ode_mask

        self.control_in_ode_mask = th.tensor(control_in_ode_mask, dtype=th.bool, device=device).view(-1)
        self.control_in_f_mask = th.tensor(list(chain([[False]*n+[control_in_ode_mask[i]] for i, n in enumerate(state_derivative_orders)])), dtype=th.bool, device=device).view(-1)
    
    def first_state_representation(self,
                   time: th.DoubleTensor,
                   state: th.DoubleTensor) -> th.DoubleTensor:

        first_states = []

        for order_n_idx in range(len(self.state_derivative_orders)):
            state_order = state[:, order_n_idx:order_n_idx+1]
            first_states.append(state_order)

            for _ in range(self.state_derivative_orders[order_n_idx]):
                state_order = th.autograd.grad(state_order.sum(), time, create_graph=True)[0]
                first_states.append(state_order)
    
        return th.cat(first_states, dim=1)
    
    def first_control_representation(self,
                   time: th.DoubleTensor,
                   control: th.DoubleTensor) -> th.DoubleTensor:
        
        first_controls = []

        for order_n_idx in range(len(self.control_derivative_orders)):
            control_order = control[:, order_n_idx:order_n_idx+1]
            first_controls.append(control_order)

            for _ in range(self.control_derivative_orders[order_n_idx]):
                control_order = th.autograd.grad(control_order.sum(), time, create_graph=True)[0]
                first_controls.append(control_order)
    
        return th.cat(first_controls, dim=1)
    
    def highest_order_state(self,
          time: th.DoubleTensor, 
          first_order_state: th.DoubleTensor) -> th.DoubleTensor:
        
        return th.cat([th.autograd.grad(first_order_state[:, i].sum(), time, create_graph=True)[0] for i in self.highest_first_state_order_idxs], dim=1)
    
    def extract_u(self,
          time: th.DoubleTensor, 
          first_order_state: th.DoubleTensor,
          highest_order_state: th.DoubleTensor) -> Optional[th.DoubleTensor]:
        
        raise NotImplementedError
    
    def f(self,
          time: th.DoubleTensor, 
          first_order_state: th.DoubleTensor,
          first_order_control: th.DoubleTensor) -> th.DoubleTensor:
        
        raise NotImplementedError
    
    def dfdx(self,
          time: th.DoubleTensor, 
          first_order_state: th.DoubleTensor,
          first_order_control: th.DoubleTensor) -> th.DoubleTensor:
        
        raise NotImplementedError
    
    def dfdu(self,
          time: th.DoubleTensor, 
          first_order_state: th.DoubleTensor,
          first_order_control: th.DoubleTensor) -> th.DoubleTensor:
        
        raise NotImplementedError

    def ode_residual(self,
                   time: th.DoubleTensor,
                   first_order_state: th.DoubleTensor, 
                   first_order_control: th.DoubleTensor) -> th.DoubleTensor:
        """
        Stack of dynamics residual errors
        """

        f = self.f(time, first_order_state, first_order_control)[:, self.highest_first_state_order_mask]

        x_dot = self.highest_order_state(time, first_order_state)

        return x_dot - f
    
    def first_order_representation(self,
                   time: th.DoubleTensor,
                   state: th.DoubleTensor, 
                   control: th.DoubleTensor) -> Tuple[th.DoubleTensor, th.DoubleTensor]:

        return self.first_state_representation(time=time, state=state), \
               self.first_control_representation(time=time, control=control)
        
    def first_state_names(self) -> List[str]:

        orders = self.state_derivative_orders

        return [f'x_{{{orders[o_idx]}}}^{{[{i}]}}' for o_idx in range(orders) for i in range(orders[o_idx]+1)]
        
    def first_control_names(self) -> List[str]:
    
        orders = self.control_derivative_orders
        return [f'u_{{{orders[o_idx]}}}^{{[{i}]}}' for o_idx in range(orders) for i in range(orders[o_idx]+1)]
        
    def first_order_names(self) -> Tuple[List[str], List[str]]:
                   
        return self.first_state_names(), self.first_control_names()

class SecondOrderLinearDynamics(Dynamics):

    def __init__(self, device:Optional[th.device] = None) -> None:
        
        """
        Dynamics for a hanging pendulum 
        
        .. math::
            \ddot{x} = u

        """

        super().__init__(state_derivative_orders=[1],
                         control_derivative_orders=[0],
                         invariant_state_mask=[True],
                         control_in_ode_mask=[True],
                         device=device)
    
    def f(self,
          time: th.DoubleTensor, 
          first_order_state: th.DoubleTensor,
          first_order_control: th.DoubleTensor) -> th.DoubleTensor:
        
        df1 = first_order_state[:, 1:2]
        df2 = first_order_control

        return th.cat([df1, df2], dim=1)
    
    def dfdx(self, time: th.DoubleTensor, first_order_state: th.DoubleTensor, first_order_control: th.DoubleTensor) -> th.DoubleTensor:

        x1, x2 = first_order_state[:, 0:1], first_order_state[:, 1:2]

        df1 = th.cat([x1*0.0, x2/x2], dim=1)
        df2 = th.cat([x1*0.0, x2*0.0], dim=1)

        return th.stack([df1, df2], dim=1)
    
    def dfdu(self, time: th.DoubleTensor, first_order_state: th.DoubleTensor, first_order_control: th.DoubleTensor) -> th.DoubleTensor:

        u1 = first_order_control[:, 0:1]

        df1 = th.cat([u1*0.0], dim=1)
        df2 = th.cat([u1/u1], dim=1)

        return th.stack([df1, df2], dim=1)

    def ode_residual(self,
                   time: th.DoubleTensor,
                   first_order_state: th.DoubleTensor,
                   first_order_control: th.DoubleTensor) -> th.DoubleTensor:

        dx = first_order_state[:, 1:2]
        d2x = th.autograd.grad(dx.sum(), time, create_graph=True)[0]
        control = first_order_control
        
        error = d2x - control

        return error
    
    def extract_u(self, time: th.DoubleTensor, first_order_state: th.DoubleTensor, highest_order_state: th.DoubleTensor) -> Optional[th.DoubleTensor]:

        return highest_order_state
    
class VanDerPolDynamics(Dynamics):

    def __init__(self, constants:Optional[dict] = None, device:Optional[th.device] = None) -> None:
        
        """
        Dynamics for Van der Pol circuit
        
        .. math::
            \ddot{x} = \mu(1 - x^2)\dot{x} - x + u

        """

        constants = {
            'damping': 1.0,
        } if constants is None else constants

        super().__init__(constants=constants,
                         state_derivative_orders=[1],
                         control_derivative_orders=[0],
                         invariant_state_mask=[False],
                         control_in_ode_mask=[True], 
                         device=device)
    
        self.d = self.constants['damping']

    def f(self,
          time: th.DoubleTensor, 
          first_order_state: th.DoubleTensor,
          first_order_control: th.DoubleTensor) -> th.DoubleTensor:
        
        x1, x2 = first_order_state[:, 0:1], first_order_state[:, 1:2]

        df1 = x2
        df2 = self.d*(1 - x1**2)*x2 - x1 + first_order_control[:, 0:1]

        return th.cat([df1, df2], dim=1)
    
    def dfdx(self, time: th.DoubleTensor, first_order_state: th.DoubleTensor, first_order_control: th.DoubleTensor) -> th.DoubleTensor:

        x1, x2 = first_order_state[:, 0:1], first_order_state[:, 1:2]

        df1 = th.cat([x1*0.0, x2/x2], dim=1)
        df2 = th.cat([-2.0*self.d*x1*x2 - 1.0, self.d*(1 - x1**2)], dim=1)

        return th.stack([df1, df2], dim=1)
    
    def dfdu(self, time: th.DoubleTensor, first_order_state: th.DoubleTensor, first_order_control: th.DoubleTensor) -> th.DoubleTensor:

        u1 = first_order_control[:, 0:1]

        df1 = th.cat([u1*0.0], dim=1)
        df2 = th.cat([u1/u1], dim=1)

        return th.stack([df1, df2], dim=1)

    def ode_residual(self,
                   time: th.DoubleTensor,
                   first_order_state: th.DoubleTensor,
                   first_order_control: th.DoubleTensor) -> th.DoubleTensor:

        x1, x2 = first_order_state[:, 0:1], first_order_state[:, 1:2]
        d2x = th.autograd.grad(x2.sum(), time, create_graph=True)[0]
        
        return self.d*(1 - x1**2)*x2 - x1 + first_order_control[:, 0:1] - d2x
    
    def extract_u(self, time: th.DoubleTensor, first_order_state: th.DoubleTensor, highest_order_state: th.DoubleTensor) -> Optional[th.DoubleTensor]:

        x1, x2 = first_order_state[:, 0:1], first_order_state[:, 1:2]
        x_dot = highest_order_state

        return -self.d*(1 - x1**2)*x2 + x1 + x_dot

class PendulumDynamics(Dynamics):

    def __init__(self, 
                 constants:Optional[Dict] = None, device:Optional[th.device] = None) -> None:
        
        """
        Dynamics for a hanging pendulum 
        
        .. math::
            J \ddot{\theta} + W l \sin(\theta) = u
        
        Constants:
            Inertia (J; default: 1.0 kg m^2)
            Mass of the body (W; default: 1.0 kg)
            Gravity (g; default: 9.807 m^2/s)
            Length of rod (l; default: 1.0 m)

        """

        constants = {
            'inertia': 1.0,
            'mass of body': 1.0,
            'gravity': 9.807,
            'length of rod': 1.0
        } if constants is None else constants

        super().__init__(constants,
                         state_control_n = (1, 1),
                         first_order_state_control_n = (2, 1),
                         device=device)

        self.a = (self.constants['mass of body']*self.constants['gravity']*\
            self.constants['length of rod'])/self.constants['inertia']

        self.b = 1.0/self.constants['inertia']
    
    def first_control_representation(self,
                   time: th.DoubleTensor,
                   control: th.DoubleTensor) -> th.DoubleTensor:

        return control

    def first_state_representation(self,
                   time: th.DoubleTensor,
                   state: th.DoubleTensor) -> th.DoubleTensor:

        dtheta = th.autograd.grad(state.sum(), time, retain_graph=True)[0]

        return th.cat([state, dtheta], dim=1)
    
    def f(self,
          time: th.DoubleTensor, 
          first_order_state: th.DoubleTensor,
          first_order_control: th.DoubleTensor) -> th.DoubleTensor:
        
        theta, dtheta = first_order_state[:, 0:1], first_order_state[:, 1:2]
        d2theta = -self.a*th.sin(theta) + self.b*first_order_control

        return th.cat([dtheta, d2theta], dim=1)

    def ode_residual(self,
                     time: th.DoubleTensor,
                     first_order_state: th.DoubleTensor,
                     first_order_control: th.DoubleTensor) -> th.DoubleTensor:

        theta = first_order_state[:, 0:1]
        dtheta = first_order_state[:, 1:2]
        d2theta = th.autograd.grad(dtheta.sum(), time, retain_graph=True)[0]
        control = first_order_control.detach()

        # TODO: Change residual to be error = (state estimation - f)

        """ error = control - self.constants['inertia']*d2theta \
              + self.constants['mass of the body']*self.constants['gravity']*th.sin(state) """
        
        error = d2theta + self.a*th.sin(theta) - self.b*control

        return error

class RobotArmDynamics(Dynamics):

    def __init__(self, 
                 constants:Optional[Dict] = None) -> None:

        constants = {
            'm1': 1.0,
            'm2': 1.0,
            'I1': 1.0,
            'I2': 1.0,
            'lc1': 0.5,
            'lc2': 0.5,
            'l1': 1.0,
            'l2': 1.0,
            'g': 9.807
        } if constants is None else constants

        super().__init__(constants=constants,
                         state_derivative_orders=[1, 1],
                         control_derivative_orders=[2, 2],
                         invariant_state_mask=[False],
                         control_in_ode_mask=[True])
        
        self.m11 = self.constants['I1'] + self.constants['m1']*self.constants['lc1']**2 + self.constants['m2']*self.constants['l1']**2

        self.m12 = self.m21 = self.constants['m2']*self.constants['l1']*self.constants['lc2']

        self.m22 = self.constants['I2'] + self.constants['m2']*self.constants['lc2']**2

        self.g1 = -(self.constants['m1']*self.constants['lc1'] + self.constants['m2']*self.constants['l1'])*self.constants['g']

        self.g2 = -self.constants['m2']*self.constants['lc2']*self.constants['g']

        self.c1 = self.constants['m2']*self.constants['l1']*self.constants['lc2']

        self.c2 = -self.constants['m2']*self.constants['l1']*self.constants['lc2']

    def ode_residual(self,
                   time: th.DoubleTensor,
                   first_order_state: th.DoubleTensor,
                   first_order_control: th.DoubleTensor) -> th.DoubleTensor:

        q1, d1q1, q2, d1q2 = first_order_state[:, 0:1], first_order_state[:, 1:2], first_order_state[:, 2:3], first_order_state[:, 3:4]

        state = self.top_state_derivative(time, first_order_state)
        d2q1, d2q2 = state[:, 0:1], state[:, 1:2]

        u1, u2 = first_order_control[:, 0:1], first_order_control[:, 1:2]

        diff = q1 - q2

        error1 = self.m11*d2q1 + self.m12*th.cos(diff)*d2q2 + self.c1*th.sin(diff)*d1q2**2 + self.g1*th.sin(q1) - u1
        error2 = self.m21*th.cos(diff)*d2q1 + self.m22*d2q2 + self.c2*th.sin(diff)*d1q1**2 + self.g2*th.sin(q2) - u2
        
        return th.cat([error1, error2], dim=1)
    
    def dfdu(self, time: th.DoubleTensor, first_order_state: th.DoubleTensor, first_order_control: th.DoubleTensor) -> th.DoubleTensor:

        u1, u2 = first_order_control[:, 0:1], first_order_control[:, 1:2]

        df1 = th.cat([u1/u1], dim=1)
        df2 = th.cat([u2/u2], dim=1)

        return th.stack([df1, df2], dim=1)
