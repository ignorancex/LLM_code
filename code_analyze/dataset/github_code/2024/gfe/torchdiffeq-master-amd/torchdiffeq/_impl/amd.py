 # Kyriakos flouris + Ender konukoglu
import torch
from .misc import ( _convert_to_tensor, _is_finite, _handle_unused_kwargs, _is_iterable,
    _optimal_step_size_amd, _compute_error_ratio_amd
)
from .solvers import AMDAdaptiveStepsizeODESolver
from .amd_common import   _AMDstate, _amd_step
import numpy

# ------Plot temp--------
import matplotlib
from matplotlib import pyplot as plt
# -----------------------
    # if self.plot_dterr:
    #     self._plot_func(self.plot_err, self.plot_dt)
    #     self.plot_err=[]
    #     self.plot_dt=[] 
        # if self.plot_dterr:      
        #     self.plot_err.append(LHS.mean())
        #     self.plot_dt.append(dt_next)


def _ta_append(list_of_tensors, value):
    """Append a value to the end of a list of PyTorch tensors."""
    list_of_tensors.append(value)
    return list_of_tensors


class AdaptiveMinimiseDistance(AMDAdaptiveStepsizeODESolver):

    def __init__(
        self, func, floss, y0, rtol, atol, first_step=None,tau=1e4, dtstep=1, dtmaxstep=10, dtfactor=0.75, dtiter=50, 
        conv_graddt=1e-9 ,conv_percentagetau=0.9, conv_dtstep=100, armijosigma=1e-5 ,max_num_steps=2**31 - 1, plot_dterr=False,
        **unused_kwargs ):
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs
        self.floss=floss
        self.func = func
        self.y0 = y0
        self.rtol = rtol if _is_iterable(rtol) else [rtol] * len(y0)
        self.atol = atol if _is_iterable(atol) else [atol] * len(y0)
        self.first_step = first_step
        self.dtstep = _convert_to_tensor(dtstep, dtype=torch.float64, device=y0[0].device)
        self.dtmaxstep = _convert_to_tensor(dtmaxstep, dtype=torch.float64, device=y0[0].device)
        self.conv_dtstep = _convert_to_tensor(conv_dtstep, dtype=torch.float64, device=y0[0].device)
        self.conv_graddt = _convert_to_tensor(conv_graddt, dtype=torch.float64, device=y0[0].device)
        self.dtfactor = _convert_to_tensor(dtfactor, dtype=torch.float64, device=y0[0].device)
        self.dtiter = _convert_to_tensor(dtiter, dtype=torch.float64, device=y0[0].device)
        self.conv_percentagetau = _convert_to_tensor(conv_percentagetau, dtype=torch.float64, device=y0[0].device)
        self.tau = _convert_to_tensor(tau, dtype=torch.float64, device=y0[0].device)
        self.armijosigma=armijosigma
        self.reject_counter=0
        self.plot_dterr=plot_dterr
        self.plot_err=[]
        self.plot_dt=[]        
        self.max_num_steps = _convert_to_tensor(max_num_steps, dtype=torch.int32, device=y0[0].device)
        self.convergent=False
        self.accept_step=False
        self.failedtoaccept_count=0
        self.loss_list=[]

    def before_integrate(self, t):
        f0 = self.func(t[0].type_as(self.y0[0]), self.y0)
        first_step = self.dtstep
        self.amd_state = _AMDstate(self.y0, f0, t[0], t[0], first_step)        
#  is not  used anymore:
    def advance(self, next_t):
        """Interpolate through the next time point, integrating as necessary."""
        while not self.convergent and next_t > self.amd_state.t1:
            self.amd_state = self._adaptive_amd_step(self.amd_state, next_t )
        return self.amd_state.y1

    def _plot_func(arr_err,arr_dt):
        fig, (ax0 ) = plt.subplots(nrows=1, ncols=1, sharex=False,
                                            figsize=(18, 6))
        ax0.plot(arr_dt,arr_err, 'o') # linestyle='--    
        ax0.set_ylabel('err_ratio')
        ax0.set_xlabel('dt')
        plt.show()
        plt.close()    

    def _adaptive_amd_step(self, amd_state, advance_next_t ):
        """Take an adaptive Runge-Kutta step to integrate the ODE."""
        y0, f0, _, t0, dt  = amd_state
        fl0 = self.floss( y0)
        ########################################################
        #                      Assertions                      #
        ########################################################
        assert t0 + dt > t0, 'underflow in dt {}'.format(dt.item())
        for y0_ in y0:
            assert _is_finite(torch.abs(y0_)), 'non-finite values in state `y`: {}'.format(y0_)
        y1, f1 = _amd_step(self.func, y0, f0, t0, dt)
        fl1 = self.floss( y1)
        
        ########################################################
        #                     Error Ratio                      #
        ########################################################     
        # mean_sq_error_ratio = (fl1[0]/fl0[0],) 
        LHS=fl0[0]-fl1[0]
        RHS=+self.armijosigma*dt*self.dtmaxstep*torch.einsum('ij,ij->i', f1[0],f1[0] ) 
# 
        # RHS=torch.mean(RHS)
        # LHS=torch.mean(LHS)

        # This is for momentum:
        # try: RHS=-self.armijosigma*dt*self.dtmaxstep*torch.einsum('ij,ij->i', f1[0],f1[0] ) 
        # except RuntimeError:  RHS=-self.armijosigma*dt*self.dtmaxstep*torch.einsum('ij,ij->i', f1[0][0],f1[0][0] )
        # accept_step = ((mean_sq_error_ratio[0] <= 1).all() or dt< self.dtfactor**self.dtiter)
        accept_step = ((LHS>=RHS).all())
        if  dt< self.dtfactor**self.dtiter: #or self.failedtoaccept_count > 100:
            print('> failed to min dist at t=', t0.item())
            dt=self.conv_dtstep
            dt = torch.min(dt, advance_next_t - t0) 
            # self.failedtoaccept_count+=1
            accept_step=True                         
        ########################################################
        #                   Update RK State                    #
        ########################################################
        y_next = y1 if accept_step else y0
        f_next = f1 if accept_step else f0
        t_next = t0 + dt if accept_step else t0
        dt_next= dt if accept_step else self.dtfactor*dt
        if accept_step:
            dt_next = torch.max(_convert_to_tensor(self.dtstep, dtype=t0.dtype, device=t0.device), dt*1.1)
            dt_next = torch.min(dt_next,_convert_to_tensor(self.dtmaxstep, dtype=t0.dtype, device=t0.device))
            dt_next = torch.min(dt_next, advance_next_t - t0)
            if ((self.tau.item()-t_next.item())/self.tau.item() <= self.conv_percentagetau):
                self.loss_list.append(fl1[0].mean().item())
                if (len(self.loss_list) >= 5):
                    loss_grads=torch.from_numpy(numpy.gradient(self.loss_list[-5:]))
                    if(torch.abs(torch.mean(loss_grads)) <  self.conv_graddt):
                        self.convergent=True
                        print('>> converged at t=', t_next.item())
            # print('accepted_step ==, error={0:2.3f}, loss={1:2.3f} , dt={2:2.3f}, dt_next={3:2.3f}, t={4:2.4f}'.format(
            #                                                                                             mean_sq_error_ratio[0].item(),
            #                                                                                             fl1[0].item(),
            #                                                                                             dt.item(),
            #                                                                                             dt_next.item(),
            #                                                                                             t_next.item()))
        else:
            # print('REJ:', LHS)
            self.reject_counter+=1
            # import sys
            # sys.exits(0)


            
        state = _AMDstate(y_next, f_next, t0, t_next, dt_next)
        self.accept_step=accept_step
        return state
        del y0, f0, _, t0, dt, y1, f1, state, RHS, LHS, accept_step,

