"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like
from ldm.modules.diffusionmodules.util import extract_into_tensor

class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model

        self.ddpm_num_timesteps = model.num_timesteps

        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddpm_num_steps=None, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        if ddpm_num_steps is None:
            self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        else:
            self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=ddpm_num_steps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))
        self.register_buffer('alphas_cumprod_next', to_torch(self.model.alphas_cumprod_next))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)


    # @torch.no_grad()
    # def ddim_reverse_sample(
    #     self,
    #     x,
    #     t,
    #     index,
    #     clip_denoised=False,
    #     return_x0=True,
    #     model_kwargs=None,
    #     eta=0.0,
    # ):
    #     """
    #     Sample x_{t+1} from the model using DDIM reverse ODE.
    #     """
    #     assert eta == 0.0, "Reverse ODE only for deterministic path"
        
    #     _, _, _, x_recon = self.model.p_mean_variance(
    #         x,
    #         model_kwargs['cond'],
    #         t,
    #         clip_denoised=clip_denoised,
    #         return_x0=return_x0
    #     )
    #     # Usually our model outputs epsilon, but we re-derive it
    #     # in case we used x_start or x_prev prediction.
    #     eps = (extract_into_tensor(self.sqrt_recip_alphas_cumprod, index, x.shape) * x
    #         - x_recon) / extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, index, x.shape)
    #     alpha_bar_next = extract_into_tensor(self.alphas_cumprod_next, index, x.shape)

    #     # Equation 12. reversed
    #     mean_pred = (
    #         x_recon * torch.sqrt(alpha_bar_next)
    #         + torch.sqrt(1 - alpha_bar_next) * eps
    #     )

    #     return {"sample": mean_pred, "pred_xstart": x_recon}


    # @torch.no_grad()
    # def ddim_reverse_sampling(
    #     self,
    #     x,
    #     clip_denoised=True,
    #     return_x0=True,
    #     model_kwargs=None,
    #     eta=0.0,
    #     device=None,
    #     num_timesteps=100
    # ):

    #     self.make_schedule(ddim_num_steps=num_timesteps, ddim_eta=eta, verbose=False)
    #     if device is None:
    #         device = next(self.model.parameters()).device
    #     sample_t = []
    #     xstart_t = []
    #     T = []
    #     timesteps = self.ddim_timesteps
    #     # timesteps = np.asarray([step for step in self.ddim_timesteps if step <= timesteps])

    #     time_range = timesteps
    #     total_steps = timesteps.shape[0]
    #     print(f"Running DDIM Sampling with {total_steps} timesteps")

    #     iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        
    #     sample = x
    #     for i, step in enumerate(iterator):
    #         # index = total_steps - i - 1
    #         index = torch.full((x.size(0),), step -1 , device=device, dtype=torch.long)
    #         ts = torch.full((x.size(0),), step, device=device, dtype=torch.long)
            
    #     # for i in indices:
    #     #     t = torch.tensor([i] * len(sample), device=device)
    #         out = self.ddim_reverse_sample(sample,
    #                                         t=ts,
    #                                         index=index,
    #                                         clip_denoised=clip_denoised,
    #                                         return_x0=return_x0,
    #                                         model_kwargs=model_kwargs,
    #                                         eta=eta)
    #         sample = out['sample']
    #         if i == total_steps-1:
    #             sample_t.append(out['sample'])
    #             # [0, ...., T-1]
    #             xstart_t.append(out['pred_xstart'])
    #             # [0, ..., T-1] ready to use
    #             T.append(ts[0])

    #     return {
    #         #  xT "
    #         'sample': sample,
    #         # (1, ..., T)
    #         'sample_t': sample_t,
    #         # xstart here is a bit different from sampling from T = T-1 to T = 0
    #         # may not be exact
    #         'xstart_t': xstart_t,
    #         'T': T,
    #     }

    @torch.no_grad()
    def reverse_ddim_sampling(self, cond, shape,
                              x_T=None, ddim_use_original_steps=False,
                              callback=None, timesteps=None, timesteps_start=None, timesteps_end=None,
                              quantize_denoised=False, mask=None, x0=None, img_callback=None, log_every_t=100,
                              temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                              alpha=[2.0,1.0,0.0], target="sc_joint", validation=False):
        b = shape[0]
        if cond is not None:
            if isinstance(cond, dict):
                cbs = cond[list(cond.keys())[0]].shape[0]
                if cbs != b:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {b}")
            elif isinstance(cond, list):
                print(f"Got condtions of conditioning[0]: {cond[0].size()} and conditioning[1]: {cond[1].size()}")
            else:
                if cond.shape[0] != cond:
                    print(f"Warning: Got {cond.shape[0]} conditionings but batch-size is {b}")

        self.make_schedule(ddim_num_steps=100, ddim_eta=0.0, verbose=False)
        device = self.model.betas.device
        
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]
        else:
            pass

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = range(0,timesteps) if ddim_use_original_steps else timesteps

        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        if ddim_use_original_steps:
            alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.Tensor([0.0]).to(self.model.device)])
        else:
            alphas_cumprod_next = torch.cat([self.ddim_alphas[1:], torch.Tensor([0.0]).to(self.model.device)])

        for i, step in enumerate(iterator):
            #index = total_steps - i - 1
            index = i
            index_tensor = torch.Tensor([int(index)]).to(self.model.device).type(torch.int64)
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img
            img, pred_x0, eps = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                                  quantize_denoised=quantize_denoised, temperature=temperature,
                                                  noise_dropout=noise_dropout, score_corrector=score_corrector,
                                                  corrector_kwargs=corrector_kwargs,
                                                  alpha=alpha, target=target, validation=validation)
            # img, pred_x0 = outs
            alpha_bar_next = extract_into_tensor(alphas_cumprod_next, index_tensor, img.shape)
            # Equation 12. reversed  (DDIM paper)  (torch.sqrt == torch.sqrt)
            img = pred_x0 * torch.sqrt(alpha_bar_next) + torch.sqrt(1 - alpha_bar_next) * eps
            
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img


    @torch.no_grad()
    def sample_within_t(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               timesteps=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               alpha=[1.0, 1.0], 
               target="s",
               ddpm_num_steps=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            elif isinstance(conditioning, list):
                print(f"Got condtions of conditioning[0]: {conditioning[0].size()} and conditioning[1]: {conditioning[1].size()}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddpm_num_steps=ddpm_num_steps, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    timesteps=timesteps,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    alpha=alpha, target=target
                                                    )
        return samples, intermediates


    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               timesteps=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               alpha=[1.0, 1.0], 
               target="s",
               validation=False,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            elif isinstance(conditioning, list):
                print(f"Got condtions of conditioning[0]: {conditioning[0].size()} and conditioning[1]: {conditioning[1].size()}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    timesteps=timesteps,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    alpha=alpha, target=target,
                                                    validation=validation
                                                    )
        return samples, intermediates


    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      alpha=[1.0, 1.0, 0.0], target="sc_joint", validation=False):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            # subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            # timesteps = self.ddim_timesteps[:subset_end]
            timesteps = np.asarray([step for step in self.ddim_timesteps if step <= timesteps])

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            
            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      alpha=alpha, target=target, validation=validation)
            img, pred_x0, _ = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates


    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      alpha=[1.0, 1.0], target="s", validation=False):
        b, *_, device = *x.shape, x.device

        # if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
        #     e_t = self.model.apply_model(x, t, c)
        # else:
        #     x_in = torch.cat([x] * 2)
        #     t_in = torch.cat([t] * 2)
        #     c_in = torch.cat([unconditional_conditioning, c])
        #     e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
        #     e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if alpha[0] == 0 and alpha[1] == 0:
            e_t = self.model.apply_model(x, t, c, validation=validation)

        else:
            uncond_c = [
                    torch.zeros_like(c[0], device=device), 
                    torch.zeros_like(c[1], device=device)
            ]
            style_c = [
                    c[0],
                    torch.zeros_like(c[1], device=device)
            ]
            content_c = [
                    torch.zeros_like(c[0], device=device),
                    c[1]
            ]

            if target == "sc":
                uncond_e_t = self.model.apply_model(x, t, uncond_c, validation=validation)
                style_e_t = self.model.apply_model(x, t, style_c, validation=validation)
                content_e_t = self.model.apply_model(x, t, content_c, validation=validation)

                e_t = uncond_e_t + \
                    alpha[0] * (style_e_t - uncond_e_t) + \
                    alpha[1] * (content_e_t - uncond_e_t)

            elif target == "sc_joint":
                uncond_e_t = self.model.apply_model(x, t, uncond_c, validation=validation)
                if alpha[1] == 1:
                    style_e_t = 0
                    content_e_t = 0 
                    style_content_e_t = self.model.apply_model(x, t, c, validation=validation)
                elif alpha[1] == 0:
                    if alpha[2] == 1:
                        style_e_t = self.model.apply_model(x, t, style_c, validation=validation)
                        content_e_t = 0
                    elif alpha[2] == 0:
                        style_e_t = 0
                        content_e_t = self.model.apply_model(x, t, content_c, validation=validation)
                    else:
                        style_e_t = self.model.apply_model(x, t, style_c, validation=validation)
                        content_e_t = self.model.apply_model(x, t, content_c, validation=validation)
                    style_content_e_t = 0
                else:
                    style_e_t = self.model.apply_model(x, t, style_c, validation=validation)
                    content_e_t = self.model.apply_model(x, t, content_c, validation=validation)
                    style_content_e_t = self.model.apply_model(x, t, c, validation=validation)

                # alpha = alpha[0], 0 <= alpha
                # lambda = alpha[1], 0 <= lambda <= 1
                
                e_t = uncond_e_t + \
                    alpha[0] * (
                        alpha[1] * (style_content_e_t - uncond_e_t) + \
                        (1 - alpha[1]) * 2 * (
                            alpha[2] * (style_e_t - uncond_e_t) + \
                            (1 - alpha[2]) * (content_e_t - uncond_e_t)
                        )
                    )
                    
                    # (style_content_e_t + uncond_e_t - style_e_t - content_e_t)

            elif target == "s":
                uncond_e_t = self.model.apply_model(x, t, uncond_c, validation=validation)
                style_e_t = self.model.apply_model(x, t, style_c, validation=validation)
                e_t = uncond_e_t + alpha[0] * (style_e_t - uncond_e_t)

            elif target == "c":
                uncond_e_t = self.model.apply_model(x, t, uncond_c, validation=validation)
                content_e_t = self.model.apply_model(x, t, content_c, validation=validation)
                e_t = uncond_e_t + alpha[1] * (content_e_t - uncond_e_t)

            elif target == "uncond":
                uncond_e_t = self.model.apply_model(x, t, uncond_c, validation=validation)
                e_t = uncond_e_t

            elif target == "two_conds": # without the guidance.
                e_t = self.model.apply_model(x, t, c, validation=validation)

            else:
                raise ValueError("target should be set to be one of s, c, sc, sc_joint, uncond, two_conds.")


        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0, e_t
