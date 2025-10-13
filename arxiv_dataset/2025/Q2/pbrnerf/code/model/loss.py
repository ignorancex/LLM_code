import torch
from torch import nn
from utils import general
import torch.nn.functional as F

class NeILFLoss(nn.Module):
    def __init__(self,
            rgb_loss_type,
            lambertian_weighting,
            smoothness_weighting,
            reflection_weighting,
            var_weighting,
            energy_cons_weighting,
            brdf_weighted_specular_weighting,
            geo_loss_fn,
            phase,
            num_train_incident_samples,
            **kwargs):
        super().__init__()
        self.rgb_loss_fn = general.get_class(rgb_loss_type)(reduction='none')

        # Material and Joint phase loss weights
        self.lam_weight = lambertian_weighting
        self.smth_weight = smoothness_weighting
        self.ref_weight = reflection_weighting
        self.var_weight = var_weighting
        self.energy_cons_weight = energy_cons_weighting
        self.brdf_weighted_specular_weight = brdf_weighted_specular_weighting

        self.mono_neilf = kwargs.get('mono_neilf', False)
        self.remove_black = kwargs.get('remove_black', False)
        self.reflection_grad_scale = kwargs.get('reflection_grad_scale', 0)

        self.geo_loss_fn = geo_loss_fn
        self.phase = phase
        self.num_train_incident_samples = num_train_incident_samples

    def brdf_smoothness_loss_fn(self, brdf_grads, rgb_grad):
        return brdf_grads.norm(dim=-1).sum(dim=-1) * (-rgb_grad).exp()

    def lambertian_loss_fn(self, roughness, metallic):
        return (roughness - 1).abs() + (metallic - 0).abs()

    def reflection_loss_fn(self, reflection_nn_rgb, reflection_render_rgb):
        return (reflection_nn_rgb - reflection_render_rgb).abs()

    def variance_loss_fn(self, rgb_variance, roughness):
        var_min, var_max = 0.01, 0.15
        tgt_min, tgt_max = 0.25, 0.75
        roughness_target = (rgb_variance - var_min) / (var_max - var_min)
        roughness_target = (1-roughness_target.clamp(0,1)) * (tgt_max - tgt_min) + tgt_min
        return (roughness - roughness_target).abs()

    def energy_cons_loss_fn(self, brdf, n_d_i, num_train_incident_samples):
        d_omega = 2 * torch.pi / num_train_incident_samples
        brdf_hemisphere_integ = (brdf * n_d_i * d_omega).sum(dim=1) # [N, 3]
        return torch.nn.functional.relu(brdf_hemisphere_integ - 1)

    def brdf_weighted_specular_loss_fn(self, NDF, diffuse_brdf):
        return torch.softmax(NDF.detach(), dim=1) * diffuse_brdf

    def forward(self, model_outputs, ground_truth, progress_iter):
        ZERO = torch.tensor(0.0).cuda().float()
        mat_loss = unweighted_mat_loss = geo_loss = ZERO
        # Track losses in dict for logging
        loss_dict = {}

        if self.phase in ['mat', 'joint']:
            rgb_gt = ground_truth['rgb'].cuda().reshape(-1, 3) # [N, 3]
            masks = model_outputs['render_masks'] # [N]

            # get rid of invalid pixels (because of undistortion)
            if self.remove_black:
                valid = (rgb_gt > 0).long().sum(-1) > 0
                masks = masks & valid

            # We mask out training examples that are not within the foreground mask
            masks = masks.float() # [N]
            mask_sum = masks.sum().clamp(min=1e-7)

            masks = masks.unsqueeze(dim=1) # [N, 1]

            # Note: we need to weigh each training example by the inverse PDF.
            # This ensures that the loss gradient is unbiased.
            # We save the unweighted sub-losses in wandb to make it easier to compare across methods

            # rendered rgb
            rgb_values = model_outputs['rgb_values'] # [N, 3]
            rgb_loss = self.rgb_loss_fn(rgb_values, rgb_gt) # [N, 3]
            unweighted_rgb_loss = (rgb_loss * masks).sum() / mask_sum / 3
            loss_dict['rgb_loss'] = unweighted_rgb_loss.clone().detach()

            # BRDF smoothness prior
            rgb_grad = ground_truth['rgb_grad'].cuda().reshape(-1) # [N]
            brdf_grads = model_outputs['brdf_grads']               # [N, 2, 3]
            smooth_loss = self.brdf_smoothness_loss_fn(brdf_grads, rgb_grad) # [N]
            unweighted_smooth_loss = (smooth_loss * masks.squeeze()).mean()
            loss_dict['smooth_loss'] = unweighted_smooth_loss.clone().detach()

            # lambertian assumption
            roughness = model_outputs['roughness'] # [N, 1]
            metallic = model_outputs['metallic'] # [N, 1]
            lambertian_loss = self.lambertian_loss_fn(roughness, metallic) # [N, 1]
            unweighted_lambertian_loss = (lambertian_loss * masks).sum() / mask_sum
            loss_dict['lambertian_loss'] = unweighted_lambertian_loss.clone().detach()

            # inter-reflection loss
            # NeRF and NeILF output should still be part of computational graph
            reflection_mask = model_outputs['trace_mask'].clone().detach() # [N', S]
            reflection_nn_rgb = model_outputs['trace_nn_rgb'] # [M, 3]
            reflection_render_rgb = model_outputs['trace_render_rgb'] # [M, 3]
            reflection_render_rgb = general.scale_grad(reflection_render_rgb, self.reflection_grad_scale)

            if self.mono_neilf:
                gray_weight = torch.tensor([[0.2989, 0.5870, 0.1140]], dtype=reflection_render_rgb.dtype, device=reflection_render_rgb.device)
                reflection_render_rgb = (reflection_render_rgb * gray_weight).sum(dim=-1, keepdim=True)
                reflection_nn_rgb = reflection_nn_rgb[..., :1]

            reflection_loss = self.reflection_loss_fn(reflection_nn_rgb, reflection_render_rgb) # [M, 3]
            unweighted_reflection_loss = reflection_loss.sum() / (reflection_nn_rgb.numel() + 1e-9)
            loss_dict['reflection_loss'] = unweighted_reflection_loss.clone().detach()

            # variance guidance
            var_loss = ZERO
            unweighted_var_loss = ZERO
            if 'rgb_var' in model_outputs and progress_iter <= 1000:
                rgb_variance = model_outputs['rgb_var']
                var_loss = variance_loss_fn(rgb_variance, roughness)
                unweighted_var_loss = (var_loss * masks).sum() / mask_sum
            loss_dict['var_loss'] = unweighted_var_loss.clone().detach()

            # conservation of energy constraint
            brdf = model_outputs['brdf'] # [N', S, 3]
            n_d_i = model_outputs['n_d_i'] # [N', S, 1]
            energy_cons_loss = self.energy_cons_loss_fn(brdf, n_d_i, self.num_train_incident_samples) # [N', 3]
            unweighted_energy_cons_loss = energy_cons_loss.sum() / mask_sum
            loss_dict['energy_cons_loss'] = unweighted_energy_cons_loss.clone().detach()

            # BRDF weighted specular loss
            NDF = model_outputs['NDF']
            diffuse_brdf = model_outputs['diffuse_brdf'] # [N', S, 3]
            brdf_weighted_specular_loss = self.brdf_weighted_specular_loss_fn(NDF, diffuse_brdf) # [N', S, 3]
            unweighted_brdf_weighted_specular_loss = brdf_weighted_specular_loss.sum() / self.num_train_incident_samples / mask_sum / 3
            loss_dict['brdf_weighted_specular_loss'] = unweighted_brdf_weighted_specular_loss.clone().detach()

            unweighted_mat_loss =  unweighted_rgb_loss + \
                        self.smth_weight * unweighted_smooth_loss + \
                        self.lam_weight * unweighted_lambertian_loss + \
                        self.ref_weight * unweighted_reflection_loss + \
                        self.var_weight * unweighted_var_loss + \
                        self.energy_cons_weight * unweighted_energy_cons_loss + \
                        self.brdf_weighted_specular_weight * unweighted_brdf_weighted_specular_loss

        # slf loss
        if self.phase in ['geo', 'joint']:
            geo_loss = self.geo_loss_fn(model_outputs, ground_truth, progress_iter)

        loss = unweighted_mat_loss + geo_loss
        loss_dict['loss'] = loss.clone().detach()

        return loss, loss_dict
