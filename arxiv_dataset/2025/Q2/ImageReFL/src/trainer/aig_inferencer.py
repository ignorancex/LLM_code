import torch

from src.constants.dataset import DatasetColumns
from src.trainer.combined_generation_inference import CombinedGenerationInferencer


class AIGInferencer(CombinedGenerationInferencer):
    def run_inference(self):
        part_logs = {}
        for i, aig_p in enumerate(self.cfg_trainer.aig_ps):
            self.global_image_index = 0
            self.aig_p = aig_p
            self.start_timestep_index = aig_p

            for part, dataloader in self.evaluation_dataloaders.items():
                logs = self._inference_part(
                    part,
                    dataloader,
                )
                part_logs[part] = logs

            self.writer.set_step(i)
            for metric_name in self.all_metrics.keys():
                self.writer.add_scalar(
                    f"{metric_name}", self.all_metrics.avg(metric_name)
                )
            self.all_metrics.reset()
        return part_logs

    def _sample_image(self, batch: dict[str, torch.Tensor]):
        T = self.cfg_trainer.end_timestep_index
        self.model.set_timesteps(T, device=self.device)
        self.original_model.set_timesteps(T, device=self.device)
        batch_size = batch[DatasetColumns.tokenized_text.name].shape[0]

        latents = self.original_model.get_latents(
            batch_size=batch_size, device=self.device
        )

        original_model_hid_state = self.original_model.get_encoder_hidden_states(
            batch=batch, do_classifier_free_guidance=False
        )

        model_hid_state = self.model.get_encoder_hidden_states(
            batch=batch,
            do_classifier_free_guidance=False,
        )

        for step_index in range(T):
            real_step = T - step_index
            scale = 1 - ((T - real_step) / T) ** self.aig_p

            original_model_noise_pred = self.original_model.get_noise_prediction(
                latents=latents,
                timestep_index=step_index,
                encoder_hidden_states=original_model_hid_state,
                do_classifier_free_guidance=False,
            )

            model_noise_pred = self.model.get_noise_prediction(
                latents=latents,
                timestep_index=step_index,
                encoder_hidden_states=model_hid_state,
                do_classifier_free_guidance=False,
            )

            noise_pred = original_model_noise_pred * scale + model_noise_pred * (
                1 - scale
            )
            latents = self.original_model.sample_next_latents(
                latents=latents,
                timestep_index=step_index,
                noise_pred=noise_pred,
                return_pred_original=(step_index == T - 1),
            )

        latents /= self.model.vae.config.scaling_factor

        raw_image = self.model.vae.decode(latents).sample
        reward_images, pil_images = self.model.get_reward_image(
            raw_image
        ), self.model.get_pil_image(raw_image)

        batch["image"] = reward_images
        batch["pil_images"] = pil_images
