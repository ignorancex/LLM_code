import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import ClapTextModelWithProjection, RobertaTokenizer, RobertaTokenizerFast, SpeechT5HifiGan, CLIPTextModel, CLIPTokenizer
from torchvision import transforms

from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL, UNet2DConditionModel, UNetMotionModel, UNet3DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
# from diffusers.utils import is_accelerate_available, logging, randn_tensor, replace_example_docstring
from diffusers.utils import is_accelerate_available, logging, BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from .unet_svg import UNetModelSVG
from .scheduler_ddim_rev import DDIMSchedulerRev



logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def tensor2vid(video: torch.Tensor, processor, output_type="np"):
    # Based on:
    # https://github.com/modelscope/modelscope/blob/1509fdb973e5871f37148a4b5e5964cafd43e64d/modelscope/pipelines/multi_modal/text_to_video_synthesis_pipeline.py#L78

    batch_size, channels, num_frames, height, width = video.shape
    outputs = []
    for batch_idx in range(batch_size):
        batch_vid = video[batch_idx].permute(1, 0, 2, 3)
        batch_output = processor.postprocess(batch_vid, output_type)

        outputs.append(batch_output)

    return outputs


class VideoAudioGenWithSVGPipelineOutput(BaseOutput):
    audios: Union[torch.Tensor, np.ndarray]
    videos: Union[torch.Tensor, np.ndarray]


class VideoAudioGenWithSVGPipeline(DiffusionPipeline):

    def __init__(
        self,
        vae_a: AutoencoderKL,
        text_encoder_a: ClapTextModelWithProjection,
        tokenizer_a: Union[RobertaTokenizer, RobertaTokenizerFast],
        vocoder: SpeechT5HifiGan,
        scheduler_a: KarrasDiffusionSchedulers,
        unet_a: UNet2DConditionModel,
        vae_v: AutoencoderKL,
        text_encoder_v: CLIPTextModel,
        tokenizer_v: CLIPTokenizer,
        scheduler_v: KarrasDiffusionSchedulers,
        unet_v: Union[UNetMotionModel, UNet3DConditionModel],
        unet_svg: UNetModelSVG,
    ):
        super().__init__()

        scheduler_a = DDIMSchedulerRev.from_config(scheduler_a.config)
        scheduler_v = DDIMSchedulerRev.from_config(scheduler_v.config)

        self.register_modules(
            vae_a=vae_a,
            text_encoder_a=text_encoder_a,
            tokenizer_a=tokenizer_a,
            vocoder=vocoder,
            scheduler_a=scheduler_a,
            unet_a=unet_a,
            vae_v=vae_v,
            text_encoder_v=text_encoder_v,
            tokenizer_v=tokenizer_v,
            scheduler_v=scheduler_v,
            unet_v=unet_v,
            unet_svg=unet_svg,
        )

        # self.unet_svg.set_unets(self.unet_a, self.unet_v)
        self.vae_a_scale_factor = 2 ** (len(self.vae_a.config.block_out_channels) - 1)
        self.vae_v_scale_factor = 2 ** (len(self.vae_v.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_v_scale_factor)

    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae and vocoder have their state dicts saved to CPU and then are moved to a `torch.device('meta')
        and loaded to GPU only when their specific submodule has its `forward` method called.
        """
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet_svg, self.unet_a, self.unet_v, self.text_encoder_a, self.text_encoder_v, self.vae_a, self.vae_v, self.vocoder]:
            cpu_offload(cpu_offloaded_model, device)

    def _encode_prompt_a(
        self,
        prompt,
        device,
        do_classifier_free_guidance,
        negative_prompt=None,
    ):
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        text_inputs = self.tokenizer_a(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_a.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask
        untruncated_ids = self.tokenizer_a(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer_a.batch_decode(
                untruncated_ids[:, self.tokenizer_a.model_max_length - 1 : -1]
            )
            logger.warning(
                "The following part of your input was truncated because CLAP can only handle sequences up to"
                f" {self.tokenizer_a.model_max_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder_a(
            text_input_ids.to(device),
            attention_mask=attention_mask.to(device),
        )
        prompt_embeds = prompt_embeds.text_embeds
        # additional L_2 normalization over each hidden-state
        prompt_embeds = F.normalize(prompt_embeds, dim=-1)

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_a.dtype, device=device)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer_a(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            uncond_input_ids = uncond_input.input_ids.to(device)
            attention_mask = uncond_input.attention_mask.to(device)

            negative_prompt_embeds = self.text_encoder_a(
                uncond_input_ids,
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds.text_embeds
            # additional L_2 normalization over each hidden-state
            negative_prompt_embeds = F.normalize(negative_prompt_embeds, dim=-1)

        if do_classifier_free_guidance:
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder_a.dtype, device=device)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    def _encode_prompt_v(
        self,
        prompt,
        device,
        do_classifier_free_guidance,
        negative_prompt=None,
    ):
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        text_inputs = self.tokenizer_v(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_v.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_v(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer_v.batch_decode(
                untruncated_ids[:, self.tokenizer_v.model_max_length - 1 : -1]
            )
            logger.warning(
                "The following part of your input was truncated because CLAP can only handle sequences up to"
                f" {self.tokenizer_v.model_max_length} tokens: {removed_text}"
            )

        if hasattr(self.text_encoder_v.config, "use_attention_mask") and self.text_encoder_v.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        prompt_embeds = self.text_encoder_v(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_v.dtype, device=device)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer_v(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            uncond_input_ids = uncond_input.input_ids.to(device)
            if hasattr(self.text_encoder_v.config, "use_attention_mask") and self.text_encoder_v.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder_v(
                uncond_input_ids,
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder_a.dtype, device=device)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    def decode_latents_a(self, latents):
        latents = 1 / self.vae_a.config.scaling_factor * latents
        mel_spectrogram = self.vae_a.decode(latents).sample
        return mel_spectrogram

    def decode_latents_v(self, latents):
        latents = 1 / self.vae_v.config.scaling_factor * latents

        batch_size, channels, num_frames, height, width = latents.shape
        latents = latents.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)

        image = self.vae_v.decode(latents).sample
        video = (
            image[None, :]
            .reshape(
                (
                    batch_size,
                    num_frames,
                    -1,
                )
                + image.shape[2:]
            )
            .permute(0, 2, 1, 3, 4)
        )
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        video = video.float()
        return video
    
    def mel_spectrogram_to_waveform(self, mel_spectrogram):
        if mel_spectrogram.dim() == 4:
            mel_spectrogram = mel_spectrogram.squeeze(1)

        waveform = self.vocoder(mel_spectrogram)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        waveform = waveform.cpu().float()
        return waveform

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta_a = "eta" in set(inspect.signature(self.scheduler_a.step).parameters.keys())
        accepts_eta_v = "eta" in set(inspect.signature(self.scheduler_v.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta_a and accepts_eta_v:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator_a = "generator" in set(inspect.signature(self.scheduler_a.step).parameters.keys())
        accepts_generator_v = "generator" in set(inspect.signature(self.scheduler_v.step).parameters.keys())
        if accepts_generator_a and accepts_generator_v:
            extra_step_kwargs["generator"] = generator

        return extra_step_kwargs

    def check_inputs(
        self,
    ):
        pass

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents with width->self.vocoder.config.model_in_dim
    def prepare_latents_a(self, batch_size, num_channels_latents, height, dtype, device, generator, latents=None):
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_a_scale_factor,
            self.vocoder.config.model_in_dim // self.vae_a_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler_a.init_noise_sigma
        return latents

    def prepare_latents_v(self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None):
        shape = (
            batch_size,
            num_channels_latents,
            num_frames,
            height // self.vae_v_scale_factor,
            width // self.vae_v_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler_v.init_noise_sigma
        return latents

    @torch.no_grad()
    def __call__(
        self,
        prompt_a: Union[str, List[str]] = None,
        prompt_v: Union[str, List[str]] = None,
        length_in_s: Optional[float] = None,
        height_v: Optional[int] = None,
        width_v: Optional[int] = None,
        num_inference_steps: int = 50,
        frame_rate: int = 8,
        guidance_scale_a: float = 2.5,
        guidance_scale_v: float = 7.5,
        negative_prompt_a: Optional[Union[str, List[str]]] = None,
        negative_prompt_v: Optional[Union[str, List[str]]] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents_a: Optional[torch.FloatTensor] = None,
        latents_v: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
        output_type: Optional[str] = "np",
    ):
        # 0. Convert audio input length from seconds to spectrogram height
        if length_in_s is None:
            length_in_s = 2.0
        batch_size = len(prompt_a) if isinstance(prompt_a, list) else 1

        num_frames = int(frame_rate * length_in_s)
        height_v = height_v or self.unet_v.config.sample_size * self.vae_v_scale_factor
        width_v = width_v or self.unet_v.config.sample_size * self.vae_v_scale_factor

        vocoder_upsample_factor = np.prod(self.vocoder.config.upsample_rates) / self.vocoder.config.sampling_rate
        height_a = int(length_in_s / vocoder_upsample_factor)
        original_waveform_length = int(length_in_s * self.vocoder.config.sampling_rate)
        if height_a % (self.vae_a_scale_factor * (2 ** self.unet_a.num_upsamplers)) != 0:
            r = self.vae_a_scale_factor * (2 ** self.unet_a.num_upsamplers)
            height_a = int(np.ceil(height_a / r)) * r
            logger.info(
                f"Audio length in seconds {length_in_s} is increased to {height_a * vocoder_upsample_factor} "
                f"so that it can be handled by the model. It will be cut to {length_in_s} after the "
                f"denoising process."
            )

        # 1. Check inputs. Raise error if not correct
        self.check_inputs()

        # 2. Define call parameters            
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale_a > 1.0 or guidance_scale_v > 1.0

        # 3. Encode input prompt
        prompt_embeds_a = self._encode_prompt_a(prompt_a, device, do_classifier_free_guidance, negative_prompt_a)
        prompt_embeds_v = self._encode_prompt_v(prompt_v, device, do_classifier_free_guidance, negative_prompt_v)

        # 4. Prepare timesteps
        self.scheduler_a.set_timesteps(num_inference_steps, device=device)
        self.scheduler_v.set_timesteps(num_inference_steps, device=device)
        timesteps_a = self.scheduler_a.timesteps
        timesteps_v = self.scheduler_v.timesteps

        # 5. Prepare latent variables
        latents_a = self.prepare_latents_a(
            batch_size,
            self.unet_a.config.in_channels,
            height_a,
            prompt_embeds_a.dtype,
            device,
            generator,
            latents_a,
        )
        latents_v = self.prepare_latents_v(
            batch_size,
            self.unet_v.config.in_channels,
            num_frames,
            height_v,
            width_v,
            prompt_embeds_v.dtype,
            device,
            generator,
            latents_v,
        )

        # 6. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        context = [None, prompt_embeds_v]
        class_labels = [prompt_embeds_a, None]
        xtype = ["audio", "video"]
        block_connection = None
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, (t_a, t_v) in enumerate(zip(timesteps_a, timesteps_v)):
                # expand the latents if we are doing classifier free guidance
                latents_a_scaled = self.scheduler_a.scale_model_input(latents_a, t_a)
                latents_v_scaled = self.scheduler_v.scale_model_input(latents_v, t_v)
                if do_classifier_free_guidance:
                    latent_model_input = [torch.cat([latents_a_scaled] * 2), torch.cat([latents_v_scaled] * 2)]
                else:
                    latent_model_input = [latents_a_scaled, latents_v_scaled]

                # predict the noise residual
                if self.unet_svg.connector_audio_config["connector_out_input"] == "x_cur":
                    x_con = latent_model_input
                    t_con = [t_a, t_v]
                elif self.unet_svg.connector_audio_config["connector_out_input"] == "x_pred":
                    x_con = []
                    for j in range(len(latent_model_input)):
                        if i == 0:
                            x_pred = torch.zeros_like(latent_model_input[j])                            
                        else:
                            latent_pred_input = [torch.cat([latent_pred[0]] * 2), torch.cat([latent_pred[1]] * 2)]
                            x_pred = latent_pred_input[j]
                        x_con.append(torch.cat([latent_model_input[j], x_pred], dim=1))
                    t_con = [t_a, t_v]
                noise_pred = self.unet_svg([self.unet_a, self.unet_v], latent_model_input, [t_a, t_v], context, class_labels, xtype=xtype, block_connection=block_connection, x_con=x_con, t_con=t_con)

                # perform guidance
                if do_classifier_free_guidance:
                    for j in range(len(noise_pred)):
                        if xtype[j] == "audio":
                            gs = guidance_scale_a
                        elif xtype[j] == "video":
                            gs = guidance_scale_v

                        noise_pred_uncond, noise_pred_cond = noise_pred[j].chunk(2)
                        noise_pred[j] = noise_pred_uncond + gs * (noise_pred_cond - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latent_pred = []
                for j in range(len(xtype)):
                    if xtype[j] == "audio":
                        pred = self.scheduler_a.step(noise_pred[j], t_a, latents_a, use_clipped_model_output=True, **extra_step_kwargs)
                        latents_a = pred.prev_sample
                    elif xtype[j] == "video":
                        pred = self.scheduler_v.step(noise_pred[j], t_v, latents_v, use_clipped_model_output=True, **extra_step_kwargs)
                        latents_v = pred.prev_sample
                    else:
                        raise
                    if self.unet_svg.connector_audio_config["connector_out_input"] == "x_pred":
                        latent_pred.append(pred.pred_original_sample)

                # call the callback, if provided
                if i == num_inference_steps - 1 or (i + 1) % self.scheduler_a.order == 0:
                    progress_bar.update()

        # 8. Post-processing
        # audio
        mel_spectrogram = self.decode_latents_a(latents_a)
        audio = self.mel_spectrogram_to_waveform(mel_spectrogram)
        audio = audio[:, :original_waveform_length]
        # video
        video = self.decode_latents_v(latents_v)

        if output_type == "np":
            audio = audio.numpy()
            video = tensor2vid(video, self.image_processor, output_type="np")

        if not return_dict:
            return (audio, video)

        return VideoAudioGenWithSVGPipelineOutput(audios=audio, videos=video)
    
    