# Author: Bingxin Ke
# Last modified: 2023-12-15

from typing import List, Dict, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm.auto import tqdm
from PIL import Image

from diffusers import (
    DiffusionPipeline,
    UNet2DConditionModel,
    AutoencoderKL,
    ControlNetModel
)
from diffusers.utils import BaseOutput
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor

from .util.image_util import chw2hwc, colorize_depth_maps, resize_max_res, norm_to_rgb
from .util.batchsize import find_batch_size
from .util.ensemble import ensemble_depths
from marigold.image_projector import ImageProjModel
from marigold.models import DPTHead, CustomUNet2DConditionModel
import matplotlib.pyplot as plt

from marigold.util.scheduler_customized import DDIMSchedulerCustomized as DDIMScheduler

class MarigoldAlbedoShadingOutput(BaseOutput):

    albedo: np.ndarray
    shading: np.ndarray
    albedo_colored: Image.Image
    shading_colored: Image.Image
    uncertainty: Union[None, np.ndarray]

class MarigoldNormalOutput(BaseOutput):

    normal_np: np.ndarray
    normal_colored: Image.Image
    uncertainty: Union[None, np.ndarray]

class MarigoldDepthOutput(BaseOutput):
    """
    Output class for Marigold monocular depth prediction pipeline.

    Args:
        depth_np (np.ndarray):
            Predicted depth map, with depth values in the range of [0, 1]
        depth_colored (PIL.Image.Image):
            Colorized depth map, with the shape of [3, H, W] and values in [0, 1]
        uncertainty (None` or `np.ndarray):
            Uncalibrated uncertainty(MAD, median absolute deviation) coming from ensembling.
    """

    depth_np: np.ndarray
    depth_colored: Image.Image
    uncertainty: Union[None, np.ndarray]


class MarigoldSegOutput(BaseOutput):
    """
    Output class for Marigold monocular depth prediction pipeline.

    Args:
        depth_np (np.ndarray):
            Predicted depth map, with depth values in the range of [0, 1]
        depth_colored (PIL.Image.Image):
            Colorized depth map, with the shape of [3, H, W] and values in [0, 1]
        uncertainty (None` or `np.ndarray):
            Uncalibrated uncertainty(MAD, median absolute deviation) coming from ensembling.
    """
    # seg_np: np.ndarray
    seg_colored: Image.Image
    uncertainty: Union[None, np.ndarray]


class MarigoldPipeline(DiffusionPipeline):
    """
    Pipeline for monocular depth estimation using Marigold: https://arxiv.org/abs/2312.02145.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        unet (UNet2DConditionModel):
            Conditional U-Net to denoise the depth latent, conditioned on image latent.
        vae (AutoencoderKL):
            Variational Auto-Encoder (VAE) Model to encode and decode images and depth maps
            to and from latent representations.
        scheduler (DDIMScheduler):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        text_encoder (CLIPTextModel):
            Text-encoder, for empty text embedding.
        tokenizer (CLIPTokenizer):
            CLIP tokenizer.
    """

    rgb_latent_scale_factor = 0.18215
    depth_latent_scale_factor = 0.18215
    seg_latent_scale_factor = 0.18215
    normal_latent_scale_factor = 0.18215

    def __init__(
        self,
        unet: Union[UNet2DConditionModel, CustomUNet2DConditionModel],
        vae: AutoencoderKL,
        scheduler: DDIMScheduler,
        tokenizer: CLIPTokenizer,
        text_embeds: Union[torch.Tensor, None],
        text_encoder: Union[CLIPTextModel, None],
        image_encoder: Union[CLIPVisionModelWithProjection, None],
        image_projector: Union[ImageProjModel, None],
        controlnet: Union[ControlNetModel, None],
        customized_head: Union[DPTHead, None],
    ):
        super().__init__()

        register_dict = dict(
            unet=unet,
            vae=vae,
            text_embeds=text_embeds,
            scheduler=scheduler,
            tokenizer=tokenizer,
        )

        if image_encoder is not None:
            self.text_embed_flag = False
            self.vision_embed_flag = True
            register_dict['image_encoder'] = image_encoder
            register_dict['text_encoder'] = None
        elif text_encoder is not None:
            self.text_embed_flag = True
            self.vision_embed_flag = False
            register_dict['image_encoder'] = None
            register_dict['text_encoder'] = text_encoder
            self.empty_text_embed = text_embeds
        else:
            raise ValueError

        self.text_encoder = text_encoder
        self.image_encoder = image_encoder

        if isinstance(controlnet, ControlNetModel):
            register_dict['controlnet'] = controlnet
        else:
            self.controlnet = None
            register_dict['controlnet'] = None
        
        if customized_head is None:
            self.customized_head = None
        else:
            self.customized_head = customized_head
            
        register_dict['customized_head'] = self.customized_head

        register_dict['image_projector'] = image_projector
        self.image_projector = image_projector

        self.register_modules(**register_dict)

        # self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        # self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        # Copied from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline.get_guidance_scale_embedding
    def get_guidance_scale_embedding(self, w, embedding_dim=512, dtype=torch.float32):
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            timesteps (`torch.Tensor`):
                generate embedding vectors at these timesteps
            embedding_dim (`int`, *optional*, defaults to 512):
                dimension of the embeddings to generate
            dtype:
                data type of the generated embeddings

        Returns:
            `torch.FloatTensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb


    @property
    def guidance_scale(self):
        return self._guidance_scale

    
    @torch.no_grad()
    def __call__(
        self,
        input_image: Union[Image.Image, torch.Tensor],
        denoising_steps: int = 10,
        # num_inference_steps: int = 10,
        ensemble_size: int = 10,
        processing_res: int = 768,
        match_input_res: bool = True,
        batch_size: int = 0,
        color_map: str = "Spectral",
        show_progress_bar: bool = True,
        ensemble_kwargs: Dict = None,
        mode: str = 'depth',
        rgb_paths = [],
        seed = None,
    ) -> MarigoldDepthOutput:
        """
        Function invoked when calling the pipeline.

        Args:
            input_image (Image):
                Input RGB (or gray-scale) image.
            processing_res (int, optional):
                Maximum resolution of processing.
                If set to 0: will not resize at all.
                Defaults to 768.
            match_input_res (bool, optional):
                Resize depth prediction to match input resolution.
                Only valid if `limit_input_res` is not None.
                Defaults to True.
            denoising_steps (int, optional):
                Number of diffusion denoising steps (DDIM) during inference.
                Defaults to 10.
            ensemble_size (int, optional):
                Number of predictions to be ensembled.
                Defaults to 10.
            batch_size (int, optional):
                Inference batch size, no bigger than `num_ensemble`.
                If set to 0, the script will automatically decide the proper batch size.
                Defaults to 0.
            show_progress_bar (bool, optional):
                Display a progress bar of diffusion denoising.
                Defaults to True.
            color_map (str, optional):
                Colormap used to colorize the depth map.
                Defaults to "Spectral".
            ensemble_kwargs ()
        Returns:
            `MarigoldDepthOutput`
        """

        device = self.device

        if mode == 'depth':
            task_channel_num = 1
        elif mode == 'seg':
            task_channel_num = 3
        elif mode == 'normal':
            task_channel_num = 3
        elif mode == 'feature':
            task_channel_num = 4
        else:
            raise ValueError

        if not match_input_res:
            assert (
                processing_res is not None
            ), "Value error: `resize_output_back` is only valid with "
        assert processing_res >= 0
        assert denoising_steps >= 1
        assert ensemble_size >= 1

        # ----------------- Image Preprocess -----------------

        if type(input_image) == torch.Tensor: # [B, 3, H, W]            
            rgb_norm = input_image.to(device)
            input_size = input_image.shape[2:]
            assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0
            bs_imgs = rgb_norm.shape[0]
            assert len(rgb_paths) > 0
            clip_images = []
            for rgb_path in rgb_paths:
                clip_image_i = Image.open(rgb_path)
                clip_image_i = CLIPImageProcessor()(images=clip_image_i, return_tensors="pt").pixel_values[0] # TODO: clip_image for augmentation?
                clip_images.append(clip_image_i)
            clip_image = np.stack(clip_images, axis=0)
            clip_image = torch.from_numpy(clip_image)
            
            # np.save('temp_rgb_batch.npy', rgb_norm.detach().cpu().numpy())
            # plt.imsave('temp_rgb_batch.png', rgb_norm[0].permute(1,2,0).detach().cpu().numpy())
            # import pdb
            # pdb.set_trace()
        else:
            if len(rgb_paths) > 0 and 'kitti' in rgb_paths[0]:
                # kb crop
                height = input_image.size[1]
                width = input_image.size[0]
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                input_image = input_image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))

            input_size = (input_image.size[1], input_image.size[0])
            # Resize image
            if processing_res > 0:
                input_image = resize_max_res(
                    input_image, max_edge_resolution=processing_res
                )
            # Convert the image to RGB, to 1.remove the alpha channel 2.convert B&W to 3-channel
            input_image = input_image.convert("RGB")
            clip_image = CLIPImageProcessor()(images=input_image, return_tensors="pt").pixel_values[0]

            image = np.asarray(input_image)

            # Normalize rgb values
            rgb = np.transpose(image, (2, 0, 1))  # [H, W, rgb] -> [rgb, H, W]
            rgb_norm = rgb / 255.0 * 2.0 - 1.0
            rgb_norm = torch.from_numpy(rgb_norm).to(self.dtype)
            rgb_norm = rgb_norm[None].to(device)
            assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0
            bs_imgs = 1

            # np.save('temp_rgb_read.npy', rgb_norm.detach().cpu().numpy())
            # plt.imsave('temp_rgb_read.png', rgb_norm[0].permute(1,2,0).detach().cpu().numpy())
            # import pdb
            # pdb.set_trace()


        # ----------------- Predicting depth -----------------
        # Batch repeated input image
        duplicated_rgb = torch.stack([rgb_norm] * ensemble_size) # [ensemble_size, bs_imgs, 3, H, W]
        duplicated_clip_rgb = torch.stack([clip_image] * ensemble_size)

        duplicated_rgb = duplicated_rgb.view(-1, 3, duplicated_rgb.shape[-2], duplicated_rgb.shape[-1]) # [ensemble_size * bs_imgs, 3, H, W]
        duplicated_clip_rgb = duplicated_clip_rgb.view(-1, 3, duplicated_clip_rgb.shape[-2], duplicated_clip_rgb.shape[-1])

        # single_rgb_dataset = TensorDataset(duplicated_rgb)
        single_rgb_dataset = TensorDataset(duplicated_rgb, duplicated_clip_rgb)
        if batch_size > 0:
            # _bs = batch_size
            _bs = find_batch_size(
                ensemble_size=ensemble_size * batch_size,
                input_res=max(rgb_norm.shape[1:]),
                dtype=self.dtype,
            )
        else:
            _bs = find_batch_size(
                ensemble_size=ensemble_size, 
                input_res=max(rgb_norm.shape[1:]),
                dtype=self.dtype,
            )
        print('_bs :', _bs)

        single_rgb_loader = DataLoader(
            single_rgb_dataset, batch_size=_bs, shuffle=False
        )

        # Predict depth maps (batched)
        depth_pred_ls = []
        if show_progress_bar:
            iterable = tqdm(
                single_rgb_loader, desc=" " * 2 + "Inference batches", leave=False
            )
        else:
            iterable = single_rgb_loader
        for batch in iterable:
            (batched_img, batch_clip_img) = batch
            depth_pred_raw = self.single_infer(
                rgb_in=batched_img,
                clip_rgb_in=batch_clip_img,
                num_inference_steps=denoising_steps,
                show_pbar=show_progress_bar,
                mode=mode,
                seed=seed,
            )
            depth_pred_ls.append(depth_pred_raw.detach().clone())
        depth_preds = torch.concat(depth_pred_ls, axis=0).squeeze()
        if mode != 'feature':
            depth_preds = depth_preds.view(ensemble_size, bs_imgs, task_channel_num, depth_preds.shape[-2], depth_preds.shape[-1]) # [ensemble_size, bs_imgs, task_channel_num, H, W]
        else:
            depth_preds = depth_preds.view(ensemble_size, bs_imgs, denoising_steps, task_channel_num, depth_preds.shape[-2], depth_preds.shape[-1]) # [ensemble_size, bs_imgs, steps, task_channel_num, H, W]
        torch.cuda.empty_cache()  # clear vram cache for ensembling

        # ----------------- Test-time ensembling -----------------
        if mode == 'depth':
            if ensemble_size > 1:
                depth_pred_list = []
                pred_uncert_list = []
                for i in range(bs_imgs):
                    depth_pred_i, pred_uncert_i = ensemble_depths(
                        depth_preds[:, i, 0], **(ensemble_kwargs or {})
                    )
                    depth_pred_list.append(depth_pred_i)
                    pred_uncert_list.append(pred_uncert_i)
                depth_preds = torch.stack(depth_pred_list, dim=0)[:, None] # [bs_imgs, task_channel_num, H, W]
                pred_uncert = torch.stack(pred_uncert_list, dim=0)[:, None].squeeze()
            else:
                depth_preds = depth_preds.mean(dim=0) # [bs_imgs, task_channel_num, H, W]
                pred_uncert = None
        else:
            depth_preds = depth_preds.mean(dim=0) # [bs_imgs, task_channel_num, H, W] or [bs_imgs, steps, task_channel_num, H, W]
        
        if match_input_res:
            if mode == 'depth' or mode == 'normal':
                depth_preds = F.interpolate(depth_preds, input_size, mode='bilinear')
            elif mode == 'seg':
                depth_preds = F.interpolate(depth_preds, input_size, mode='nearest')
            elif mode == 'feature':
                pass
            else:
                raise NotImplementedError

        # ----------------- Post processing -----------------
        if mode == 'depth':
            depth_preds = depth_preds[:, 0] # [bs_imgs, H, W]
            # Scale prediction to [0, 1]
            min_d = depth_preds.min(dim=-1)[0].min(dim=-1)[0]
            max_d = depth_preds.max(dim=-1)[0].max(dim=-1)[0]
            depth_preds = (depth_preds - min_d[:, None, None]) / (max_d[:, None, None] - min_d[:, None, None])

            # Resize back to original resolution
                # pred_img = Image.fromarray(depth_preds)
                # pred_img = pred_img.resize(input_size)
                # depth_preds = np.asarray(pred_img)
            
            # Convert to numpy
            depth_preds = depth_preds.cpu().numpy().astype(np.float32)

            # Clip output range
            depth_preds = depth_preds.clip(0, 1)

            # Colorize
            depth_colored_img_list = []
            for i in range(depth_preds.shape[0]):
                depth_colored_i = colorize_depth_maps(
                    depth_preds[i], 0, 1, cmap=color_map
                ).squeeze()  # [3, H, W], value in (0, 1)
                depth_colored_i = (depth_colored_i * 255).astype(np.uint8)
                depth_colored_i_hwc = chw2hwc(depth_colored_i)
                depth_colored_i_image = Image.fromarray(depth_colored_i_hwc)
                depth_colored_img_list.append(depth_colored_i_image)

            return MarigoldDepthOutput(
                depth_np=np.squeeze(depth_preds),
                depth_colored=depth_colored_img_list[0] if len(depth_colored_img_list) == 1 else depth_colored_img_list,
                uncertainty=pred_uncert,
            )

        elif mode == 'seg':
            # Clip output range
            seg_colored = depth_preds.clip(0, 255).cpu().numpy().astype(np.uint8)

            seg_colored_img_list = []
            for i in range(seg_colored.shape[0]):
                seg_colored_hwc_i = chw2hwc(seg_colored[i])
                seg_colored_img_i = Image.fromarray(seg_colored_hwc_i).resize((input_size[1], input_size[0]))
                seg_colored_img_list.append(seg_colored_img_i)

            return MarigoldSegOutput(
                seg_colored=seg_colored_img_list[0] if len(seg_colored_img_list) == 1 else seg_colored_img_list,
                uncertainty=None,
            )

        elif mode == 'normal':
            normal = depth_preds.clip(-1, 1).cpu().numpy() # [-1, 1]

            normal_colored_img_list = []
            for i in range(normal.shape[0]):
                normal_colored_i = norm_to_rgb(normal[i])
                normal_colored_hwc_i = chw2hwc(normal_colored_i)
                normal_colored_img_i = Image.fromarray(normal_colored_hwc_i).resize((input_size[1], input_size[0]))
                normal_colored_img_list.append(normal_colored_img_i)

            return MarigoldNormalOutput(
                normal_np=np.squeeze(normal),
                normal_colored=normal_colored_img_list[0] if len(normal_colored_img_list) == 1 else normal_colored_img_list,
                uncertainty=None,
            )
        
        elif mode == 'feature':
            # depth_preds: [B, steps, 4, H/8, W/8] for when mode is 'feature'
            return np.squeeze(depth_preds.detach().cpu().numpy())

        else:
            raise NotImplementedError
    
    def encode_clip_feature(self, clip_rgb_in=None):
        """
        Encode text embedding for empty prompt
        """
        
        if self.text_encoder is not None:
            prompt = ""
            text_inputs = self.tokenizer(
                prompt,
                padding="do_not_pad",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
            text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype)
            return text_embed

        if self.image_encoder is not None and clip_rgb_in is not None:
            image_embeds = self.image_encoder(clip_rgb_in.to(self.image_encoder.device)).image_embeds
            
            if self.image_projector is not None:
                image_embeds = self.image_projector(image_embeds)
                # bs, 1, 1024
                # image_embeds = image_embeds.unsqueeze(1)
                # image_embeds = torch.cat([image_embeds, ip_tokens], dim=1)
            else:
                image_embeds = image_embeds.unsqueeze(1)

            return image_embeds

    @torch.no_grad()
    def single_infer(
        self, 
        rgb_in: torch.Tensor, 
        clip_rgb_in: torch.Tensor, 
        num_inference_steps: int, 
        show_pbar: bool,
        mode: str = 'depth',
        seed = None,
    ) -> torch.Tensor:
        """
        Perform an individual depth prediction without ensembling.

        Args:
            rgb_in (torch.Tensor):
                Input RGB image.
            num_inference_steps (int):
                Number of diffusion denoising steps (DDIM) during inference.
            show_pbar (bool):
                Display a progress bar of diffusion denoising.

        Returns:
            torch.Tensor: Predicted depth map.
        """
        device = rgb_in.device
        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps  # [T]
        # tensor([951, 901, 851, 801, 751, 701, 651, 601, 551, 501, 451, 401, 351, 301,
        # 251, 201, 151, 101,  51,   1], device='cuda:0')
        # Encode image
        rgb_latent = self.encode_rgb(rgb_in)   # [10,3,768,512] [10,4,96,64]

        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
            # Initial depth map (noise)
            depth_latent = torch.randn(
                rgb_latent.shape[1:], device=device, dtype=self.dtype, generator=generator)[None].repeat(rgb_latent.shape[0], 1, 1, 1)  # [B, 4, h, w]
        else:
            generator = None
            # Initial depth map (noise)
            depth_latent = torch.randn(
                rgb_latent.shape, device=device, dtype=self.dtype)  # [B, 4, h, w]

        # import pdb
        # pdb.set_trace()

        if self.text_embed_flag:
            if self.empty_text_embed is not None:
                batch_embed = self.empty_text_embed  # [B, 2, 1024]    [1,2,1024]
            else:
                batch_embed = self.encode_clip_feature(clip_rgb_in)
        elif self.vision_embed_flag:
            batch_embed = self.encode_clip_feature(clip_rgb_in)
        else:
            raise ValueError
        
        batch_embed = batch_embed.repeat((rgb_latent.shape[0], 1, 1)).to(device)   # [10,2,1024]

        # Denoising loop
        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)

        scheduler_z0_list = []
        for i, t in iterable:
            if self.controlnet is None:
                unet_input = torch.cat(
                    [rgb_latent, depth_latent], dim=1
                )  # this order is important  将rgb和depth noise concate到一起  [10,8,96,64]

                # predict the noise residual
                unet_output = self.unet(
                    unet_input, t, encoder_hidden_states=batch_embed
                )  # [B, 4, h, w]

                noise_pred = unet_output.sample
                if self.customized_head:
                    assert isinstance(self.unet, CustomUNet2DConditionModel) or isinstance(self.unet, UNet2DConditionModel)
                    if self.customized_head.in_channels == 4:
                        unet_feature = unet_output.sample
                    elif self.customized_head.in_channels == 320:
                        unet_feature = unet_output.sample_320
                    else:
                        raise ValueError
            else:
                # if depth_latent.shape[-1] == 71 or depth_latent.shape[-2] == 71:
                #     import pdb
                #     pdb.set_trace()

                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    depth_latent, # 10, 4, 96, 72
                    t, # 
                    encoder_hidden_states=batch_embed, # 10, 2, 1024
                    controlnet_cond=rgb_in, # 10, 3, 768, 576
                    return_dict=False,
                )

                if self.customized_head:
                    raise NotImplementedError

                # Predict the noise residual
                noise_pred = self.unet(
                    depth_latent,
                    t,
                    encoder_hidden_states=batch_embed,
                    down_block_additional_residuals=[
                        sample.to(dtype=self.dtype) for sample in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=self.dtype),
                ).sample # [B, 4, h, w]

            # compute the previous noisy sample x_t -> x_t-1
            step_output = self.scheduler.step(noise_pred, t, depth_latent)
            depth_latent = step_output.prev_sample # [B, 4, h, w]
            if mode == 'feature':
                scheduler_z0_list.append(step_output.pred_original_sample)
        
        depth_latent = step_output.pred_original_sample

        if mode == 'depth':
            if self.customized_head:
                depth = self.customized_head(unet_feature)
                depth = (depth - depth.min()) / (depth.max() - depth.min())
            else:
                depth = self.decode_depth(depth_latent)
                # clip prediction
                depth = torch.clip(depth, -1.0, 1.0)
                # shift to [0, 1]
                depth = (depth * 0.5) + 0.5
            return depth

        elif mode == 'seg':
            if self.customized_head:
                raise NotImplementedError
            seg = self.decode_seg(depth_latent)
            # clip prediction
            # seg = seg.mean(dim=0) # NOTE: average ensemble after single_infer.
            seg = torch.clip(seg, -1.0, 1.0)
            # # shift to [0, 1]
            # seg = (seg + 1.0) / 2.0 
            seg = (seg * 0.5) + 0.5
            # # shift to [0, 255]
            seg = seg * 255

            # import pdb
            # pdb.set_trace()
            # output_type = "pil"
            # image = self.image_processor.postprocess(seg, output_type=output_type)

            return seg
        elif mode == 'normal':
            if self.customized_head:
                raise NotImplementedError
            normal = self.decode_normal(depth_latent)
            normal = torch.clip(normal, -1.0, 1.0)
            return normal
        elif mode == 'feature':
            # import pdb
            # pdb.set_trace()
            z0_features = torch.stack(scheduler_z0_list, dim=1)
            return z0_features # [B, steps, 4, H/8, W/8]
        else:
            raise NotImplementedError


    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.

        Args:
            rgb_in (torch.Tensor):
                Input RGB image to be encoded.

        Returns:
            torch.Tensor: Image latent
        """
        try:
            # encode
            h_temp = self.vae.encoder(rgb_in)
            moments = self.vae.quant_conv(h_temp)
        except:
            # encode
            h_temp = self.vae.encoder(rgb_in.float())
            moments = self.vae.quant_conv(h_temp.float())
            
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # scale latent
        rgb_latent = mean * self.rgb_latent_scale_factor
        return rgb_latent

    
    def decode_depth(self, depth_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode depth latent into depth map.

        Args:
            depth_latent (torch.Tensor):
                Depth latent to be decoded.

        Returns:
            torch.Tensor: Decoded depth map.
        """
        # scale latent
        depth_latent = depth_latent / self.depth_latent_scale_factor
        # decode
        z = self.vae.post_quant_conv(depth_latent)
        stacked = self.vae.decoder(z)

        # mean of output channels
        depth_mean = stacked.mean(dim=1, keepdim=True)
        return depth_mean


    def decode_seg(self, seg_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode depth latent into depth map.

        Args:
            depth_latent (torch.Tensor):
                Depth latent to be decoded.

        Returns:
            torch.Tensor: Decoded depth map.
        """
        # scale latent
        seg_latent = seg_latent / self.seg_latent_scale_factor
        # decode
        z = self.vae.post_quant_conv(seg_latent)
        seg = self.vae.decoder(z)
        seg = seg.clip(-1, 1)

        return seg
    
    def decode_normal(self, normal_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode depth latent into depth map.

        Args:
            depth_latent (torch.Tensor):
                Depth latent to be decoded.

        Returns:
            torch.Tensor: Decoded depth map.
        """
        # scale latent
        normal_latent = normal_latent / self.normal_latent_scale_factor
        # decode
        z = self.vae.post_quant_conv(normal_latent)
        seg = self.vae.decoder(z)

        return seg
