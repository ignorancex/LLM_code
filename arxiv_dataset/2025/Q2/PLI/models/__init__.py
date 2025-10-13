# from utils.mixins import GradientControlDataParallel
import torch
from models.pipeline_stable_diffusion_img2img import StableDiffusionImg2ImgPipeline
from models.clip_image_encoder_pipeline import ClipImageEncoderPipeline
from models.clip_text_encoder_pipeline import ClipTextEncoderPipeline
from models.blip2_pipeline import BLIP2Pipeline
from models.text_inversion_pipeline import TextInversionPipeline
from models.image_encoders import image_encoder_factory
from models.text_encoders import text_encoder_factory
from models.compositors import compositors_factory
from utils.mixins import GradientControlDataParallel
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def create_models(configs: dict):
    if configs['pretrain_dataset'] != 'None': # create models
        image_encoder = image_encoder_factory(configs=configs)
        text_encoder = text_encoder_factory(configs=configs)

        input_dim = image_encoder.layer_shapes()['output_dim']
        # output_dim = image_encoder.layer_shapes()['output_dim']
        compositor = compositors_factory(input_dim=input_dim, configs=configs)
        models = {
            'image_encoder': image_encoder,
            'text_encoder': text_encoder,
            'compositor': compositor
        }

        # TODO: 封装后原方法没了
        if configs['num_gpu'] > 1:
            for name, model in models.items():
                models[name] = GradientControlDataParallel(model.cuda())

        return models
    return create_pipeline(configs)


def create_pipeline(configs):
    # Text Inversion Settings
    if configs['text_inversion_model'] != "None":
        # text inversion pipeline default use clip model
        text_inversion_pipeline = TextInversionPipeline(text_encoder_name=configs['clip_text_model'],
                                                        image_encoder_name=configs['clip_image_model'],
                                                        blip_model_name=configs['text_inversion_model'],
                                                        model_type=configs['text_inversion_model_type'])
        text_inversion_pipeline.to(device)
        return text_inversion_pipeline

    # BLIP-2 Settings
    if configs['blip_model_name'] != "None":
        blip2_pipeline = BLIP2Pipeline(bilp2_model_name=configs['blip_model_name'],
                                       model_type=configs['blip_model_type'])
        blip2_pipeline.to(device)
        return blip2_pipeline

    models = {}
    # Diffusion Settings
    if configs['diffusion_model_id_or_path'] != "None":
        model_id_or_path = configs.get('diffusion_model_id_or_path', 'runwayml/stable-diffusion-v1-5')
        stable_diffusion_img2img_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path,
                                                                                           torch_dtype=torch.float32)
        stable_diffusion_img2img_pipeline.to(device)
        models = {'diffusion': stable_diffusion_img2img_pipeline}

    # return latent will be (bs, 4, *, *), not vae decoded image
    if configs['clip_image_model'] != "None":
        assert configs['diffusion_model_id_or_path'] != "None" and configs[
            "output_type"] != "latent", "clip image model must be used with diffusion model and output type is not latent"
        clip_image_encoder_pipeline = ClipImageEncoderPipeline(clip_model_name=configs['clip_image_model'])
        clip_image_encoder_pipeline.to(device)
        # ensemble clip in the models
        models['clip_image_model'] = clip_image_encoder_pipeline

    if configs['clip_text_model'] != "None":
        clip_text_encoder_pipeline = ClipTextEncoderPipeline(clip_model_name=configs['clip_text_model'])
        clip_text_encoder_pipeline.to(device)
        models['clip_text_model'] = clip_text_encoder_pipeline
    return models

if __name__ == "__main__":
    # from pathlib import Path
    # import yaml
    # print(Path("./v1-5-pruned-emaonly.ckpt").is_file())
    # # 读取 yaml 文件
    # # config_files = yaml.load(open("./v1-inference.yaml", "r", encoding="utf-8"), Loader=yaml.FullLoader)
    # StableDiffusionImg2ImgPipeline.from_pretrained('stable-diffusion-v1-5', torch_dtype=torch.float16)
    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        return timesteps, num_inference_steps - t_start