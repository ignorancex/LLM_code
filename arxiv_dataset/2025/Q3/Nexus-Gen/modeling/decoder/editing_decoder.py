import torch
from diffsynth import ModelManager
from diffsynth.models.utils import load_state_dict
from diffsynth.models.flux_dit import FluxDiT
from modeling.decoder.modules import ImageEmbeddingMerger
from transformers import AutoConfig
from .pipelines import NexusGenEditingPipeline


class FluxDiTStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        return state_dict


def state_dict_converter():
    return FluxDiTStateDictConverter()


class NexusGenEditingDecoder:

    def __init__(self, decoder_path, flux_path, qwenvl_path, device='cuda', torch_dtype=torch.bfloat16, enable_cpu_offload=False, fp8_quantization=False):
        self.device = device
        self.torch_dtype = torch_dtype
        self.enable_cpu_offload = enable_cpu_offload
        self.fp8_quantization = fp8_quantization
        self.pipe, self.embedding_merger = self.get_pipe(decoder_path, flux_path, qwenvl_path, device, torch_dtype)

    def get_pipe(self, decoder_path, flux_path, qwenvl_path, device="cuda", torch_dtype=torch.bfloat16):
        if self.enable_cpu_offload:
            model_manager = ModelManager(torch_dtype=torch_dtype, device='cpu')
        else:
            model_manager = ModelManager(torch_dtype=torch_dtype, device=device)
        model_manager.load_models([
            f"{flux_path}/FLUX/FLUX.1-dev/text_encoder/model.safetensors",
            f"{flux_path}/FLUX/FLUX.1-dev/text_encoder_2",
            f"{flux_path}/FLUX/FLUX.1-dev/ae.safetensors",
        ])

        state_dict = load_state_dict(decoder_path)
        dit_state_dict = {key.replace("pipe.dit.", ""): value for key, value in state_dict.items() if key.startswith('pipe.dit.')}
        embedding_merger_state_dict = {key.replace("embedding_merger.", ""): value for key, value in state_dict.items() if key.startswith('embedding_merger.')}

        model_config = AutoConfig.from_pretrained(qwenvl_path)
        embedding_merger = ImageEmbeddingMerger(model_config, num_layers=1, out_channel=4096, device=device)
        embedding_merger.load_state_dict(embedding_merger_state_dict)
        embedding_merger.to(device, dtype=torch_dtype)

        FluxDiT.state_dict_converter = staticmethod(state_dict_converter)
        model_manager.load_model_from_single_file(decoder_path, state_dict=dit_state_dict, model_names=['flux_dit'], model_classes=[FluxDiT], model_resource='diffusers')
        dit_torch_dtype = torch_dtype if not self.fp8_quantization else torch.float8_e4m3fn
        model_manager.model[-1].to(dtype=dit_torch_dtype)

        pipe = NexusGenEditingPipeline.from_model_manager(model_manager, device=device)
        if self.enable_cpu_offload:
            pipe.enable_cpu_offload()
        if self.fp8_quantization:
            pipe.dit.quantize()

        return pipe, embedding_merger

    @torch.no_grad()
    def decode_image_embeds(self,
                            embed,
                            ref_embed=None,
                            embeds_grid=torch.tensor([[1, 18, 18]]),
                            ref_embeds_grid=torch.tensor([[1, 36, 36]]),
                            height=512,
                            width=512,
                            num_inference_steps=50,
                            seed=42,
                            negative_prompt="",
                            cfg_scale=1.0,
                            embedded_guidance=3.5,
                            **pipe_kwargs):
         # long 
        embeds_grid = embeds_grid.to(device=self.device, dtype=torch.long)
        ref_embeds_grid = ref_embeds_grid.to(device=self.device, dtype=torch.long)

        embed = embed.unsqueeze(0) if len(embed.size()) == 2 else embed
        embed = embed.to(device=self.device, dtype=self.torch_dtype)
        ref_embed = ref_embed.unsqueeze(0) if ref_embed is not None and len(ref_embed.size()) == 2 else ref_embed
        ref_embed = ref_embed.to(device=self.device, dtype=self.torch_dtype) if ref_embed is not None else None

        visual_emb = self.embedding_merger(embed, embeds_grid, ref_embed, ref_embeds_grid)
        visual_emb = visual_emb.to(device=self.device, dtype=self.torch_dtype)

        image = self.pipe(prompt="",
                          image_embed=visual_emb,
                          num_inference_steps=num_inference_steps,
                          embedded_guidance=embedded_guidance,
                          negative_prompt=negative_prompt,
                          cfg_scale=cfg_scale,
                          height=height,
                          width=width,
                          seed=seed,
                          **pipe_kwargs)
        return image
