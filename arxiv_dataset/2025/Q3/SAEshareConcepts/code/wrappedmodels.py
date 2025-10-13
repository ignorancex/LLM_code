import open_clip
import torch
from abc import abstractmethod, ABC
from torchvision.transforms import v2
from transformers import AutoModel, AutoTokenizer, SiglipModel, SiglipProcessor
from transformers import AutoImageProcessor, AutoModelForImageClassification

class WrappedModel(torch.nn.Module, ABC):
    """Abstract class to record activations from models"""
    def __init__(self, layers,*args,**kwargs):
        super().__init__()
        self.activations = {}
        self.model, self.transform = self.setup_model(**kwargs)
        self.layers = layers
        self.register_layers()

    @abstractmethod
    def setup_model(self, **kwargs):
        """Construct model and transform of a WrappedModel
        Returns:
            - model (nn.Module) : model to analyze
            - transform (Callable) : image transform or tokenizer corresponding to `model`
        """
        pass

    def forward(self, x):
        """Pass data to the model in order to record the corresponding activations"""
        return self.model(x)

    @property
    @abstractmethod
    def layers_to_record(self):
        """The list of layers that may be recorded for a type of models"""
        pass

    @abstractmethod
    def d_vit(self, layer_id):
        """Dimension of the layer at a given index"""
        pass

    def register_layers(self):
        """Registering hooks for selected layers"""
        for layer in self.layers:
            self.layers_to_record[layer].register_forward_hook(
                self.get_activations(layer)
            )

    def get_activations(self, name):
        """Hook function to store activations in self.activations"""
        def hook(model, input, output):
            if isinstance(output, tuple) and len(output) == 1:
                self.activations[name] = output[0].detach()
            elif isinstance(output, tuple):
                self.activations[name] = output[0].detach()
            else:
                self.activations[name] = output.detach()
        return hook
    

class ClipVision(WrappedModel):
    def __init__(self, layers, **kwargs):
        super().__init__(layers, **kwargs)

    def setup_model(self, clip_model_name='ViT-L/14-quickgelu', clip_pretrained='openai',**kwargs):
        if "hf-hub" in clip_model_name: clip_pretrained=None
        clip, transform = open_clip.create_model_from_pretrained(
            clip_model_name, pretrained=clip_pretrained
        )

        return clip.visual, transform

    @property
    def layers_to_record(self):
        return self.model.transformer.resblocks
    
    def d_vit(self, layer_id):
        return self.layers_to_record[layer_id].mlp[0].in_features
    

class SigLIP2Vision(WrappedModel):
    def __init__(self, layers, *args, **kwargs):
        super().__init__(layers, *args, **kwargs)

    def setup_model(self, siglip2_model_name="google/siglip2-large-patch16-384", **kwargs):
        model = SiglipModel.from_pretrained(siglip2_model_name).vision_model
        processor = SiglipProcessor.from_pretrained(siglip2_model_name)
        return model, lambda img: processor(images=img)['pixel_values'].squeeze(0)
    
    @property
    def layers_to_record(self):
        return self.model.encoder.layers

    def d_vit(self, layer_id):
        return self.layers_to_record[layer_id].layer_norm2.normalized_shape[0]

class DinoV2(WrappedModel):
    def __init__(self, layers, **kwargs):
        super().__init__(layers, **kwargs)

    def setup_model(self, dino_model_name='dinov2_vitl14', **kwargs):
        model = torch.hub.load("facebookresearch/dinov2", dino_model_name)
        transform = v2.Compose([
            v2.Resize(size=256),
            v2.CenterCrop(size=(224, 224)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]),
        ])
        return model, transform
    
    @property
    def layers_to_record(self):
        return self.model.blocks
    
    def d_vit(self, layer_id):
        return self.layers_to_record[layer_id].norm1.normalized_shape[0]
    

class ViT(WrappedModel):
    def __init__(self, layers, *args, **kwargs):
        super().__init__(layers, *args, **kwargs)

    def setup_model(self, vit_model_name="google/vit-large-patch16-384", **kwargs):
        processor = AutoImageProcessor.from_pretrained(vit_model_name, use_fast=True)
        model = AutoModelForImageClassification.from_pretrained(vit_model_name)
        return model.vit, lambda img: processor(images=img, return_tensors="pt")['pixel_values'].squeeze(0)
    
    @property
    def layers_to_record(self):
        return self.model.encoder.layer

    def d_vit(self, layer_id):
        return self.layers_to_record[layer_id].output.dense.out_features
    
    

class ClipText(WrappedModel):
    def __init__(self, layers, *args, **kwargs):
        super().__init__(layers, *args, **kwargs)

    def setup_model(self, clip_model_name='ViT-L/14-quickgelu', clip_pretrained='openai',**kwargs):
        if "hf-hub" in clip_model_name: clip_pretrained=None
        clip, _image_transform = open_clip.create_model_from_pretrained(
            clip_model_name, pretrained=clip_pretrained
        )
        try:
            tokenizer = open_clip.get_tokenizer(clip_model_name.replace('/','-'))            
        except: tokenizer = open_clip.get_tokenizer(clip_model_name)
        return clip, tokenizer
    
    @property
    def layers_to_record(self):
        return self.model.transformer.resblocks
    
    def d_vit(self, layer_id):
        return self.layers_to_record[layer_id].mlp[0].in_features
    def forward(self, inputs):
        return self.model.encode_text(inputs)
    
class SiglipText(WrappedModel):
    def __init__(self, layers, *args, **kwargs):
        super().__init__(layers, *args, **kwargs)

    def setup_model(self, siglip2_model_name="google/siglip2-large-patch16-384", **kwargs):
        model = SiglipModel.from_pretrained(siglip2_model_name).text_model
        processor = SiglipProcessor.from_pretrained(siglip2_model_name)
        tokenizer_fn = lambda x: processor(text=x, padding=True, return_tensors='pt', truncation=True, max_length=64).input_ids
        return model, tokenizer_fn
    @property
    def layers_to_record(self):
        return self.model.encoder.layers

    def d_vit(self, layer_id):
        return self.layers_to_record[layer_id].layer_norm2.normalized_shape[0]
    

class Deberta(WrappedModel):
    def __init__(self, layers, *args, **kwargs):
        super().__init__(layers, *args, **kwargs)
    def setup_model(self, deberta_model_name="microsoft/deberta-large", **kwargs):
        model = AutoModel.from_pretrained(deberta_model_name)
        tokenizer = AutoTokenizer.from_pretrained(deberta_model_name)
        tokenizer_fn = lambda x: tokenizer(x, return_tensors='pt', padding=True)
        return model, tokenizer_fn

    @property
    def layers_to_record(self):
        return self.model.encoder.layer
    def d_vit(self, layer_id):
        return self.layers_to_record[layer_id].output.dense.out_features
    def forward(self, inputs):
        return self.model(**inputs)
    

class Bert(WrappedModel):    
    def __init__(self, layers, *args, **kwargs):
        super().__init__(layers, *args, **kwargs)
    def setup_model(self, bert_model_name='bert-large-uncased',**kwargs):
        model = AutoModel.from_pretrained(bert_model_name)
        tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        tokenizer_fn = lambda x: tokenizer(x, return_tensors='pt', padding=True, truncation=True)
        return model, tokenizer_fn

    @property
    def layers_to_record(self):
        return self.model.encoder.layer
    def d_vit(self, layer_id):
        return self.layers_to_record[layer_id].output.dense.out_features
    def forward(self, inputs):
        return self.model(**inputs)