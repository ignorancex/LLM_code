import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from featup.core.dinov2 import DINOv2UpFeatBackbone
from prediction_heads import BasicPredictionHead, ReconstructionBasicPredictionHead


class DINOv2FeatUp(nn.Module):
    def __init__(self, core_size, output_size, reconstruction=True):
        """ DINOv2+FeatUp predictor
        
        :param core_size: core inference resolution (H,W) of DINOv2
        :param output_size: output size (H,W) of final output (to enable resizing to different resolution than input)
        :param reconstruction: if True, the network has a reconstruction head and output
        """
        super().__init__()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.core_size = core_size
        self.output_size = output_size
        self.reconstruction = reconstruction

        self.embedder = DINOv2UpFeatBackbone.load_backbone(pretrained=True, use_norm=True)
        self.predictor = self.get_new_prediction_head()

    def get_model_name(self):
        """ Returns the model name """
        if self.reconstruction:
            return "dinov2featup_recons"
        else:
            return "dinov2featup"

    def get_new_prediction_head(self):
        """ Returns a new prediction head """
        if self.reconstruction:
            return ReconstructionBasicPredictionHead(384)
        else:
            return BasicPredictionHead(384)
    
    def forward(self, x, should_normalize=True, resize=True):
        """ Perform inference
        
        :param x: image tensor of shape (3,H,W)
        :param should_normalize: if True, then the image will be normalized before inference
        :param resize: if True, image is resized automatically if not the right size
        """
        # This network assumes all data is non-batched
        assert(len(x.shape) == 3)

        # Resize if specified
        if x.shape[1] != self.core_size[0] or x.shape[2] != self.core_size[1]:
            if resize:
                x = F.interpolate(x.unsqueeze(0), size=self.core_size, mode='bilinear')[0]
            else:
                raise ValueError("size is not correct and resize=False")
        
        # Normalize
        if should_normalize:
            x = self.normalize(x)

        # Inference
        embed = self.embedder(x.unsqueeze(0))
        features = F.interpolate(embed['features'], size=self.core_size, mode='bilinear')
        out = self.predictor(features)
        out = {k: F.interpolate(v, size=self.output_size, mode='bilinear')[0,0] for k, v in out.items()} # All output maps are (H,W)
        features = F.interpolate(features, size=self.output_size, mode='bilinear')
        
        return {**out, **{
            'features': features[0], # (384,H,W)
            'cls_token': embed['cls_token'][0] # (384)
        }}
    
    def update_model(self, state_dict):
        """ Update model state dict """
        self.predictor.load_state_dict(state_dict)
    
    def load_model(self, filename):
        """ Load model from file """
        self.predictor.load_state_dict(torch.load(filename))

    def save_model(self, filename):
        """ Save model to file """
        torch.save(self.predictor.state_dict(), filename)
