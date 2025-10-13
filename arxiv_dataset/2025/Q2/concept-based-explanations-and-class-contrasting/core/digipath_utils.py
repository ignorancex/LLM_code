# Digipath specific code

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.feature_extraction import create_feature_extractor

from tqdm import tqdm

from decoder.core import Decoder_Digipath, Decoder
from core.config import LABELS, SEED, DEVICE
from core.imagenet_utils import load_model





class AttributionWrapper(nn.Module):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def forward(self, x):
        return torch.mean(self.model(x), dim=[2, 3])



class AttributionLimitToTargetClassWrapper(nn.Module):
    def __init__(self, model, target_class, probability_function=lambda x: F.softmax(x, dim=1),
                        cutoff_score=0.5, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.probability_function = probability_function
        self.target_class = target_class
        self.cutoff_score = cutoff_score


    def forward(self, x):
        #print("wrapper forward")
        pred_logits = self.model(x)

        with torch.no_grad():
            pred_probabilities = self.probability_function(pred_logits)[:, self.target_class] - self.cutoff_score
            mask = torch.where(pred_probabilities > 0.0, torch.ones_like(pred_probabilities), torch.zeros_like(pred_probabilities))

        return torch.mean(pred_logits*mask.unsqueeze(dim=1), dim=[2, 3])


def collect_dataset(target_class):
    dataset = PatchesDataset.load_from_file("derma2_non_normal.dt")
    dataset.set_cache("/localdata/xai_derma/derma2_non_normal.dt_cache")

    return filter_dataset(dataset, target_class)
    #return

def filter_dataset(dataset, target_class):
    CLASS_MAP = {x: 0 for x in range(256)}
    CLASS_MAP.update({x["value"]: x["map_to"] for x in LABELS.values()})

    datasets = []
    for label_info in LABELS.values():
        if label_info["map_to"] == target_class:
            ds = dataset.select('class-' + str(label_info['value']), min_value=200*200)
            datasets.append(ds)
    if len(datasets) > 0:
        concat_ds = torch.utils.data.ConcatDataset(datasets)
        if len(concat_ds) > 0:
            return concat_ds
    return None





def load_model_digipath(num_classes, state_dict_ckpt_path, decoder_depth=5, encoder_depth=5, encoder_pretrained=False,
               encoder_name="resnet34", focal_loss=True, load_mean_std=True, padding_mode="zeros"):

    return

def get_model(model_str):
    return


def load_decoder(model_name="version_299"):
    if model_name == "resnet50_robust":
        layer = "layer3.5"

        return_nodes = {
            # node_name: user-specified key for output dict
            layer: 'out',
        }

        classifier = load_model("resnet50_robust")

        model_from = create_feature_extractor(classifier, return_nodes).eval()

        decoder = Decoder.load_from_checkpoint("/home/digipath2/projects/xai/vector_visualization/logs/resnet50_robust/" + "layer3.5" + "/lightning_logs/version_0/checkpoints/epoch=3-step=40040.ckpt",
                                            feature_extractor=model_from.cpu(), normalization=torch.nn.Identity()).eval().to(DEVICE)

        return decoder


def collect_patches_from_prediction(dataloader, model, target_channel,
                                    probability_function=lambda x: F.softmax(x, dim=1), num_to_sample=60, use_argmax_as_filter=False):
    all_images = []

    torch.manual_seed(SEED)
    for data in tqdm(dataloader, ascii=True, desc="Collect Patches"):
        images = data[0].to(DEVICE)

        use_data_point = False
        with torch.no_grad():
            if use_argmax_as_filter:
                pred = model(images)
                if torch.argmax(pred) == target_channel:
                    use_data_point = True
            else:
                pred_logits = model(images)
                pred = F.relu(probability_function(pred_logits)[:, target_channel] - 0.5)

                if torch.sum(pred) > 0.01:
                    use_data_point = True

        if use_data_point:
            all_images.append(images.cpu())

        if len(all_images) > num_to_sample:
            break

        #print(len(all_images))

    if len(all_images) > 0:
        return torch.cat(all_images, dim=0)
    return all_images
