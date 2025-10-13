from torch.utils.data import DataLoader, TensorDataset
import os
from tqdm import tqdm
import numpy as np
from pathlib import Path

from core.core import extract_attribution_filtered_activation_vectors
from core.digipath_utils import collect_dataset, get_model, collect_patches_from_prediction, AttributionLimitToTargetClassWrapper
from core.config import DEVICE, OUTPUT_PATH



for class_idx in tqdm(range(46), ascii=True):
    if class_idx > 0:
        dataset = collect_dataset(target_class=class_idx)
        if dataset is not None:
            dataloader = DataLoader(dataset, shuffle=True, batch_size=1, num_workers=1)

            model = get_model("version_299").eval().to(DEVICE)
            layer = getattr(model.model.encoder.down_blocks, 'down block 3').resnet_blocks[2]

            wrapped_model = AttributionLimitToTargetClassWrapper(model, target_class=class_idx)

            patches = collect_patches_from_prediction(dataloader, model, target_channel=class_idx, num_to_sample=60)
            if len(patches) > 0:
                dataset = TensorDataset(patches)

                save_folder_path = os.path.join(OUTPUT_PATH, "activations_for_gmm_training", "version_299")

                above_cutoff_activations = extract_attribution_filtered_activation_vectors(dataset, target_channel=class_idx,
                                                model=model, wrapped_model=wrapped_model, layer=layer,
                                                save_folder_path=save_folder_path,
                                                batch_size_attribution=1, batch_size_activation=16,
                                                attribution_cutoff=0.25)

                save_name = os.path.join(save_folder_path, str(class_idx) + ".npy")
                Path(save_folder_path).mkdir(parents=True, exist_ok=True)
                np.save(save_name, above_cutoff_activations.cpu().numpy())



