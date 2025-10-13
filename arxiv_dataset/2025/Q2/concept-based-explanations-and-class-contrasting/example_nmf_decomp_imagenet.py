from torch.utils.data import DataLoader, TensorDataset

import torch

from tqdm import tqdm

from core.core import nmf_attribution_whole_ds_decomp, collect_patches
from core.imagenet_utils import get_dataset_for_class, load_model
from core.config import DEVICE, OUTPUT_PATH
import os
import random

#OUTPUT_PATH = os.path.join(OUTPUT_PATH, "grid_search")


def run_nmf_decomp(model, layer, save_name, model_name="resnet50_robust", use_train_ds=False, num_samples=60,
                    use_noise_tunnel=True, n_components=6, class_frequency=1, attribution_method="deep_lift",
                    random_seed=False):
    for class_idx in tqdm(range(1000), ascii=True):
        #print(class_idx)
        if class_idx % class_frequency != 0:
            continue
        dataset = get_dataset_for_class(model_name=model_name, class_idx=class_idx, use_train_ds=use_train_ds)
        if dataset is not None:
            if len(dataset) > 0:
                dataloader = DataLoader(dataset, shuffle=True, batch_size=1, num_workers=0)

                if random_seed:
                    torch.manual_seed(random.randint(0, 999999999))
                else:
                    torch.manual_seed(42)
                patches = collect_patches(dataloader)
                if len(patches) > num_samples:
                    patches = patches[:num_samples]
                if len(patches) > 0:
                    dataset = TensorDataset(patches)


                    save_folder_path = OUTPUT_PATH + save_name

                    nmf_attribution_whole_ds_decomp(dataset, target_channel=class_idx, model=model, layer=layer,
                                                    save_folder_path=save_folder_path, n_components=n_components,
                                                    save_images=False, batch_size=1, use_noise_tunnel=use_noise_tunnel,
                                                    attribution_method=attribution_method)




#---------------------------------------------------------------------------------------------------------


model_name = "resnet50"
model = load_model(model_name).eval().to(DEVICE)
layer = model.layer4[2]
layer_name = "_layer4[2]"
run_nmf_decomp(model, layer, model_name=model_name, save_name=model_name+layer_name+"_4_comps", use_train_ds=True,
                use_noise_tunnel=False, num_samples=500, n_components=4, class_frequency=1)


model_name = "resnet50"
model = load_model(model_name).eval().to(DEVICE)
layer = model.layer4[2]
layer_name = "_layer4[2]"
run_nmf_decomp(model, layer, model_name=model_name, save_name=model_name+layer_name+"_8_comps", use_train_ds=True,
                use_noise_tunnel=False, num_samples=500, n_components=8, class_frequency=1)





model_name = "resnet34"
model = load_model(model_name).eval().to(DEVICE)
layer = model.layer4[2]
layer_name = "_layer4[2]"
run_nmf_decomp(model, layer, model_name=model_name, save_name=model_name+layer_name+"_4_comps", use_train_ds=True,
                use_noise_tunnel=False, num_samples=500, n_components=4, class_frequency=10)



model_name = "resnet50_robust"
model = load_model(model_name).eval().to(DEVICE)
layer = model.layer4[2]
layer_name = "_layer4[2]"
run_nmf_decomp(model, layer, model_name=model_name, save_name=model_name+layer_name+"_4_comps", use_train_ds=True,
                use_noise_tunnel=False, num_samples=500, n_components=4, class_frequency=10)


#---------------------------------------------------------------------------------------------------------



number_of_samples = [50] + [(i+1)*100 for i in range(9)]

for num_samples in number_of_samples:
    num_comp = 4
    model_name = "resnet50"
    model = load_model(model_name).eval().to(DEVICE)
    layer = model.layer4[2]
    layer_name = "_layer4[2]"
    run_nmf_decomp(model, layer, model_name=model_name, save_name=model_name+layer_name+"_4_comps_"+str(num_samples)+"_samples", use_train_ds=True,
                    use_noise_tunnel=False, num_samples=num_samples, n_components=num_comp, class_frequency=10)



#---------------------------------------------------------------------------------------------------------




num_components = [i+1 for i in range(10)]

for num_comp in num_components:
    model_name = "resnet50"
    model = load_model(model_name).eval().to(DEVICE)
    layer = model.layer4[2]
    layer_name = "_layer4[2]"
    run_nmf_decomp(model, layer, model_name=model_name, save_name=model_name+layer_name+"_"+str(num_comp)+"_comps_grid_search", use_train_ds=True,
                    use_noise_tunnel=False, num_samples=500, n_components=num_comp, class_frequency=10)


for num_comp in num_components:
    model_name = "resnet50"
    model = load_model(model_name).eval().to(DEVICE)
    layer = model.layer3[5]
    layer_name = "_layer3[5]"
    run_nmf_decomp(model, layer, model_name=model_name, save_name=model_name+layer_name+"_"+str(num_comp)+"_comps_grid_search", use_train_ds=True,
                    use_noise_tunnel=False, num_samples=500, n_components=num_comp, class_frequency=10)


for num_comp in num_components:
    model_name = "resnet50"
    model = load_model(model_name).eval().to(DEVICE)
    layer = model.layer2[3]
    layer_name = "_layer2[3]"
    run_nmf_decomp(model, layer, model_name=model_name, save_name=model_name+layer_name+"_"+str(num_comp)+"_comps_grid_search", use_train_ds=True,
                    use_noise_tunnel=False, num_samples=500, n_components=num_comp, class_frequency=10)


for num_comp in num_components:
    model_name = "resnet50"
    model = load_model(model_name).eval().to(DEVICE)
    layer = model.layer1[2]
    layer_name = "_layer1[2]"
    run_nmf_decomp(model, layer, model_name=model_name, save_name=model_name+layer_name+"_"+str(num_comp)+"_comps_grid_search", use_train_ds=True,
                    use_noise_tunnel=False, num_samples=500, n_components=num_comp, class_frequency=10)

#---------------------------------------------------------------------------------------------------------



# Gradient times Activation
model_name = "resnet50"
model = load_model(model_name).eval().to(DEVICE)
layer = model.layer4[2]
layer_name = "_layer4[2]"
run_nmf_decomp(model, layer, model_name=model_name, save_name=model_name+layer_name+"_4_comps_gradientXactivation", use_train_ds=True,
                use_noise_tunnel=False, num_samples=50, n_components=4, class_frequency=10, attribution_method="gradient")




# DeepLift plus SmoothGrad
model_name = "resnet50"
model = load_model(model_name).eval().to(DEVICE)
layer = model.layer4[2]
layer_name = "_layer4[2]"
run_nmf_decomp(model, layer, model_name=model_name, save_name=model_name+layer_name+"_4_comps_deeplift_smoothgrad", use_train_ds=True,
                use_noise_tunnel=True, num_samples=50, n_components=4, class_frequency=10, attribution_method="deep_lift")




# Gradient times Activation for layer3.5
model_name = "resnet50"
model = load_model(model_name).eval().to(DEVICE)
layer = model.layer3[5]
layer_name = "_layer3[5]"
run_nmf_decomp(model, layer, model_name=model_name, save_name=model_name+layer_name+"_4_comps_gradientXactivation", use_train_ds=True,
                use_noise_tunnel=False, num_samples=500, n_components=4, class_frequency=10, attribution_method="gradient")




