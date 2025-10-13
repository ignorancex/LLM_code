import torch
import torch.nn as nn
import utilities as util
import torch.onnx
from models_spatial import *
from maxunpool_onnx import MaxUnpool3d

import onnx
import onnxruntime as ort
from onnx import numpy_helper

from torchsummaryX import summary

def replace_max_unpool(model):
    def recursive_func(module):
        i = 0
        for name, child in module.named_children():
            if isinstance(child, nn.MaxUnpool3d):
                print("replacing func", name, child, model.cnn_block.d_layer_spatial_dim[i+1])
                # unpool_onnx_layer = MaxUnpool3d(child.kernel_size, stride=child.stride, padding=child.padding) # uncomment if you use custom MaxUnpool3d instead of upsampling, but inference may have an error due to sensitivity in internal module versions 
                unpool_onnx_layer = nn.Upsample(size=tuple(model.cnn_block.d_layer_spatial_dim[i+1]))

                setattr(module, name, unpool_onnx_layer)
                i = i + 1
            else:
                recursive_func(child)
        
        # setattr(module, 'unpooling', MaxUnpool3d) # uncomment if you use custom MaxUnpool3d instead of upsampling

    recursive_func(model)
    model.upsample_layer = "upsample"
    return model

subdetector_name = 'hb'
subdetector_name_upper = subdetector_name.upper()
model_dirpath = rf"C:\Users\mulugetawa\OneDrive - Universitetet i Agder\CERN\InductionProject\CMS_HCAL_ML_OnlineDQM\results"
model_filepath = rf"{model_dirpath}\AE_MODEL.pkl"

# model_dict = util.load_pickle(model_filepath)
# ae_model = model_dict["model"]

# Mulugeta envs: torch-1.12.1, onnx==1.13.0, onnxruntime==1.12.1
# test model init AD-AE model
memorysize = 5
batch_size = 1
t = 1

# temporal model
# # ae_model = CNNRNNAE_MultiDim_SPATIAL(feature_dim=1,
#                                     latent_dim=8,
#                                     target_dim=1,
#                                     spatial_dims=[84, 72, 8],
#                                     memorysize=memorysize,
#                                     e_num_conv_layers=4,
#                                     rnn_model="lstm",
#                                     # upsample_layer="upsample",
#                                     memorysize=memorysize,
#                                     isvariational=False,
#                                     spatial_dims=[84, 72, 8]
#                                     ).to(device)
# input_shape = (batch_size, t, 84, 72, 8, 1)  # Corrected to match typical input dimensions

# # onnx still need to support models with graph using DGL custom ops
# # temporal + graph
# hcal_config_dir = r"C:\Users\mulugetawa\OneDrive - Universitetet i Agder\CERN\InductionProject\CMS_HCAL_ML_OnlineDQM\data\HCAL_CONFIG"
# node_edge_index_filename = r"\he_graph_edge_array_indices_depth.npy"
# node_edge_index = util.load_npdata(rf"{hcal_config_dir}\{node_edge_index_filename}")
# node_edge_index.shape
# ae_model = CNNGNNRNNAE_MultiDim_SPATIAL(feature_dim=1,
#                                     latent_dim=8,
#                                     target_dim=1,
#                                     spatial_dims=[64, 72, 7],
#                                     memorysize=memorysize,
#                                     e_num_conv_layers=4,
#                                     rnn_model="lstm",
#                                     # upsample_layer="upsample",
#                                     isvariational=True,
#                                     node_edge_index=node_edge_index
#                                     ).to(device)
# input_shape = (batch_size, t, 64, 72, 7, 1)  # Corrected to match typical input dimensions

# # GraphSTAD_Optimizer
# # temporal + graph + GraphSTAD_Optimizer: RIN
# model_alg = "CNNGNNRNNAE_MultiDim_SPATIAL"
# ae_model = GraphSTAD_RIN_DC_MultiDim_SPATIAL(model_alg, feature_dim=1,
#                                     latent_dim=8,
#                                     target_dim=1,
#                                     spatial_dims=[64, 72, 7],
#                                     memorysize=memorysize,
#                                     e_num_conv_layers=4,
#                                     rnn_model="lstm",
#                                     # upsample_layer="upsample",
#                                     isvariational=True,
#                                     node_edge_index=node_edge_index,
#                                     use_rnorm_spatial_div=2, # reversible norm using the iphi axis
#                                     ).to(device)
# input_shape = (batch_size, t, 64, 72, 7, 1)  # Corrected to match typical input dimensions

# # temporal + GraphSTAD_Optimizer: RIN, significant accuracy performance improvement
# model_alg = "CNNRNNAE_MultiDim_SPATIAL"
# ae_model = GraphSTAD_RIN_DC_MultiDim_SPATIAL(model_alg, feature_dim=1,
#                                     latent_dim=8,
#                                     target_dim=1,
#                                     spatial_dims=[64, 72, 7],
#                                     memorysize=memorysize,
#                                     e_num_conv_layers=4,
#                                     rnn_model="lstm",
#                                     # upsample_layer="upsample",
#                                     isvariational=True,
#                                     use_rnorm_spatial_div=2, # reversible norm using the iphi axis
#                                     ).to(device)
# input_shape = (batch_size, t, 64, 72, 7, 1)  # Corrected to match typical input dimensions

# temporal + GraphSTAD_Optimizer: DC + RIN, must specify subdetector_name, significant model size reduction
model_alg = "CNNRNNAE_MultiDim_SPATIAL"
subdetector_name = "he"
ae_model = GraphSTAD_RIN_DC_MultiDim_SPATIAL(model_alg, feature_dim=1,
                                    latent_dim=8,
                                    target_dim=1,
                                    spatial_dims=[64, 72, 7],
                                    memorysize=memorysize,
                                    e_num_conv_layers=4,
                                    rnn_model="lstm",
                                    # upsample_layer="upsample",
                                    isvariational=True,
                                    use_rnorm_spatial_div=2, # reversible norm using the iphi axis
                                    use_spatial_split=True, # spatial dimension compression,
                                    subdetector_name=subdetector_name
                                    ).to(device)

input_shape = (batch_size, t, 64, 72, 7, 1)  # Corrected to match typical input dimensions

# Replace max_unpool with custom function
if hasattr(ae_model, "model"):
    ae_model.model = replace_max_unpool(ae_model.model)
else:
    ae_model = replace_max_unpool(ae_model)
    
print(ae_model)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ae_model = ae_model.to(device)  
dummy_input = torch.ones(input_shape).to(device)

# vis model
summary(ae_model, dummy_input)

try:
    ae_model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        output = ae_model(dummy_input)
        if ae_model.isvariational:
            print(output[0].shape)
        else:
            print(output.shape)
    print("Model forward pass successful.")
except Exception as e:
    print("Model forward pass failed:", e)

# Convert the model to ONNX format
onnx_filepath = rf"{model_dirpath}/AE_MODEL.onnx"


ae_model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    torch.onnx.export(
        ae_model,
        (dummy_input,),
        onnx_filepath,
        export_params=True,
        opset_version=15,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'] if not ae_model.isvariational else ['output', 'encode_0', 'encode_1'],
        verbose=True
    )

print(f"Model has been converted to ONNX and saved at {onnx_filepath}")

# Inference Testing
session = ort.InferenceSession(onnx_filepath)
result = session.run(None, {'input': dummy_input.cpu().detach().numpy()})[:1] # the first output is the reconstructed X
for r in result:
    print(r.shape)

print(f"ONNX model inference success!")