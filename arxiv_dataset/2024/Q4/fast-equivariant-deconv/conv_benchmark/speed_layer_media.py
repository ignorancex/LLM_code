from utils.utils_equivariance import get_sh_matrices, get_raw_sampling_hp
import torch
import numpy as np
from utils.graphconv import ConvPrecomputed
from utils.graphconv import Conv
import pandas as pd
import gc
import argparse


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
parser = argparse.ArgumentParser()
parser.add_argument(
    '--conv_name',
    required=True,
    help='mixed, bekker, spherical, spatial_sh, spatial_vec',
    type=str
)
parser.add_argument(
    '--ablation',
    action='store_true',
    help='Run ablation',
)
parser.add_argument(
    '--use_hemisphere',
    action='store_true',
    help='When ablation is True: use hemisphere',
)
parser.add_argument(
    '--use_dense',
    action='store_true',
    help='When ablation is True: use dense',
)
parser.add_argument(
    '--use_precomputed',
    action='store_true',
    help='When ablation is True: use precomputed',
)
args = parser.parse_args()

conv_name = args.conv_name
ablation = args.ablation
use_hemisphere = args.use_hemisphere if ablation else True
use_dense = args.use_dense if ablation else True
use_precomputed = args.use_precomputed if ablation else True
assert ablation or (use_hemisphere*use_dense*use_precomputed)
isoSpa = True
name_exp = f'{conv_name}{"_ablation" if ablation else ""}{"_hemisphere" if use_hemisphere else ""}{"_dense" if use_dense else ""}{"_precomputed" if use_precomputed else ""}'

batch_size = 1
channel_in = 32
channel_out = 64
kernel_sizeSph = 5
kernel_sizeSpa = 3
if conv_name=='spherical':
    kernel_sizeSpa = 1
sh_degree_bl = 2
symmetric = True

# Grid coordinate
patch_size = 8
xx = np.arange(patch_size) - patch_size/2 + ((patch_size+1)%2)*0.5
yy = np.arange(patch_size) - patch_size/2 + ((patch_size+1)%2)*0.5
zz = np.arange(patch_size) - patch_size/2 + ((patch_size+1)%2)*0.5
vol_coord = np.stack(np.meshgrid(xx, yy, zz, indexing='ij'), axis=0)

N = 100
data = []
with torch.no_grad():
    for sampling_param in [1, 2, 4, 8]:
        print(sampling_param)
        run_mean = []
        memory_track = []
        
        # Random spherical grid
        vec, lap = get_raw_sampling_hp(sampling_param, legacy=False, hemisphere=use_hemisphere)
        n_vec = vec.shape[0]
        S2SH_bl, SH2S_bl = get_sh_matrices(sh_degree_bl, vec, symmetric)

        # Random layer
        if conv_name=='spatial_vec':
            layer = ConvPrecomputed(channel_in*n_vec, channel_out*n_vec, lap, kernel_sizeSph, kernel_sizeSpa, conv_name='spatial', isoSpa=isoSpa)
        elif conv_name=='spatial_sh':
            layer = ConvPrecomputed(channel_in*S2SH_bl.shape[1], channel_out*S2SH_bl.shape[1], lap, kernel_sizeSph, kernel_sizeSpa, conv_name='spatial', isoSpa=isoSpa)
        elif conv_name=='bekkers':
            layer = ConvPrecomputed(channel_in, channel_out, lap, kernel_sizeSph, kernel_sizeSpa, conv_name='bekkers', vec=vec)
        else:
            if ablation:
                layer = Conv(channel_in, channel_out, lap, kernel_sizeSph, kernel_sizeSpa, conv_name=conv_name, isoSpa=isoSpa, dense=use_dense, precompute=use_precomputed, einsum=True, repeat_interleave=True)
            else:
                layer = ConvPrecomputed(channel_in, channel_out, lap, kernel_sizeSph, kernel_sizeSpa, conv_name=conv_name, isoSpa=isoSpa)
        layer = layer.to(device)

        torch.cuda.reset_peak_memory_stats(device=device)
        torch.cuda.reset_max_memory_cached(device=device)
        memory_track.append(torch.cuda.memory_stats(device=device)['reserved_bytes.all.peak'] / 1024 / 1024)
        memory_model = memory_track[-1]
        layer.eval()
        memory_track.append(torch.cuda.memory_stats(device=device)['reserved_bytes.all.peak'] / 1024 / 1024)
        S2SH_bl = S2SH_bl.to(device)
        SH2S_bl = SH2S_bl.to(device)
        memory_track.append(torch.cuda.memory_stats(device=device)['reserved_bytes.all.peak'] / 1024 / 1024)
        memory_with_input = memory_track[-1]
        # Random signal
        x = torch.rand(batch_size, channel_in, n_vec, patch_size, patch_size, patch_size, device=device)
        memory_track.append(torch.cuda.memory_stats(device=device)['reserved_bytes.all.peak'] / 1024 / 1024)

        # Smooth signal
        x_bl_sh = torch.einsum('bfvxyz,vc->bfcxyz', x, S2SH_bl)
        memory_track.append(torch.cuda.memory_stats(device=device)['reserved_bytes.all.peak'] / 1024 / 1024)
        x_bl =  torch.einsum('bfcxyz,cv->bfvxyz', x_bl_sh, SH2S_bl)
        memory_track.append(torch.cuda.memory_stats(device=device)['reserved_bytes.all.peak'] / 1024 / 1024)
        memory_without_input = memory_track[-1]
        memory_track.append(torch.cuda.memory_stats(device=device)['reserved_bytes.all.peak'] / 1024 / 1024)
        result = np.zeros(N)
        runtime = np.zeros(N)
        for i in range(N):
            memory_track.append(torch.cuda.memory_stats(device=device)['reserved_bytes.all.peak'] / 1024 / 1024)
            start1 = torch.cuda.Event(enable_timing=True)
            end1 = torch.cuda.Event(enable_timing=True)
            # CONVOLUTION FIRST
            # Layer
            start1.record()
            if conv_name=='spatial_sh':
                out_layer = layer(x_bl_sh)
            else:
                out_layer = layer(x_bl)
            end1.record()
            torch.cuda.synchronize()
            runtime[i] = start1.elapsed_time(end1)
            if i>1:
                print(f'{(i+1)/N*100:.3f}% - Mean: {np.mean(runtime[50:i+1]):.3f}', end='\r')
        print()
        memory_track.append(torch.cuda.memory_stats(device=device)['reserved_bytes.all.peak'] / 1024 / 1024)
        memory_without_input = memory_track[-1] - memory_without_input
        memory_with_input = memory_track[-1] - memory_with_input
        print(memory_with_input, memory_model)
        print(f'Mean: {np.mean(runtime[50:]):.3}')
        data.append([name_exp, str(sampling_param), np.mean(runtime[50:]), memory_with_input])
        gc.collect()
        torch.cuda.empty_cache()
        del out_layer, layer, x_bl, x_bl_sh, x, S2SH_bl, SH2S_bl, vec, lap
        gc.collect()
        torch.cuda.empty_cache()

df = pd.DataFrame(data, columns=['Convolution', 'Healpix Resolution', 'Runtime', 'Memory'])
df.to_csv(path_or_buf=f'layer_speed_{name_exp}.csv', index=False)