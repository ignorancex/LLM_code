# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import dift_args
from utils.model_util import create_model_and_diffusion_general_skeleton, load_model
from utils import dist_util
from data_loaders.get_data import get_dataset_loader
from data_loaders.truebones.truebones_utils.plot_script import plot_general_skeleton_correspondance, save_multiple_samples
from data_loaders.tensors import truebones_batch_collate
from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np
from data_loaders.truebones.data.dataset import create_temporal_mask_for_window
from os.path import join as pjoin
from model.conditioners import T5Conditioner
import random
import multiprocessing
from data_loaders.truebones.truebones_utils.get_opt import get_opt


def encode_joints_names(joints_names, t5_conditioner): # joints names should be padded with None to be of max_len 
        names_tokens = t5_conditioner.tokenize(joints_names)
        embs = t5_conditioner(names_tokens)
        return embs

def create_sample_in_batch(motion, object_type, cond_dict_for_object, temporal_window, t5_conditioner, max_joints):
    batch=list()
    parents = cond_dict_for_object['parents']
    n_joints = len(parents)
    n_frames = motion.shape[0]
    mean = cond_dict_for_object['mean']
    std = cond_dict_for_object['std']
    motion = (motion - mean[None]) / (std[None] + 1e-6)
    motion = np.nan_to_num(motion)
    tpos_first_frame = cond_dict_for_object['tpos_first_frame']
    tpos_first_frame =  (tpos_first_frame - mean) / (std + 1e-6)
    tpos_first_frame = np.nan_to_num(tpos_first_frame)
    joint_relations = cond_dict_for_object['joint_relations']
    joints_graph_dist = cond_dict_for_object['joints_graph_dist']
    offsets = cond_dict_for_object['offsets']
    joints_names_embs = encode_joints_names(cond_dict_for_object['joints_names'] , t5_conditioner).detach().cpu().numpy()
    batch.append(motion)
    batch.append(n_frames)
    batch.append(parents)
    batch.append(tpos_first_frame)
    batch.append(offsets)
    batch.append(create_temporal_mask_for_window(temporal_window, n_frames))
    batch.append(joints_graph_dist)
    batch.append(joint_relations)
    batch.append(object_type)
    batch.append(joints_names_embs)
    batch.append(0)
    batch.append(mean)
    batch.append(std)
    batch.append(max_joints)
    return batch

def create_batch_from_motion_paths(motion_paths, cond_dict, temporal_window, t5_conditioner, max_joints):
    batches = list()
    motions = list()
    cond_dicts = list()
    for motion_path in motion_paths:
        object_type = os.path.basename(motion_path).split('_')[0]
        motion, cond_dict_object_type = process_object_type(motion_path=motion_path, object_type=object_type, cond=cond_dict)
        batches.append(create_sample_in_batch(motion, object_type, cond_dict_object_type, temporal_window, t5_conditioner, max_joints))
        motions.append(motion)
        cond_dicts.append(cond_dict_object_type)
    return *truebones_batch_collate(batches), motions, cond_dicts

"""
Returns motion features of the given BVH file. besides bvh_path and face_joints, all other parameters 
are optional, but if supplied might improve DIFT results. 

bvh_path: path to a bvh file of a given animaiton 
face_joints: based on these joints the orientation is determined. 
             Should be given in the order [right hip, left hip, right soulder, left shoulder] 
object_type: relevant only if bvh is from an already seen topology. 
bvh_tpos: bvh of rest pose. relevant mostly if the given bvh tpos is unnatural 
feet: feet indices. The order does not matter 
"""
def process_object_type(object_type, motion_path, cond):
    object_cond_dict = cond[object_type]
    motion_sample = motion_path
    if os.path.isdir(motion_path):
        all_motions = [pjoin(motion_path, f) for f in os.listdir(motion_path) if f.endswith('.bvh') and f.startswith(f'{object_type}__')]
        motion_sample = np.load(random.choice(all_motions))
    motion = np.load(motion_sample)
    return motion, object_cond_dict

def vis_dift(t, layer, activations, cond_ref, cond_tgt, motion_ref, motion_tgt, model_path, dift_type='spatial'):
    # color per kinematic chain, not vertex 
    # plot_single_frame_kinchains
    # [ref, tgt]
    buckets = [0, 14, 22, 35, 51, 70, 84, 95, 129]
    values = [0, 1, 2, 3, 2, 3, 1, 0]
    ref_len = motion_ref.shape[0]
    ref_n_joints = motion_ref.shape[1]
    tgt_len = motion_tgt.shape[0]
    ref_kinchains = cond_ref["kinematic_chains"]
    n_kinchains = len(ref_kinchains)
    tgt_n_joints = motion_tgt.shape[1]
    if dift_type == 'spatial':
        min_length = min(ref_len, tgt_len)
        ref_activations = activations[:min_length, 0, :ref_n_joints]
        tgt_activations = activations[:min_length, 1, :tgt_n_joints]
        
    if dift_type =='temporal':
        ref_activations = activations[:ref_len, 0, :ref_n_joints]
        tgt_activations = activations[:tgt_len, 1, :tgt_n_joints]
        ref_activations = ref_activations.transpose(0, 1)
        tgt_activations = tgt_activations.transpose(0, 1)
        
    ref_activations = ref_activations.mean(0)
    tgt_activations = tgt_activations.mean(0)         
    cos_sim=tgt_activations @ ref_activations.transpose(-1, -2)
    vec_norms = (torch.norm(tgt_activations, dim=-1)[..., None] @ torch.norm(ref_activations, dim=-1)[..., None].transpose(-1, -2))
    cos_sim = cos_sim/vec_norms
    correspondance = torch.argmax(cos_sim, dim=-1)
    if dift_type == 'temporal':
        joint2color_ref = dict()
        for frame in range(ref_len):
            for b in range(len(buckets)-1):
                if frame < buckets[b+1]:
                    break
            frames_cls = values[b]
            joint2color_ref[frame] = dict()
            for j in range(ref_n_joints):
                joint2color_ref[frame][j] = frames_cls

        joint2color_tgt = dict()
        for frame in range(tgt_len):
            joint2color_tgt[frame] = dict()
            for j in range(tgt_n_joints):
                joint2color_tgt[frame][j] = joint2color_ref[correspondance[frame].item()][0]
   
    else:
        # Tailored for Monkey as reference
        joint2color_ref = dict()
        for frame in range(min_length):
            joint2color_ref[frame] = dict()
            for chain_ind, chain in enumerate(ref_kinchains):
                for j in chain:
                    if len(chain) <= 5 and chain[0] != 0:
                        joint2color_ref[frame][j] = joint2color_ref[frame][cond_ref["parents"][j]]
                    else:
                        joint2color_ref[frame][j] = chain_ind
                    if j == 34:
                        joint2color_ref[frame][j] = 0
                    if j == 8:
                        joint2color_ref[frame][j] = 21
                    

        joint2color_tgt = dict()
        for frame in range(min_length):
            joint2color_tgt[frame] = dict()
            for j in range(tgt_n_joints):
                joint2color_tgt[frame][j] = joint2color_ref[frame][correspondance[j].item()]
    

    if dift_type=='temporal':
        positions_ref = recover_from_bvh_ric_np(motion_ref[:ref_len])
        positions_tgt  = recover_from_bvh_ric_np(motion_tgt[:tgt_len])
        animations = np.empty(shape=(1, 2), dtype=object)
        animations[0, 0] = plot_general_skeleton_correspondance(cond_ref["parents"], joint2color_ref, len(set(values)), positions_ref, "", "truebones", fps=20)
        animations[0, 1] = plot_general_skeleton_correspondance(cond_tgt["parents"], joint2color_tgt, len(set(values)), positions_tgt, "", "truebones", fps=20) 
    else:
        positions_ref = recover_from_bvh_ric_np(motion_ref[:min_length])
        positions_tgt = recover_from_bvh_ric_np(motion_tgt[:min_length])
        animations = np.empty(shape=(1, 2), dtype=object)
        animations[0, 0] = plot_general_skeleton_correspondance(cond_ref["parents"], joint2color_ref, n_kinchains, positions_ref, "", "truebones", fps=20)
        animations[0, 1] = plot_general_skeleton_correspondance(cond_tgt["parents"], joint2color_tgt, n_kinchains, positions_tgt, "", "truebones", fps=20)
    if dift_type=='temporal':
        fname = f"{cond_tgt['object_type']}_diffusion_step_{t}_layer_{layer}_ref_{cond_ref['object_type']}_tgt_{cond_tgt['object_type']}_temporal.mp4"
        npy_fname = f"{cond_tgt['object_type']}_diffusion_step_{t}_layer_{layer}_ref_{cond_ref['object_type']}_tgt_{cond_tgt['object_type']}_temporal.npy"
    else:
        fname = f"diffusion_step_{t}_layer_{layer}_ref_{cond_ref['object_type']}_tgt_{cond_tgt['object_type']}_spatial.mp4"
        npy_fname = f"diffusion_step_{t}_layer_{layer}_ref_{cond_ref['object_type']}_tgt_{cond_tgt['object_type']}_spatial.npy"
    model_dir = os.path.dirname(model_path)
    model_number = os.path.basename(model_path)[:-3]
    
    out_dir = pjoin(model_dir, model_number, "dift_out")
    os.makedirs(out_dir, exist_ok=True)
    mapping_dict = {"ref": joint2color_ref, "tgt": joint2color_tgt}
    np.save(pjoin(out_dir, npy_fname), mapping_dict, allow_pickle=True)
    if dift_type == 'spatial':
        save_multiple_samples(out_dir, fname, animations, 20, min_length) 
    else:
        save_multiple_samples(out_dir, fname, animations, 20, max(tgt_len, ref_len))
        

def run_dift(args = None, cond_dict = None):
    if args is None:
        # args is None unless this method is called from another function (e.g. during training)
        args = dift_args()
    fixseed(args.seed)    
    opt = get_opt(args.device)
    if cond_dict is None:
        if args.cond_path:
            cond_dict=np.load(args.cond_path, allow_pickle=True).item()
        else:
            cond_dict = np.load(opt.cond_file, allow_pickle=True).item()
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))        
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    dist_util.setup_dist(args.device)

    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'samples_{}_{}_seed{}'.format(name, niter, args.seed))
    # mkdir outpath
    os.makedirs(out_path, exist_ok=True)
    # this block must be called BEFORE the dataset is loaded
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    args.batch_size = 2  # Sampling a single batch from the testset, with exactly args.num_samples
    args.num_repetitions = 1
    t=args.timestep
    layer=args.layer
    # total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion_general_skeleton(args)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model(model, state_dict)
    
    print("Loading T5 model")
    t5_conditioner = T5Conditioner(name=args.t5_name, finetune=False, word_dropout=0.0, normalize_text=False, device='cuda')
    model.to(dist_util.dev())
    model.eval()  # disable random masking
    batch, model_kwargs, motions, cond_dicts =  create_batch_from_motion_paths([args.sample_ref] + args.sample_tgt, cond_dict, args.temporal_window, t5_conditioner, max_joints=opt.max_joints)
    sample_fn = diffusion.p_sample_single_timestep
    for rep_i in range(args.num_repetitions):
        print(f'### Sampling [repetitions #{rep_i}]')
        sample, activations = sample_fn(
            model, 
            batch.shape,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            init_image=batch.to(device=dist_util.dev()),
            noise=None,
            const_noise=False,
            get_activations={"layer": layer, "timestep": t},
        )
        with multiprocessing.get_context("spawn").Pool(4) as pool:
            pool.map(vis_dift_args, ((t, layer, activations[t][layer][:,[0,i],:], cond_dicts[0], cond_dicts[i], motions[0], motions[i], args.model_path, args.dift_type) for i in range(1, len(motions))))
    
def vis_dift_args(args):
    print("starting vis_dift")
    vis_dift(*args)
    print("finished vis_dift")


if __name__ == "__main__":
    run_dift()