# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion_general_skeleton, load_model
from utils import dist_util
from data_loaders.truebones.truebones_utils.plot_script import plot_general_skeleton_3d_motion
from data_loaders.tensors import truebones_batch_collate
from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np, recover_from_bvh_rot_np
from data_loaders.truebones.data.dataset import create_temporal_mask_for_window
from os.path import join as pjoin
from model.conditioners import T5Conditioner
import BVH
from InverseKinematics import animation_from_positions
from data_loaders.truebones.truebones_utils.get_opt import get_opt

def main(args = None, cond_dict = None):
    if args is None:
        # args is None unless this method is called from another function (e.g. during training)
        args = generate_args()
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
    fps = opt.fps
    n_frames = int(args.motion_length * fps)
    max_joints = opt.max_joints
    dist_util.setup_dist(args.device)
    object_types = args.object_type
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'samples_{}_{}_seed{}'.format(name, niter, args.seed))
    # mkdir outpath
    os.makedirs(out_path, exist_ok=True)
    args.batch_size = len(object_types)  # Sampling a single batch from the testset, with exactly args.num_samples
    # args.num_repetitions = 1

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion_general_skeleton(args)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model(model, state_dict)
    
    print("Loading T5 model")
    t5_conditioner = T5Conditioner(name=args.t5_name, finetune=False, word_dropout=0.0, normalize_text=False, device='cuda')
    model.to(dist_util.dev())
    model.eval()  # disable random masking
    _, model_kwargs = create_condition(object_types, cond_dict, n_frames, args.temporal_window, t5_conditioner=t5_conditioner, max_joints=opt.max_joints, feature_len=opt.feature_len)


    for rep_i in range(args.num_repetitions):
        print(f'### Sampling [repetitions #{rep_i}]')
        sample_fn = diffusion.p_sample_loop

        sample = sample_fn(
            model,
            (args.batch_size, max_joints, model.feature_len, n_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )

        # Recover XYZ *positions* from matrix representation
        bs, max_joints, n_feats, n_frames = sample.shape
        for i, motion in enumerate(sample):
            n_joints = model_kwargs['y']["n_joints"][i].item()
            motion = motion[:n_joints]
            object_type = model_kwargs['y']["object_type"][i]
            parents = model_kwargs['y']["parents"][i]
            mean = cond_dict[object_type]['mean'][None, :]
            std = cond_dict[object_type]['std'][None, :]
            motion = motion.cpu().permute(2, 0, 1).numpy() * std + mean
            offsets = cond_dict[object_type]['offsets']
            global_positions = recover_from_bvh_ric_np(motion)
            #global_positions, out_anim = recover_from_bvh_rot_np(motion, parents, offsets)
            out_anim, _1, _2 = animation_from_positions(positions=global_positions, parents=parents, offsets=offsets, iterations=150)
            name_pref = '%s_rep_%d'%(object_type, rep_i)
            existing_npy_files = [filename for filename in os.listdir(out_path) if filename.startswith(name_pref) and filename.endswith('.npy')]
            existing_mp4_files = [filename for filename in os.listdir(out_path) if filename.startswith(name_pref) and filename.endswith('.mp4')]
            npy_name = name_pref+'_#%d.npy'%(len(existing_npy_files))
            mp4_name = name_pref+'_#%d.mp4'%(len(existing_mp4_files))
            bvh_name = name_pref+'_#%d.bvh'%(len(existing_mp4_files))
            plot_general_skeleton_3d_motion(pjoin(out_path, mp4_name), parents, global_positions, title=name_pref, fps=fps)
            np.save(pjoin(out_path, npy_name), motion)
            if out_anim is not None:
                BVH.save(pjoin(out_path, bvh_name), out_anim, cond_dict[object_type]['joints_names'])
            print("repetition #" + str(rep_i) + " ,created motion: "+ npy_name)

def encode_joints_names(joints_names, t5_conditioner): # joints names should be padded with None to be of max_len 
        names_tokens = t5_conditioner.tokenize(joints_names)
        embs = t5_conditioner(names_tokens)
        return embs
    
def create_condition(object_types, cond_dict, n_frames, temporal_window, t5_conditioner, max_joints, feature_len):
    batches = list()
    for object_type in object_types:
        batch=list()
         # motion, m_length, parents, joints_perm, inv_joints_perm, tpos_first_frame, offsets, self.temporal_mask_template, joints_graph_dist, joints_relations, object_type, joints_names
        parents = cond_dict[object_type]['parents']
        n_joints = len(parents)
        mean = cond_dict[object_type]['mean']
        std = cond_dict[object_type]['std']
        tpos_first_frame = cond_dict[object_type]['tpos_first_frame']
        tpos_first_frame =  (tpos_first_frame - mean) / (std + 1e-6)
        tpos_first_frame = np.nan_to_num(tpos_first_frame)
        joint_relations = cond_dict[object_type]['joint_relations']
        joints_graph_dist = cond_dict[object_type]['joints_graph_dist']
        offsets = cond_dict[object_type]['offsets']
        joints_names_embs = encode_joints_names(cond_dict[object_type]['joints_names'] , t5_conditioner).detach().cpu().numpy()
        batch.append(np.zeros((n_frames, n_joints, feature_len)))
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
        batches.append(batch)
        
    return truebones_batch_collate(batches)


if __name__ == "__main__":
    main()
