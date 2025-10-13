# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from utils.fixseed import fixseed
import os
from os.path import join as pjoin
import numpy as np
import torch
from utils.parser_util import edit_args
from utils.model_util import create_model_and_diffusion_general_skeleton, load_model
from utils import dist_util
from data_loaders.truebones.truebones_utils.plot_script import plot_general_skeleton_correspondance, save_multiple_samples
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
        args = edit_args()
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
    dist_util.setup_dist(args.device)
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
            "edit_{}_{}_{}_seed{}".format(name, niter, args.edit_mode, args.seed),
                                'samples_{}_{}_seed{}'.format(name, niter, args.seed))
    # mkdir outpath
    os.makedirs(out_path, exist_ok=True)
    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion_general_skeleton(args)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model(model, state_dict)
    
    print("Loading T5 model")
    t5_conditioner = T5Conditioner(name=args.t5_name, finetune=False, word_dropout=0.0, normalize_text=False, device='cuda')
    model.to(dist_util.dev())
    model.eval()  # disable random masking
    motions_list = [np.load(sample) for sample in args.samples]
    motions, model_kwargs = prepare_inpainting_inputs(
        motions_list,
        args.object_type,
        cond_dict[args.object_type],
        args.temporal_window,
        t5_conditioner,
        max_joints=opt.max_joints, 
        feature_len=opt.feature_len
    )
    motions = motions.to(dist_util.dev())
    max_frames = motions.shape[-1]
    # add inpainting mask according to args
    model_kwargs["y"]["inpainted_motion"] = motions
    if args.edit_mode == "in_between":
        model_kwargs["y"]["inpainting_mask"] = torch.ones_like(
            motions, dtype=torch.bool, device=motions.device
        )  # True means use gt motion
        for i, length in enumerate(model_kwargs["y"]["lengths"].cpu().numpy()):
            start_idx, end_idx = (
                int(args.prefix_end * length),
                int(args.suffix_start * length),
            )
            model_kwargs["y"]["inpainting_mask"][i, :, :, start_idx:end_idx] = (
                False  # do inpainting in those frames
            )
    elif args.edit_mode == "upper_body":
        model_kwargs["y"]["inpainting_mask"] = torch.ones_like(
            motions, dtype=torch.bool, device=motions.device
        )  # True means use gt motion
        prev_len=0
        upper_body_joints = set(args.upper_body_root)
        while prev_len!=len(upper_body_joints):
            prev_len = len(upper_body_joints)
            for joint, parent in enumerate(cond_dict[args.object_type]["parents"]):
                if int(parent) in upper_body_joints:
                    upper_body_joints.add(joint)
        upper_body_joints = list(upper_body_joints)
        print(f"Upper body joints:{upper_body_joints}")
        model_kwargs["y"]["inpainting_mask"][:, upper_body_joints, :, :] = False
        

    for rep_i in range(args.num_repetitions):
        print(f'### Start sampling [repetitions #{rep_i}]')
        sample_fn = diffusion.p_sample_loop
        name_pref = f"rep_{rep_i}_object_type_{args.object_type}_edit_mode_{args.edit_mode}_seed_{args.seed}{args.unique_str}"

        sample = sample_fn(
            model,
            motions.shape,
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
        animations = np.empty(shape=(1 + bs // 5, bs % 5), dtype=object)
        for i, motion in enumerate(sample):
            n_joints = model_kwargs["y"]["n_joints"][i].item()
            length = model_kwargs["y"]["lengths"][i].item()
            motion = motion[:n_joints, :, :length]
            object_type = model_kwargs["y"]["object_type"][i]
            parents = model_kwargs["y"]["parents"][i]
            mean = cond_dict[object_type]["mean"][None, :]
            std = cond_dict[object_type]["std"][None, :]
            inpaint_mask = model_kwargs["y"]["inpainting_mask"][i].any(dim=1)	
            joint2color = {
                frame: {
                    joint: 0 if inpaint_mask[joint, frame] else 1
                    for joint in range(n_joints)
                }
                for frame in range(n_frames)
            }
            motion = motion.cpu().permute(2, 0, 1).numpy() * std + mean
            offsets = cond_dict[object_type]["offsets"]
            global_positions = recover_from_bvh_ric_np(motion)
            #global_positions, out_anim = recover_from_bvh_rot_np(motion, parents, offsets)
            out_anim, _1, _2 = animation_from_positions(positions=global_positions, parents=parents, offsets=offsets, iterations=150)
            name_pref_i = name_pref + f"_{i}"
            npy_name = name_pref_i + ".npy"
            bvh_name = name_pref_i + ".bvh"
            animations[i // 5, i % 5] = plot_general_skeleton_correspondance(
                parents,
                joint2color,
                2,
                global_positions,
                dataset="truebones",
                title=name_pref_i,
                fps=fps,
            )
            data = {
                'motion':motion,
                'joint2color':joint2color,
                'sample_path':args.samples[i]
            }
            np.save(pjoin(out_path, npy_name), data, allow_pickle=True)
            if out_anim is not None:
                BVH.save(pjoin(out_path, bvh_name), out_anim, cond_dict[object_type]['joints_names'])
            print("repetition #" + str(rep_i) + " ,created motion: "+ npy_name)
        
        mp4_name = name_pref + ".mp4"
        save_multiple_samples(out_path, mp4_name, animations, 20, model_kwargs["y"]["lengths"][i].max())

def encode_joints_names(joints_names, t5_conditioner): # joints names should be padded with None to be of max_len 
        names_tokens = t5_conditioner.tokenize(joints_names)
        embs = t5_conditioner(names_tokens)
        return embs
    
def prepare_inpainting_inputs(motions, object_type, cond_dict, temporal_window, t5_conditioner, max_joints, feature_len):
    batches = list()
    for motion in motions:
        n_frames = motion.shape[0]
        batch = list()
        parents = cond_dict['parents']
        n_joints = len(parents)
        mean = cond_dict['mean']
        std = cond_dict['std']
        tpos_first_frame = cond_dict['tpos_first_frame']
        tpos_first_frame =  (tpos_first_frame - mean) / (std + 1e-6)
        tpos_first_frame = np.nan_to_num(tpos_first_frame)
        joint_relations = cond_dict['joint_relations']
        joints_graph_dist = cond_dict['joints_graph_dist']
        offsets = cond_dict['offsets']
        joints_names_embs = encode_joints_names(cond_dict['joints_names'] , t5_conditioner).detach().cpu().numpy()
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
