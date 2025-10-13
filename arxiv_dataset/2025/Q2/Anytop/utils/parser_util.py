from argparse import ArgumentParser
import argparse
import os
import json
import copy
import sys

def parse_and_load_from_model(parser):
    # args according to the loaded model
    # do not try to specify them from cmd line since they will be overwritten
    add_model_options(parser)
    args = parser.parse_args()
    args_to_overwrite = []
    for group_name in ['dataset', 'model', 'diffusion']:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)

    if isinstance(args.model_path, list) and len(args.model_path) == 1:
        args.model_path = args.model_path[0]
    
    # load args from model
    assert not isinstance(args, list) and not isinstance(args.model_path, list), 'Deprecated feature..'
    args = extract_args(copy.deepcopy(args), args_to_overwrite, args.model_path)

    return args

def extract_args(args, args_to_overwrite, model_path):
    args_path = os.path.join(os.path.dirname(model_path), 'args.json')
    assert os.path.exists(args_path), 'Arguments json file was not found!'
    with open(args_path, 'r') as fr:
        model_args = json.load(fr)

    for a in args_to_overwrite:
        if a in model_args.keys():
            setattr(args, a, model_args[a])

    # backward compatibility
    if isinstance(args.emb_trans_dec, bool):
        if args.emb_trans_dec:
            args.emb_trans_dec = 'cls_tcond_cross_tcond'
        else: 
            args.emb_trans_dec = 'cls_none_cross_tcond'
    return args

def get_args_per_group_name(parser, args, group_name):
    for group in parser._action_groups:
        if group.title == group_name:
            group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            return list(argparse.Namespace(**group_dict).__dict__.keys())
    return ValueError('group_name was not found.')

def get_model_path_from_args():
    try:
        dummy_parser = ArgumentParser()
        dummy_parser.add_argument('model_path')
        dummy_args, _ = dummy_parser.parse_known_args()
        return dummy_args.model_path
    except:
        raise ValueError('model_path argument must be specified.')

def add_base_options(parser):
    group = parser.add_argument_group('base')
    group.add_argument("--cuda", default=True, type=bool, help="Use cuda device, otherwise use CPU.")
    group.add_argument("--device", default=0, type=int, help="Device id to use.")
    group.add_argument("--seed", default=10, type=int, help="For fixing random seed.")
    group.add_argument("--batch_size", default=16, type=int, help="Batch size during training.")
    group = parser.add_argument_group('diffusion')
    group.add_argument("--noise_schedule", default='cosine', choices=['linear', 'cosine'], type=str,
                       help="Noise schedule type")
    group.add_argument("--diffusion_steps", default=100, type=int,
                       help="Number of diffusion steps (denoted T in the paper)")
    group.add_argument("--sigma_small", default=True, type=bool, help="Use smaller sigma values.")

def add_model_options(parser):
    group = parser.add_argument_group('model')
    group.add_argument("--arch", default='trans_enc',
                       choices=['trans_enc', 'trans_dec', 'gru'], type=str,
                       help="Architecture types as reported in the paper.")
    group.add_argument("--emb_trans_dec", default=False, type=bool,
                       help="For trans_dec architecture only, if true, will inject condition as a class token"
                            " (in addition to cross-attention).")
    group.add_argument("--layers", default=4, type=int,
                       help="Number of layers.")
    group.add_argument("--latent_dim", default=128, type=int,
                       help="Transformer/GRU width.")
    group.add_argument("--cond_mask_prob", default=.1, type=float,
                       help="The probability of masking the condition during training."
                            " For classifier-free guidance learning.")
    group.add_argument("--lambda_fs", default=0.0, type=float, help="Foot contact loss.")
    group.add_argument("--lambda_geo", default=0.0, type=float, help="Foot contact loss.")
    group.add_argument("--t5_name", default='t5-base', choices=["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b",
              "google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large",
              "google/flan-t5-xl", "google/flan-t5-xxl"], type=str,
                       help="Choose t5 pretrained model")
    group.add_argument("--temporal_window", default=31, type=int,
                       help="temporal window size")
    group.add_argument("--skip_t5", action='store_true',
                       help="If passed, joints names wont be added to features")
    group.add_argument("--value_emb", action='store_true',
                       help="If passed, graph multihead attention learns GRPE value embeddings")

def add_data_options(parser):
    group = parser.add_argument_group('dataset')
    group.add_argument("--data_dir", default="", type=str,
                       help="If empty, will use defaults according to the specified dataset.")
    group.add_argument("--objects_subset", default='all', choices=['all', 'quadropeds' , 'flying', 'bipeds', 'millipeds', 'millipeds_snakes', 'quadropeds_clean', 'millipeds_clean', 'flying_clean', 'bipeds_clean', 'all_clean'], type=str,
                       help="Object subset.")

def add_training_options(parser):
    group = parser.add_argument_group('training')
    group.add_argument("--save_dir", type=str,
                       help="Path to save checkpoints and results.")
    group.add_argument("--model_prefix", type=str,
                       help="Unique string at the beggining of the model name.")
    group.add_argument("--overwrite", action='store_true',
                       help="If True, will enable to use an already existing save_dir.")
    group.add_argument("--ml_platform_type", default='NoPlatform', choices=['NoPlatform', 'ClearmlPlatform', 'TensorboardPlatform', 'WandBPlatform'], type=str,
                       help="Choose platform to log results. NoPlatform means no logging.")
    group.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")

    group.add_argument("--weight_decay", default=0.0, type=float, help="Optimizer weight decay.")
    group.add_argument("--lr_anneal_steps", default=0, type=int, help="Number of learning rate anneal steps.")
    group.add_argument("--eval_batch_size", default=32, type=int,
                       help="Batch size during evaluation loop. Do not change this unless you know what you are doing. "
                            "T2m precision calculation is based on fixed batch size 32.")
    group.add_argument("--eval_split", default='test', choices=['val', 'test'], type=str,
                       help="Which split to evaluate on during training.")
    group.add_argument("--eval_during_training", action='store_true',
                       help="If True, will run evaluation during training.")
    group.add_argument("--eval_rep_times", default=3, type=int,
                       help="Number of repetitions for evaluation loop during training.")
    group.add_argument("--eval_num_samples", default=1_000, type=int,
                       help="If -1, will use all samples in the specified split.")
    group.add_argument("--log_interval", default=50, type=int,
                       help="Log losses each N steps")
    group.add_argument("--save_interval", default=10_000, type=int,
                       help="Save checkpoints and run evaluation each N steps")
    group.add_argument("--num_steps", default=600_000, type=int,
                       help="Training will stop after the specified number of steps.")
    group.add_argument("--num_frames", default=120, type=int,
                       help="Limit for the maximal number of frames. In HumanML3D and KIT this field is ignored.")
    group.add_argument("--resume_checkpoint", default="", type=str,
                       help="If not empty, will start from the specified checkpoint (path to model###.pt file).")
    group.add_argument("--gen_during_training", action='store_true',
                       help="If True, will generate motions during training, on each save interval.")
    group.add_argument("--gen_num_samples", default=3, type=int,
                       help="Number of samples to sample while generating")
    group.add_argument("--gen_num_repetitions", default=2, type=int,
                       help="Number of repetitions, per sample (text prompt/action)")
    group.add_argument("--use_ema", action='store_true',
                       help="If True, will use EMA model averaging.")
    group.add_argument("--balanced", action='store_true',
                       help="Use balancing sampler for fairness between topologies")

def add_sampling_options(parser):
    group = parser.add_argument_group('sampling')
    group.add_argument("--model_path", required=True, type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--output_dir", default='', type=str,
                       help="Path to results dir (auto created by the script). "
                            "If empty, will create dir in parallel to checkpoint.")
    group.add_argument("--num_samples", default=10, type=int,
                       help="Maximal number of prompts to sample, "
                            "if loading dataset from file, this field will be ignored.")
    group.add_argument("--num_repetitions", default=3, type=int,
                       help="Number of repetitions, per sample (text prompt/action)")
    group.add_argument("--cond_path", default='', type=str,
                       help="provide cond.py path in case you wish to generate motion for skeleton not included in Truebones dataset.")
    

def add_generate_options(parser):
    group = parser.add_argument_group('generate')
    group.add_argument("--motion_length", default=6.0, type=float,
                       help="The length of the sampled motion [in seconds]. "
                            "Maximum is 9.8 for HumanML3D (text-to-motion), and 2.0 for HumanAct12 (action-to-motion)")
    group.add_argument("--object_type", default=['Flamingo'], type=str, nargs='+',
                       help="An object type to be generated. If empty, will generate flamingo :).")
    
def add_dift_options(parser):
    # bvhs_dir, sample_bvh, face_joints, save_dir=None, tpos_bvh=None
    group = parser.add_argument_group('dift')
    # group.add_argument("--apply_pca", action='store_true',
    #                    help="apply pca on feats before calculating similarity.")
    group.add_argument("--sample_ref", default='assets/Monkey___B2Attack_574.npy', type=str,
                       help="sample bvh ref.")
    group.add_argument("--sample_tgt", default=['assets/Scorpion___SlowForward_837.npy'], type=str, nargs='+',
                       help="sample bvh tgt.")
    group.add_argument("--tmp_save_dir", default='', type=str,
                       help="temporal save dir.")
    group.add_argument("--suffix", default='', type=str,
                       help="file suffix.")
    group.add_argument("--dift_type", default='spatial', choices=['spatial', 'temporal'], type=str,
                       help="apply dift on spatial or temporal features")
    group.add_argument("--layer", default=0, type=int,
                       help="Layer to extract DIFT features from.")
    group.add_argument("--timestep", default=90, type=int,
                       help="Timestep to extract DIFT features from.")


def add_edit_options(parser):
    group = parser.add_argument_group('edit')
    group.add_argument("--edit_mode", default='in_between', choices=['in_between', 'upper_body'], type=str,
                       help="Defines which parts of the input motion will be edited.\n"
                            "(1) in_between - suffix and prefix motion taken from input motion, "
                            "middle motion is generated.\n"
                            "(2) upper_body - lower body joints taken from input motion, "
                            "upper body is generated.")
    group.add_argument("--prefix_end", default=0.25, type=float,
                       help="For in_between editing - Defines the end of input prefix (ratio from all frames).")
    group.add_argument("--suffix_start", default=0.75, type=float,
                       help="For in_between editing - Defines the start of input suffix (ratio from all frames).")
    group.add_argument("--samples", default=['assets/Ostrich___Attack_581.npy'], type=str, nargs='+',
                    help="samples npy")
    group.add_argument("--object_type", default='Flamingo', type=str,
                    help="An object type to be generated. If empty, will generate flamingo :).")
    group.add_argument("--upper_body_root", default=[0], type=int, nargs='+',
                       help="defines the root joints of the upper body for upper_body editing mode.")
    group.add_argument("--unique_str", default='', type=str, help="A string to be added to the file name to identify a specific change. Should start with '_'.")


def add_render_options(parser):
    group = parser.add_argument_group('render')
    group.add_argument('--bvh_path', type=str, default='assets/Truebones_Chicken', help='path of animation bvh file')
    group.add_argument('--save_dir', type=str, default='save/render_out', help='')
    group.add_argument('--scale', type=float, default=0.7, help='')
    group.add_argument('--cylinder_radius', type=float, default=0.42, help='')
    group.add_argument('--sphere_radius', type=float, default=0.56, help='')
    group.add_argument("--subset", default='bipeds', choices=['quadropeds' , 'flying', 'bipeds', 'millipeds_snakes'], type=str, help="Object subset.")


def add_evaluation_options(parser):
    group = parser.add_argument_group('eval')
    group.add_argument("--eval_mode", default='npy_loc',type=str, choices=['npy_rot', 'npy_loc'], help="Path to gt dir.")
    group.add_argument("--benchmark_path", default='eval/benchmarks/benchmark_all.txt', type=str,  help="Path to benchmark character names. If empty, will use all excluding the characters_to_exclude")
    group.add_argument("--eval_gt_dir", default='dataset/truebones/zoo/truebones_processed/motions', type=str, help="Path to gt dir.")
    group.add_argument("--eval_gen_dir", required=True, type=str, help="Path to gen dir.")
    group.add_argument("--characters_to_exclude", default='MouseyNoFingers,Mousey_m,Trex,SabreToothTiger,Raptor2', type=str, help="Comma separated list of characters to exclude. The default is character with more than 40 motions.")
    group.add_argument("--unique_str", default='', type=str, help="A string to be added to the file name to identify a specific change. Should start with '_'.")

def train_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_training_options(parser)
    return parser.parse_args()

def generate_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_data_options(parser)
    add_sampling_options(parser)
    add_generate_options(parser)
    args = parse_and_load_from_model(parser)
    return args

def process_new_skeleton_args():
    parser = ArgumentParser()
    group = parser.add_argument_group('process_new_skeleton')
    group.add_argument("--object_name", required=True, type=str,
                       help="A character's indicative name")
    group.add_argument("--bvh_dir", required=True, type=str,
                       help="Path to a directory containing BVH files of the skeleton. More files improve statistical accuracy for \
                           motion denormalization.")
    group.add_argument("--save_dir", required=True, type=str,
                       help="Output directory.")
    group.add_argument("--face_joints_names", default=["RLeg1", "LLeg1", "RArm1", "LArm1"], type=str, nargs=4,
                       help="Four joints defining skeleton orientation ([right hip, left hip, right shoulder, left shoulder] or equivalent). \
                           Used to align the skeleton to Z+ and XZ plane.")
    group.add_argument("--tpos_bvh", default='', type=str,
                       help="A BVH file of the character's natural rest pose for meaningful rotation learning. \
                            If missing, the code selects a pose from the provided BVH files.")
    args = parser.parse_args()
    return args

def dift_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_data_options(parser)
    add_sampling_options(parser)
    add_dift_options(parser)
    args = parse_and_load_from_model(parser)

    return args


def edit_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_sampling_options(parser)
    add_edit_options(parser)
    return parse_and_load_from_model(parser)


def evaluation_parser():
    parser = ArgumentParser()
    add_base_options(parser)
    add_evaluation_options(parser)
    return parser.parse_args()

def render_parser():
    parser = ArgumentParser()
    add_render_options(parser)
    if "--" not in sys.argv:
        argv = []
    else:
        argv = sys.argv[sys.argv.index("--") + 1:]
    return parser.parse_args(argv)

