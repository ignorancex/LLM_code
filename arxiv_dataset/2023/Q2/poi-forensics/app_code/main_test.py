import os
import logging
import sys
from torch.cuda import is_available
import argparse
import yaml
import numpy as np
from config import create_opt
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def create_model(logger, dir_poi, opt, device):
    poi_data = {"": dir_poi}
    typ = opt['model']['type']
    if typ == 'poi_forensics':
        from grip_unina.poi_forensics import PoiForensics
        return PoiForensics(logger, poi_data, opt, device)
    elif typ == 'id_reveal':
        from grip_unina.id_reveal import IdReveal
        return IdReveal(logger, poi_data, opt, device)
    elif typ == 'face_recognition':
        from grip_unina.face_recognition import FeceRec
        return FeceRec(logger, poi_data, opt, device)
    else:
        assert False


#def merge_track(dists, opt):
#    if opt['final_mean'] > 1:
#        if len(dists) < opt['final_mean']:
#            return np.nan
#        half = opt['final_mean']//2
#        from scipy.ndimage import uniform_filter
#        dists = uniform_filter(dists, opt['final_mean'])[half:-half]
#    return np.percentile(dists, opt['percentile'])


def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                        description='testing script.')
    parser.add_argument('--file_video_input', type=str,
                        help='input video file (with extension: .mp4, .avi).')
    parser.add_argument('--dir_poi', type=str,
                        help='features directory of POI.')
    parser.add_argument('--modality', type=str, default='auto',
                        help="the modality to use. It can be 'auto', 'audiovideo', 'onlyvideo' or 'onlyaudio'.")
    parser.add_argument('--dist_normalization', type=int, default=None,
                        help="if True, the outout distances are normalized using on the values obtained on pristine videos.")
    parser.add_argument('--skip_if_exists', type=int, default=0,
                        help='if True, the analysis will not made if the npz file already exists.')
    parser.add_argument('--file_output', type=str,
                        help='output data numpy file (with extension .npz).')
    parser.add_argument('--create_plot', type=int, default=1,
                        help='if True, the plot will be created and saved in the same location of the numpy file.')
    parser.add_argument('--create_videoout', type=int, default=1,
                        help='if True, the output video will be created and saved in the same location of the numpy file.')
    parser.add_argument('--resources_path', type=str, default="./resources/",
                        help='directory with networks weights.')
    parser.add_argument('--gpu', type=int, default=0,
                        help='index of GPU to use (set to -1 for not using the GPU).')
    parser.add_argument('--stride', type=int, default=32,
                        help='number of frames analyzed in parallel (reduce it in case of memory errors).')
    parser.add_argument('--temporal_filter_size', type=int, default=7,
                        help='size of final temporal filter.')
    parser.add_argument('--verbose', type=int, default=1, action='store')

    argd = parser.parse_args()

    optfile = os.path.join(argd.dir_poi, 'opt.yaml')
    with open(optfile) as fid:
        opt = yaml.load(fid, Loader=yaml.FullLoader)
    opt['resources_path'] = argd.resources_path
    opt['read_stride'] = 3 * argd.stride
    opt['rec_stride'] = argd.stride
    opt['det_stride'] = max(argd.stride // 2, 1)
    opt['final_mean'] = argd.temporal_filter_size
    if argd.dist_normalization is not None:
        opt['dist_normalization'] = argd.dist_normalization
    opt = create_opt(**opt)

    assert argd.file_output is not None
    file_output = os.path.splitext(argd.file_output)[0]

    print(f"input : {argd.file_video_input}")
    print(f"poi   : {argd.dir_poi}")
    print(f"output: {file_output}")
    assert os.path.isfile(argd.file_video_input)
    logger = logging.getLogger(f"main")

    if os.path.isfile(file_output + '.npz') and argd.skip_if_exists:
        # output file already exists
        dict_out = dict(np.load(file_output + '.npz'))
    else:
        device = 'cuda:%s' % argd.gpu if is_available() and int(argd.gpu) >= 0 else 'cpu'
        print('Running on device: {}'.format(device))

        # compute local scores
        method = create_model(logger, argd.dir_poi, opt, device=device)
        if argd.modality == 'audiovideo':
            dict_out, info_tracks = method.compute_distance_audiovideo(argd.file_video_input, verbose=argd.verbose)
        elif argd.modality == 'onlyaudio':
            dict_out, info_tracks = method.compute_distance_audio(argd.file_video_input, verbose=argd.verbose)
        elif argd.modality == 'onlyvideo':
            dict_out, info_tracks = method.compute_distance_video(argd.file_video_input, verbose=argd.verbose)
        elif argd.modality == 'auto':
            dict_out, info_tracks = method.compute_distance(argd.file_video_input, verbose=argd.verbose)
        else:
            print(f'ERROR: Modality {argd.modality} is not defined!')
            exit()

        # compute global score
        if 'embs_dists' in dict_out:
            embs_track = np.asarray(dict_out['embs_track'])
            embs_dists = np.asarray(dict_out['embs_dists'])
            embs_range = np.asarray(dict_out['embs_range'])
            dict_out['list_tracks'] = np.unique(embs_track)  # list of tracks
            data_each_track = [method.merge_track(embs_dists[embs_track == ids],
                                                  embs_range[embs_track == ids])
                               for ids in dict_out['list_tracks']]  # compute a score for each track
            dict_out['list_scores'] = sum([_[0] for _ in data_each_track], list())
            dict_out['list_location'] = sum([_[1] for _ in data_each_track], list())
            print(dict_out['list_scores'])
            if len(dict_out['list_scores']) == 0:
                dict_out['global_score'] = np.nan
            else:
                dict_out['global_score'] = np.nanmin(dict_out['list_scores'])  # This is the min of the average of 7 segments.
        else:
            dict_out['global_score'] = np.nan

        # save result in npz file
        os.makedirs(os.path.dirname(file_output), exist_ok=True)
        np.savez(file_output + '.npz', **dict_out)  # This saves as a numpy array file

    print('Global score:', dict_out['global_score'])

    if argd.create_plot:
        # Create plot for post video analysis
        # Plot will be saved in the same location as the npz file
        from main_create_plot import create_plot
        create_plot(dict_out, file_output + '.png', opt['dist_normalization'])

    if argd.create_videoout:
        # Generate video
        # The video will be saved in the same location as the npz file
        from main_create_videoout import generate_videoout
        generate_videoout(argd.file_video_input, file_output + '.npz',
                          file_output + '.mp4', opt, verbose=argd.verbose)


if __name__ == "__main__":
    main()
