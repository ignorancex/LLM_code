import os
from torch.cuda import is_available
import argparse
import numpy as np

from config import create_opt, get_extraction_opt
from grip_unina.extraction import extract_boxes, generate_video_tracks, generate_clips_tracks, add_audio_on_video


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--resources_path', type=str, default="./resources/")
    parser.add_argument('--stride', type=int, default=32)
    parser.add_argument('--tracker_iou_th', type=float, default=0.4)
    parser.add_argument('--video_input', type=str)
    parser.add_argument('--file_boxes', type=str)
    parser.add_argument('--file_spec', type=str, default=None)
    parser.add_argument('--file_3dmm', type=str, default=None)
    parser.add_argument('--file_opt', type=str, default=None)
    parser.add_argument('--file_track', type=str, default=None)
    parser.add_argument('--model', type=str, default='poiforensics')
    parser.add_argument('--dir_ref', type=str, default=None)
    parser.add_argument('--dir_faces', type=str, default=None)
    argd = parser.parse_args()

    device = 'cuda:%s' % argd.gpu if is_available() and int(argd.gpu) >= 0 else 'cpu'
    opt = create_opt(resources_path=argd.resources_path,
                     read_stride=3*argd.stride, rec_stride=argd.stride, det_stride=max(argd.stride//2, 1),
                     face_iou_threshold=argd.tracker_iou_th,
                     model=argd.model)

    print('Running on device: {}'.format(device))
    print(f"input : {argd.video_input}")
    assert os.path.isfile(argd.video_input)
    print(opt)

    if argd.file_opt is not None:
        if not os.path.isfile(argd.file_opt):
            import yaml
            os.makedirs(os.path.dirname(argd.file_opt), exist_ok=True)
            with open(argd.file_opt, 'w') as fid:
                documents = yaml.dump(get_extraction_opt(opt), fid)

    if not os.path.isfile(argd.file_boxes):
        print(f"\noutput: {argd.file_boxes}")
        os.makedirs(os.path.dirname(argd.file_boxes), exist_ok=True)
        dict_out, info_tracks = extract_boxes(argd.video_input, device, opt, verbose=argd.verbose)
        np.savez(argd.file_boxes, **dict_out)
        print(f"\ndone: {argd.file_boxes}", flush=True)

    if argd.file_track is not None:
        if not os.path.isfile(argd.file_track):
            print(f"\noutput: {argd.file_track}")
            os.makedirs(os.path.dirname(argd.file_track), exist_ok=True)
            tempvideo = argd.file_track + '_tmp.mp4'
            generate_video_tracks(argd.video_input, argd.file_boxes, tempvideo, opt, verbose=argd.verbose)
            add_audio_on_video(tempvideo, argd.video_input, argd.file_track)
            os.remove(tempvideo)
            print(f"\ndone: {argd.file_track}", flush=True)

    if argd.file_spec is not None:
        if not os.path.isfile(argd.file_spec):
            print(f"\noutput: {argd.file_spec}")
            os.makedirs(os.path.dirname(argd.file_spec), exist_ok=True)
            from grip_unina.poi_forensics import extract_spec
            audiodata = extract_spec(argd.video_input, opt, verbose=argd.verbose)
            np.save(argd.file_spec, audiodata)
            print(f"\ndone: {argd.file_spec}", flush=True)

    if argd.file_3dmm is not None:
        if not os.path.isfile(argd.file_3dmm):
            print(f"\noutput: {argd.file_3dmm}")
            os.makedirs(os.path.dirname(argd.file_3dmm), exist_ok=True)
            from grip_unina.id_reveal import extract_3dmm
            dict_out = extract_3dmm(argd.video_input, argd.file_boxes, device, opt, verbose=argd.verbose)
            np.savez(argd.file_3dmm, **dict_out)
            print(f"\ndone: {argd.file_3dmm}", flush=True)

    if argd.dir_ref is not None:
        if not os.path.isdir(argd.dir_ref):
            print(f"\noutput: {argd.dir_ref}")
            typ = opt['model']['type']
            if typ == 'poi_forensics':
                from grip_unina.poi_forensics import extract_feats_poi_forensics
                dict_out = extract_feats_poi_forensics(argd.video_input,
                             argd.file_boxes, argd.file_spec,
                             device=device, opt=opt, verbose=argd.verbose)
            elif typ == 'id_reveal':
                from grip_unina.id_reveal import extract_feats_idreavel
                dict_out = extract_feats_idreavel(argd.file_3dmm,
                             device=device, opt=opt, verbose=argd.verbose)
            elif typ == 'face_recognition':
                from grip_unina.face_recognition import extract_feats_facerec
                dict_out = extract_feats_facerec(argd.video_input,
                             argd.file_boxes,
                             device=device, opt=opt, verbose=argd.verbose)
            else:
                assert False

            if 'embs_track' in dict_out:
                dict_out = {k: np.asarray(dict_out[k]) for k in dict_out}
                os.makedirs(argd.dir_ref, exist_ok=True)
                embs_track = dict_out['embs_track']

                for t in np.unique(embs_track):
                    dict_out_t = {k: dict_out[k][embs_track == t] for k in dict_out if k != 'embs_track'}
                    np.savez(os.path.join(argd.dir_ref, 'embs_track%d.npz' % t), **dict_out_t)
            print(f"\ndone: {argd.dir_ref}", flush=True)

    if argd.dir_faces is not None:
        if not os.path.isdir(argd.dir_faces):
            if argd.dir_ref is not None:
                list_good_track = [int(_[10:-4]) for _ in os.listdir(argd.dir_ref) if _.startswith('embs_track')]
            else:
                list_good_track = None
            print(f"\noutput: {argd.dir_faces}", list_good_track)
            os.makedirs(argd.dir_faces, exist_ok=True)
            generate_clips_tracks(argd.video_input, argd.file_boxes, argd.dir_faces, device=device, fps=opt['fps'],
                                  write_one=True, compute_landmarks=False, list_good_track=list_good_track, verbose=argd.verbose)
            print(f"\ndone: {argd.dir_faces}", flush=True)