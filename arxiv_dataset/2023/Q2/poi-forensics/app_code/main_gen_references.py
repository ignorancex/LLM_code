import os
import argparse
import glob
import tqdm

if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                        description='script to extract features from the reference videos.')
    parser.add_argument('--dir_videos', type=str,
                        help='input directory with the reference video files (with extensions: .mp4, .avi).')
    parser.add_argument('--dir_poi', type=str,
                        help='output directory where the extracted features will be saved.')
    parser.add_argument('--resources_path', type=str, default="./resources/",
                        help='directory with networks weights.')
    parser.add_argument('--models', type=str, default='idreveal,poiforensics',
                        help='extraction feature of these models.')
    parser.add_argument('--gpu', type=int, default=0,
                        help='index of GPU to use (set to -1 for not using the GPU).')
    parser.add_argument('--workers', type=int, default=0,
                        help='number of videos analyzed in parallel (set to 0 to disable the parallel).')
    parser.add_argument('--stride', type=int, default=32,
                        help='number of frames analyzed in parallel (reduce it in case of memory errors).')
    parser.add_argument('--tracker_iou_th', type=int, default=0.4,
                        help='threshold of face tracking.')
    parser.add_argument('--verbose', type=int, default=1)
    argd = parser.parse_args()

    outputdir = argd.dir_poi
    listfile = glob.glob(os.path.join(argd.dir_videos, '*.mp4')) + \
               glob.glob(os.path.join(argd.dir_videos, '*.avi'))

    print('Number of found videos:', len(listfile), flush=True)
    listcmd = list()
    for filepath in listfile:
        filename = os.path.splitext(os.path.basename(filepath))[0]
        cmd = list()
        for model in argd.models.split(','):
            cmd.append( \
                f"python main_feat_extractor.py " \
                f"--video_input '{filepath}' " \
                f"--file_boxes  '{outputdir}/feats/{filename}/boxes.npz' " \
                f"--file_spec   '{outputdir}/feats/{filename}/spec.npy' " \
                f"--file_3dmm   '{outputdir}/feats/{filename}/3dmm.npz' " \
                f"--file_track  '{outputdir}/track/track_{filename}.mp4' " \
                f"--model       '{model}' " \
                f"--dir_ref     '{outputdir}/app_{model}/{filename}' " \
                f"--dir_faces   '{outputdir}/faces/{filename}/' " \
                f"--file_opt    '{outputdir}/app_{model}/opt.yaml' " \
                f"--gpu {argd.gpu} " \
                f"--verbose {argd.verbose} " \
                f"--resources_path '{argd.resources_path}' " \
                f"--stride {argd.stride} " \
                f"--tracker_iou_th {argd.tracker_iou_th} " \
            )
        listcmd.append("&& ".join(cmd))

    if argd.workers < 1:
        for item in tqdm.tqdm(listcmd, total=len(listcmd)):
            os.system(item)
    else:
        from multiprocessing import get_context
        ctx = get_context("fork")
        with ctx.Pool(argd.workers) as pool:
            list(tqdm.tqdm(pool.imap_unordered(os.system, listcmd), total=len(listcmd)))
