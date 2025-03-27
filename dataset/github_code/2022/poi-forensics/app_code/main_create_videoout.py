
import os
import yaml
from config import create_opt


def generate_videoout(filevideo, outputfile, outputvideo, opt, verbose=True):
    from grip_unina.util_read import ReadingResampledVideo
    from grip_unina.util_write import OutputBoxes
    from grip_unina.util_write import WritingVideo
    from grip_unina.extraction import add_audio_on_video
    from tqdm import tqdm

    op2 = OutputBoxes(outputfile, margin=(opt['model']['clip_length'] - opt['model']['clip_stride'])//2,
                      return_frame=False, color=(200, 200, 200))
    tempvideo = outputvideo + '_tmp.mp4'

    with ReadingResampledVideo(filevideo, opt['fps'], opt['read_stride']) as video:
        with WritingVideo(tempvideo, fps=opt['fps'], vid_configure=opt['output_ffmpeg_params']) as write_video:
            if verbose:
                print(f'Reading video {filevideo} of {video.get_number_frames()} frames with {video.get_fps()} fps.')

            ops = [video, op2.reset(), write_video]
            if verbose:
                print('', flush=True)
                pbar = tqdm(total=len(video))

            count = 0
            while True:
                try:
                    out = count
                    for index_op in range(len(ops)):
                        out = ops[index_op](out)
                except StopIteration:
                    break
                count = count + 1

                if verbose:
                    pbar.update(1)

    add_audio_on_video(tempvideo, filevideo, outputvideo)
    os.remove(tempvideo)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='create video from output.')
    parser.add_argument('--file_video_input', type=str,
                        help='input video file (with extension: .mp4, .avi).')
    parser.add_argument('--dir_poi', type=str,
                        help='features directory of POI.')
    parser.add_argument('--file_npz', type=str,
                        help='the numpy file (with extension .npz).')
    parser.add_argument('--output_video', type=str,
                        help='output video (with extension .mp4).')
    parser.add_argument('--verbose', type=int, default=1)
    argd = parser.parse_args()

    optfile = os.path.join(argd.dir_poi, 'opt.yaml')
    with open(optfile) as fid:
        opt = yaml.load(fid, Loader=yaml.FullLoader)
    opt = create_opt(**opt)

    generate_videoout(argd.file_video_input, argd.file_npz, argd.output_video, opt, verbose=argd.verbose)