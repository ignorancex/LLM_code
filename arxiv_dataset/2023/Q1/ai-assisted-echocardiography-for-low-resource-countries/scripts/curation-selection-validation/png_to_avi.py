import argparse
import os
from time import time

import cv2
import yaml
from tqdm import tqdm


def timer_func(func):
    """
    This function shows the execution time of
    the function object passed
    """

    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2 - t1):.4f}s')
        return result

    return wrap_func


@timer_func
def conver_pngframes_to_avi(participant_directory: str,
                            video_output_pathname: str,
                            fps: int
                            ) -> None:
    """
    Convert image frames in png format to an avi file with a given fps.
    """

    for T_days_i in sorted(os.listdir(participant_directory)):
        days_i_path = participant_directory + T_days_i
        for preprocessed_frame_path_i in sorted(os.listdir(days_i_path)):
            preprocessed_frame_path = days_i_path + '/' + preprocessed_frame_path_i

            for clips_i in sorted(os.listdir(preprocessed_frame_path)):
                if clips_i[0] == 'c':
                    clips_i_path = preprocessed_frame_path + '/' + clips_i

                    images = [img for img in sorted(os.listdir(clips_i_path)) if img.endswith(".png")]
                    frame = cv2.imread(os.path.join(clips_i_path, images[0]))
                    height, width, layers = frame.shape

                    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
                    path_preprocessed_frame_path_ = preprocessed_frame_path + '/' + video_output_pathname + '/'
                    if not os.path.isdir(os.path.join(path_preprocessed_frame_path_)):
                        try:
                            os.makedirs(path_preprocessed_frame_path_)
                        except FileExistsError:
                            pass

                    path_clips_i_avi = path_preprocessed_frame_path_ + clips_i + '.avi'
                    print(path_clips_i_avi)
                    video = cv2.VideoWriter(path_clips_i_avi, fourcc, fps, (width, height))

                    for image in tqdm(images):
                        video.write(cv2.imread(os.path.join(clips_i_path, image)))

                    cv2.destroyAllWindows()
                    video.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Specify config.yml with paths files')
    args = parser.parse_args()

    with open(args.config, 'r') as yml:
        config = yaml.load(yml, Loader=yaml.FullLoader)

    conver_pngframes_to_avi(
        config['participant_directory'],
        config['video_output_pathname'],
        config['fps'],
    )
