#
# Copyright (c) 2023 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.txt
# (included in this package) and online at
# http://www.grip.unina.it/download/LICENSE_OPEN.txt
#


def check_gpu(logger):
    from torch.cuda import is_available as is_available_cuda
    if is_available_cuda():
        from torch.cuda import get_device_name, device_count
        number_of_devices = device_count()
        logger.info(f"Cuda is available for {number_of_devices} GPUs")
        for device in range(number_of_devices):
            logger.info(f"{device} GPU: {get_device_name(device)}")
        device = 'cuda:0'
    else:
        logger.warning("Cuda is not available!")
        device = 'cpu'
    return device


def test_gpu(device):
    import torch
    return torch.ones((6 * 1024 * 1024 * 1024), dtype=torch.float16, device=device)


class IntervalLocation:
    def __init__(self, loc):
        self.loc=loc

    def get_start_time(self):
        return str(self.loc[0])

    def get_end_time(self):
        return str(self.loc[1])


class ResponseValue:
    def __init__(self, logit, id_model, loc=None):
        self.logit = logit
        self.id_model = id_model
        self.loc = loc

    def has_location(self):
        return self.loc is not None

    def get_location(self):
        return self.loc


class ResponseError:
    def __init__(self, reason, explanation):
        self.reason = reason
        self.explanation = explanation

        allowed_values = ["UnsupportedAnalysisType", "UnsupportedModality",
                          "UnsupportedFormat", "NonConformingDomain",
                          "ErrorProcessingAsset", "RedundantAnalysis", "Other"]  # noqa: E501
        if reason not in allowed_values:
            raise ValueError(
                "Invalid value for `reason` ({0}), must be one of {1}"
                .format(reason, allowed_values)
            )


def image_check_and_get_size(filename):
    import os
    import requests

    filename = filename.replace('\\', '/')
    if filename.startswith('file:///'):
        flag_exist = os.path.isfile(filename[7:])
    else:
        with requests.head(filename) as ret:
            flag_exist = ret.ok

    if not flag_exist:
        raise FileNotFoundError

    from PIL import Image, UnidentifiedImageError
    try:
        import urllib
        with urllib.request.urlopen(filename) as response:
            img = Image.open(response)
            img.load()
            size = (1, img.size[0], img.size[1])
    except UnidentifiedImageError:
        raise TypeError
    except FileNotFoundError:
        raise FileNotFoundError
    except:
        raise TypeError

    return size


def audio_check_and_get_size(filename):
    import os
    import requests
    from pydub import AudioSegment

    filename = filename.replace('\\', '/')
    if filename.startswith('file:///'):
        flag_exist = os.path.isfile(filename[7:])
    else:
        with requests.head(filename) as ret:
            flag_exist = ret.ok

    if not flag_exist:
        raise FileNotFoundError

    try:
        import urllib
        from urllib.parse import urlparse
        from io import BytesIO
        typ = os.path.splitext(filename.split('/')[-1])[1][1:]
        with urllib.request.urlopen(urlparse(filename, scheme='file').geturl()) as response:
            with BytesIO(response.read()) as dat:
                audiofile = AudioSegment.from_file(dat, typ)
        size = (int(len(audiofile)), 1, int(audiofile.channels))
    except:
        raise TypeError

    return size


def video_check_and_get_size(filename):
    import os
    import requests
    import cv2

    filename = filename.replace('\\', '/')
    if filename.startswith('file:///'):
        flag_exist = os.path.isfile(filename[7:])
    else:
        with requests.head(filename) as ret:
            flag_exist = ret.ok

    if not flag_exist:
        raise FileNotFoundError

    video_cap = cv2.VideoCapture()
    video_cap.setExceptionMode(True)
    try:
        video_cap.open(filename)
    except cv2.error:
        video_cap.release()
        raise TypeError
    size = (
        int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
           )
    video_cap.release()
    return size
