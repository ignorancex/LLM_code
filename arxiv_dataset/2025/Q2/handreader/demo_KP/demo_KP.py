import json
import pathlib

import cv2
import numpy as np
import rootutils
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from demo_KP.kp_proccesor import process_landmarks
from demo_KP.model import load_model
from demo_KP.utils import Preprocessing, get_vocab, getChicagoTokens, getRuTokens
from src.utils import Decoder


class BaseRecognition:
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        landmarks_list,
        prediction_list,
        verbose,
        video_length: int,
        path_to_save_predictions: str,
        use_ru_model: bool = False,
    ):
        self.video_length = video_length
        self.verbose = verbose
        self.started = None
        self.path_to_save_predictions = path_to_save_predictions
        self.encoder = encoder.eval()
        self.decoder = decoder.eval()

        self.landmarks_list = landmarks_list

        self.prediction_list = prediction_list
        if use_ru_model:
            chars = getRuTokens()
            self.vocab_map, self.inv_vocab_map, self.char_list = get_vocab(chars)
            self.dec = Decoder("_ абвгдежзийклмнопрстуфхцчшщъыьэюяё")

        else:
            chars = getChicagoTokens()
            self.vocab_map, self.inv_vocab_map, self.char_list = get_vocab(chars)
            self.dec = Decoder("_ &'.@abcdefghijklmnopqrstuvwxyz")

        self.sentence = ""
        self.last_letter = "_"
        self.processor = Preprocessing()
        self.cur_tokens = []

        parent_dir = pathlib.Path(__file__).parent.resolve()
        with open(parent_dir / "inference_args.json") as f:
            columns = json.load(f)["selected_columns"]
        self.filtered_columns = [
            idx
            for idx, col in enumerate(columns)
            if any(substring in col for substring in ["pose", "hand"])
        ]

    def run_recognition(self):
        pass

    def run(self):
        """
        Run the recognition model.
        """
        # encdoer part
        if len(self.landmarks_list) >= self.video_length:
            print("Start recognition")
            # list contains numpy arrays with shape [1, 390]
            input_landmarks = np.squeeze(
                np.stack(self.landmarks_list[:], axis=0)
            )  # whole video frames
            # shape is [1, video_length, n_landmarks, 3]
            data = torch.unsqueeze(
                self.processor(torch.from_numpy(input_landmarks), self.filtered_columns), 0
            ).float()
            # shape is [1, window_size_frames, 512]
            encoder_outs = self.encoder(data)
            dec_outs = self.decoder(encoder_outs)
            curr_pred = self.dec.greedy_decode(dec_outs[0].detach().numpy())

            self.prediction_list.extend(curr_pred)
            print("Result: ", " ".join(curr_pred))
            print("Reuslts are saved to ", self.path_to_save_predictions)
            with open(self.path_to_save_predictions, "w") as f:
                f.write(" ".join(self.prediction_list))


class Recognition(BaseRecognition):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        landmarks_list,
        prediction_list,
        verbose,
        video_length: int,
        path_to_save_predictions: str,
        use_ru_model: bool = False,
    ):
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            landmarks_list=landmarks_list,
            prediction_list=prediction_list,
            verbose=verbose,
            video_length=video_length,
            use_ru_model=use_ru_model,
            path_to_save_predictions=path_to_save_predictions,
        )
        self.started = True

    def start(self):
        self.run()


class Runner:
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        path_to_video: str,
        verbose: bool = False,
        use_ru_model: bool = False,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.verbose = verbose
        parent_dir = pathlib.Path(__file__).parent.resolve()
        self.path_to_save_predictions = parent_dir / "Predictions.txt"
        self.cap = cv2.VideoCapture(path_to_video)
        self.video_length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.landmarks_list = []
        self.prediction_list = []
        self.recognizer = Recognition(
            encoder=self.encoder,
            decoder=self.decoder,
            landmarks_list=self.landmarks_list,
            prediction_list=self.prediction_list,
            verbose=self.verbose,
            video_length=self.video_length,
            use_ru_model=use_ru_model,
            path_to_save_predictions=self.path_to_save_predictions,
        )

    def run(self):
        """
        Run the runner.

        """

        print("Start reading video")
        for _ in tqdm(range(self.video_length), desc="Reading video"):
            ret, frame = self.cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks = process_landmarks(rgb_frame)
            self.landmarks_list.append(landmarks)

        self.cap.release()

        self.recognizer.start()


if __name__ == "__main__":
    OmegaConf.register_new_resolver("len", lambda x: len(x))
    parent_dir = pathlib.Path(__file__).parent.resolve()
    config = "config.yaml"
    conf = OmegaConf.load(str(parent_dir / config))
    path_weights = "path_to_weights"
    encoder, decoder = load_model(path_weights, conf)
    runner = Runner(encoder, decoder, use_ru_model=True, path_to_video="path_to_video")
    runner.run()
