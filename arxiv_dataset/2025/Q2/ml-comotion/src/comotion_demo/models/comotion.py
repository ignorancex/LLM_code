# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import torch
from scenedetect.detectors import ContentDetector
from torch import nn

from ..utils import dataloading, helper, smpl_kinematics, track
from . import detect, refine


class CoMotion(nn.Module):
    """CoMotion network."""

    def __init__(self, use_coreml=False, pretrained=True):
        """Initialize CoMotion.

        Args:
        ----
            use_coreml: use CoreML version of the detection model (macOS only).
            pretrained: load pre-trained CoMotion modules.

        """
        super().__init__()

        if use_coreml:
            self.detection_model = detect.CoMotionDetectCoreML()
        else:
            self.detection_model = detect.CoMotionDetect(pretrained=pretrained)

        self.update_step = refine.PoseRefinement(
            self.detection_model.feat_dim,
            self.detection_model.cfg.pose_embed_dim,
            pretrained=pretrained,
        )

        self.smpl_decoder = smpl_kinematics.SMPLKinematics()
        self.shot_detector = ContentDetector(threshold=50.0, min_scene_len=3)
        self.frame_count = 0

    def init_tracks(self, image_res):
        """Initialize track handler."""
        self.handler = track.TrackHandler(track.default_dims, image_res)

    @torch.inference_mode()
    def forward(self, image, K, detection_only=False, use_mps=False):
        """Perform detection and tracking given a new image.

        Input images are accepted at any resolution, resizing and cropping is
        handled automatically and output 2D keypoints will be provided at the
        original input resolution.

        Args:
        ----
            image: Input image (C x H x W) float (0-1) or uint8 (0-255) tensor
            K: Intrinsics matrix (2 x 3) float tensor
            detection_only: Flag whether to only run initial detection stage
            use_mps: Flag whether to run update step on MPS on MacOS

        """
        # Prepare inputs
        device = next(self.parameters()).device
        if use_mps:
            self.update_step.to("mps")
            device = "cpu"

        # Prepare inputs
        cropped_image, cropped_K = dataloading.prepare_network_inputs(image, K, device)
        K = K.to(device)

        # Get detections
        detect_out = self.detection_model(cropped_image, cropped_K)
        nms_out = detect.decode_network_outputs(
            K,
            self.smpl_decoder,
            detect_out,
            std=0.08,
            iou_thr=0.4,
            conf_thr=0.1,
        )

        if detection_only:
            return nms_out

        def call_update(s: track.TrackTensorState):
            # Prepare inputs
            update_args = [
                detect_out.image_features,
                cropped_K,
                s.betas,
                s.pose,
                s.trans,
                s.pred_3d,
                s.hidden,
            ]
            if use_mps:
                update_args = [arg.to("mps") for arg in update_args]

            # Run update step
            updated_params = self.update_step(*update_args)
            updated_params = refine.RefineOutput(
                **{k: v.to(device) for k, v in updated_params._asdict().items()}
            )

            # Update track state
            s.pose = detect.get_smpl_pose(
                updated_params.delta_root_orient,
                updated_params.delta_body_pose,
            )
            s.trans = updated_params.trans
            s.hidden = updated_params.hidden
            s.pred_3d = self.smpl_decoder(
                s.betas, s.pose, s.trans, output_format="joints_face"
            )
            s.pred_2d = helper.project_to_2d(K, s.pred_3d)

        # Detect shot changes
        image_np = image.detach().to("cpu").permute(1, 2, 0).numpy()
        image_np = image_np[:, :, ::-1]  # RGB2BGR
        shots = self.shot_detector.process_frame(self.frame_count, image_np)
        self.frame_count += 1
        is_new_shot = len(shots) > 1 if self.frame_count > 1 else False
        return nms_out, self.handler.update(
            nms_out, call_update, shot_reset=is_new_shot
        )
