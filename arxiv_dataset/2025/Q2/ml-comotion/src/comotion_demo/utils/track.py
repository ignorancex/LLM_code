# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from __future__ import annotations

from typing import Dict

import tensordict
import torch
from tqdm import tqdm

from . import helper, smpl_kinematics

default_dims = {
    "betas": (smpl_kinematics.BETA_DOF,),
    "pose": (smpl_kinematics.POSE_DOF,),
    "trans": (smpl_kinematics.TRANS_DOF,),
    "pred_3d": (smpl_kinematics.NUM_PARTS_AUX, 3),
    "pred_2d": (smpl_kinematics.NUM_PARTS_AUX, 2),
    "hidden": (512,),
    "id": (1,),
}


@tensordict.tensorclass
class TrackTensorState:
    betas: torch.Tensor
    pose: torch.Tensor
    trans: torch.Tensor
    pred_3d: torch.Tensor
    pred_2d: torch.Tensor
    hidden: torch.Tensor
    id: torch.Tensor


class MonitorAttribute:
    def __init__(self, ema_coef=[0.75, 0.9]):
        self.log = []
        self.ema_log = {c: [] for c in ema_coef}

    def update(self, x):
        self.log.append(x)
        for c in self.ema_log:
            if len(self.ema_log[c]) == 0:
                self.ema_log[c].append(x)
            else:
                x_ = c * self.ema_log[c][-1] + (1 - c) * x
                self.ema_log[c].append(x_)


class TrackHealthMonitor:
    """Keep track of various statistics about tracks to determine when to end them."""

    def __init__(
        self,
        init_timestep,
        match_discard_thr=0.15,
        inbounds_discard_thr=0.15,
        ema_coef=0.8,
    ):
        self.match_discard_thr = match_discard_thr
        self.inbounds_discard_thr = inbounds_discard_thr
        self.ema_coef = ema_coef

        self.init_timestep = init_timestep
        self.age = 0
        self.data_keys = ["root pos", "root vel", "match", "overlap", "inbounds"]
        self.attributes = {
            k: MonitorAttribute(ema_coef=[ema_coef]) for k in self.data_keys
        }
        self.is_redundant = False

    def update(self, root_pos, match_scores, track_overlap, inbounds_pct):
        self.age += 1
        if self.age > 1:
            diff = root_pos - self.attributes["root pos"].log[-1]
            diff[..., 2] /= 3  # Rescale z down
            root_vel = diff.norm()
        else:
            root_vel = torch.zeros([])
        new_vals = [root_pos, root_vel, match_scores, track_overlap, inbounds_pct]
        for k, v in zip(self.data_keys, new_vals):
            self.attributes[k].update(v)

    def get(self, key, ema_coef=None, use_ema=False):
        if use_ema:
            ema_coef = self.ema_coef
        if ema_coef is None:
            return self.attributes[key].log[-1]
        else:
            return self.attributes[key].ema_log[ema_coef][-1]

    def get_current_health(self):
        if self.is_redundant:
            # Marked redundant
            return 0
        elif self.age <= 3 and self.get("match") < self.match_discard_thr:
            # Bad new track
            return 0
        elif self.get("match", self.ema_coef) < self.match_discard_thr:
            # Track not matching well
            return 0
        elif self.get("inbounds", self.ema_coef) < self.inbounds_discard_thr:
            # Track out-of-bounds
            return 0
        else:
            return 1


class TrackHandler:
    def __init__(
        self,
        ref_dims=None,
        res=None,
        init_conf_thr=0.2,
        init_match_thr=0.75,
        missing_match_thr=0.2,
        max_output_tracks=48,
        overlap_thr=0.6,
        overlap_time_thr=20,
        vel_outlier_thr=0.25,
    ):
        if ref_dims is None:
            ref_dims = default_dims
        if res is None:
            res = (512, 512)
        self.ref_dims = ref_dims
        self.res = res

        self.current_step = 0
        self.last_track_id = 0
        self.current_tracks: TrackTensorState | None = None
        self.health_monitors: Dict[int, TrackHealthMonitor] = {}
        self.cached_tracks = []
        self.cached_detections = []
        self.collapse_count = {}
        self.init_conf_thr = init_conf_thr
        self.init_match_thr = init_match_thr
        self.missing_match_thr = missing_match_thr
        self.max_output_tracks = max_output_tracks
        self.overlap_thr = overlap_thr
        self.overlap_time_thr = overlap_time_thr
        self.vel_outlier_thr = vel_outlier_thr

    def clear_tracks(self):
        """Clear existing tracks, preserve last_track_id."""
        self.current_tracks = None

    def convert_to_outputs(self, device="cpu"):
        if self.current_tracks is None:
            empty_state = {
                k: torch.zeros(1, 1, *ref_shape, device=device)
                for k, ref_shape in self.ref_dims.items()
            }
            tracks = TrackTensorState(**empty_state, batch_size=(1, 1))
        else:
            tracks = self.current_tracks

        num_tracks = tracks.shape[1]
        if num_tracks > self.max_output_tracks:
            tracks = tracks[:, : self.max_output_tracks]
        else:
            to_pad = self.max_output_tracks - num_tracks
            tracks = tensordict.pad(tracks, [0, 0, 0, to_pad])

        # Reset so that padded and truncated versions are compatible
        tracks = TrackTensorState(**tracks.to_dict(), batch_size=tracks.batch_size)
        return tracks

    def get_state_from_detections(self, detections):
        to_init = ["id"]
        if "hidden" not in detections:
            to_init += ["hidden"]

        to_ignore = ["scale", "conf"]
        track_state = {k: v for k, v in detections.items() if k not in to_ignore}
        conf = detections["conf"]
        batch_size = track_state["betas"].shape[:-1]
        device = track_state["betas"].device
        for k in to_init:
            track_state[k] = torch.zeros(*batch_size, *self.ref_dims[k], device=device)
        track_state["id"][:] = torch.arange(batch_size[1]).unsqueeze(-1)
        return TrackTensorState(**track_state, batch_size=batch_size), conf

    def compare_detections(self, p0, p1, normalize_factor=1024):
        """Get match scores between all pairs of predictions."""
        p0 = p0 / normalize_factor
        p1 = p1 / normalize_factor
        c0 = torch.ones_like(p0[..., 0])
        c1 = torch.ones_like(p1[..., 0])

        return helper.normalized_weighted_score(p0, c0, p1, c1)

    def initialize_tracks(self, detections, update_fn, init_state=None, conf=None):
        """Get bounding boxes from input detections, and initialize new tracks."""
        if init_state is None:
            init_state, conf = self.get_state_from_detections(detections)
        ref_2d = init_state.pred_2d.clone()
        num_detections = ref_2d.shape[1]
        if num_detections > 0:
            update_fn(init_state)

            # Check for self-consistency in detections (only keep samples that match)
            match_scores = self.compare_detections(init_state.pred_2d[0], ref_2d[0])
            to_keep = match_scores.diagonal() > self.init_match_thr
            if conf is not None:
                # We have a higher bar for initialization than for matching
                conf_filter = torch.sigmoid(conf.squeeze()) > self.init_conf_thr
                to_keep = to_keep & conf_filter
            match_scores = match_scores.diagonal()[to_keep]

            num_to_keep = to_keep.sum()
            if num_to_keep > 0:
                init_state = init_state[:, to_keep]
                new_ids = (
                    torch.arange(num_to_keep, device=init_state.id.device)
                    + self.last_track_id
                    + 1
                )
                self.last_track_id += num_to_keep
                init_state.id[:] = new_ids.float().unsqueeze(-1)

                for track in init_state[0]:
                    self.health_monitors[track.id.item()] = TrackHealthMonitor(
                        self.current_step
                    )

                return init_state, match_scores

        return None, None

    def initialize_missing_tracks(self, detections, update_fn):
        """Check for missing detections that should be initialized."""
        # Check overlap of tracks with input detections
        new_state, conf = self.get_state_from_detections(detections)
        num_tracks = self.current_tracks.pred_2d.shape[1]
        num_detections = new_state.pred_2d.shape[1]

        if num_detections > 0:
            match_scores = self.compare_detections(
                self.current_tracks.pred_2d[0], new_state.pred_2d[0]
            )
            track_max_match_scores = match_scores.max(1)[0]
            detection_max_match_score = match_scores.max(0)[0]
            unmatched = (
                (detection_max_match_score < self.missing_match_thr).nonzero().flatten()
            )

            if len(unmatched) > 0:
                missing = new_state[:, unmatched]
                missing_conf = conf[:, unmatched]
                new_tracks, new_match_scores = self.initialize_tracks(
                    None, update_fn, missing, missing_conf
                )
                if new_tracks is not None and len(new_match_scores) > 0:
                    self.current_tracks = torch.cat(
                        [self.current_tracks, new_tracks], 1
                    )
                    track_max_match_scores = torch.cat(
                        [track_max_match_scores, new_match_scores], -1
                    )

            return track_max_match_scores

        else:
            return torch.zeros(num_tracks, device=self.current_tracks.pred_2d.device)

    def update_track_health(self, match_scores):
        """Report various properties of current tracks."""
        root_pos = self.current_tracks[0].trans.cpu()
        pred_2d = self.current_tracks[0].pred_2d.cpu()

        # Check overlap with other tracks
        self_similarity = self.compare_detections(pred_2d, pred_2d)
        self_similarity.diagonal().fill_(0)
        track_overlap = self_similarity.max(1)[0]

        # Calculate percentage of keypoints that are inbounds
        inbounds = helper.check_inbounds(pred_2d, self.res)
        inbounds_pct = inbounds.float().mean(-1)

        ids = self.current_tracks[0].id
        update_args = [root_pos, match_scores, track_overlap, inbounds_pct]
        for id, *args in zip(ids, *update_args):
            id = id.item()
            self.health_monitors[id].update(*args)

        # Look for duplicate/collapsed tracks
        num_tracks = len(root_pos)
        triu = torch.triu_indices(num_tracks, num_tracks, offset=1)
        collapse_candidates = triu.T[
            self_similarity[triu[0], triu[1]] > self.overlap_thr
        ]
        for c0, c1 in collapse_candidates:
            i0, i1 = ids[c0].item(), ids[c1].item()
            if (i0, i1) not in self.collapse_count:
                self.collapse_count[(i0, i1)] = []
            self.collapse_count[(i0, i1)] += [self.current_step]
            a0 = self.health_monitors[i0].age
            a1 = self.health_monitors[i1].age
            m0 = self.health_monitors[i0].get("match", use_ema=True)
            m1 = self.health_monitors[i1].get("match", use_ema=True)

            # Clear spurious early detection
            if a0 < 5 and a1 < 5:
                # Similar age, look at match score
                if m1 > m0:
                    self.health_monitors[i0].is_redundant = True
                else:
                    self.health_monitors[i1].is_redundant = True
            elif a0 < 3:
                self.health_monitors[i0].is_redundant = True
            elif a1 < 3:
                self.health_monitors[i0].is_redundant = True
            else:
                # Count how many overlap samples we have in the last second
                recent_overlap_count = torch.tensor(self.collapse_count[(i0, i1)])
                recent_overlap_count = (recent_overlap_count - self.current_step).abs()
                recent_overlap_count = (recent_overlap_count < 30).sum()

                v0 = torch.tensor(
                    self.health_monitors[i0].attributes["root vel"].log[-30:]
                ).max()
                v1 = torch.tensor(
                    self.health_monitors[i0].attributes["root vel"].log[-30:]
                ).max()

                if recent_overlap_count > self.overlap_time_thr:
                    if v0 > self.vel_outlier_thr and v1 < self.vel_outlier_thr:
                        self.health_monitors[i0].is_redundant = True
                    elif v0 < self.vel_outlier_thr and v1 > self.vel_outlier_thr:
                        self.health_monitors[i1].is_redundant = True
                    elif m1 > m0:
                        self.health_monitors[i0].is_redundant = True
                    else:
                        self.health_monitors[i1].is_redundant = True

    def clear_invalid_tracks(self):
        """Delete any tracks whose health score is below threshold."""
        track_health = []
        for track in self.current_tracks[0]:
            track_health.append(
                self.health_monitors[track.id.item()].get_current_health()
            )
        track_health = torch.tensor(track_health)
        self.current_tracks = self.current_tracks[:, track_health > 0]

    def update(self, curr_detections, update_fn, shot_reset=False):
        """Run tracking update step."""
        self.cached_detections.append(curr_detections["pred_2d"])
        self.current_step += 1
        match_scores = None
        if shot_reset:
            self.clear_tracks()

        if self.current_tracks is not None:
            # Update current tracks and compare to new detections
            update_fn(self.current_tracks)
            match_scores = self.initialize_missing_tracks(curr_detections, update_fn)

        else:
            # No existing tracks, initialize new ones from current detections
            self.current_tracks, match_scores = self.initialize_tracks(
                curr_detections, update_fn
            )

        if self.current_tracks is not None:
            if self.current_tracks.pred_2d.shape[1] == 0:
                self.current_tracks = None
            else:
                self.update_track_health(match_scores.cpu())
                self.clear_invalid_tracks()
                if self.current_tracks.pred_2d.shape[1] == 0:
                    self.current_tracks = None

        return self.convert_to_outputs(device=curr_detections["pred_2d"].device)


def rearrange_preds(preds, track_ids):
    num_timesteps = preds.shape[0]
    dim_ref = preds.shape[2:]

    unique_ids = track_ids.unique()
    num_tracks = len(unique_ids)
    id_lookup = {
        id_idx.item(): id.item()
        for id_idx, id in zip(torch.arange(num_tracks), unique_ids)
    }

    full_pred = torch.zeros(
        num_timesteps, num_tracks, *dim_ref, dtype=preds.dtype, device=preds.device
    )

    for id_idx, id in id_lookup.items():
        if id != 0:
            src_i0, src_i1 = (track_ids == id).nonzero(as_tuple=True)
            dst_i1 = torch.ones_like(src_i1).fill_(id_idx)
            full_pred[src_i0, dst_i1] = preds[src_i0, src_i1]

    return full_pred, id_lookup


def padded_reshaped(arr, frame_idxs, ids, num_frames):
    unique_ids, remapped_ids = ids.unique(return_inverse=True)
    max_tracks = len(unique_ids)
    dim_ref = [d for d in arr.shape]
    dim_ref = [num_frames, max_tracks] + dim_ref[1:]
    dst_arr = torch.zeros(*dim_ref, device=arr.device, dtype=arr.dtype)
    dst_arr[frame_idxs, remapped_ids] = arr

    return dst_arr


def query_range(tracks, i0, i1):
    to_use = (tracks["frame_idx"] >= i0) & (tracks["frame_idx"] < i1)
    filtered = {k: v[to_use] for k, v in tracks.items()}
    reshaped = {
        k: padded_reshaped(v, filtered["frame_idx"] - i0, filtered["id"], i1 - i0)
        for k, v in filtered.items()
    }
    return reshaped


def calculate_cleanup_ranges(ids, sub_window_size=100, overlap=20, unique_id_thr=100):
    """Breakdown very long videos into subsections.

    For videos with thousands of frames and lots of people it gets unwieldy to
    perform track clean up operation on everything simultaneously, so we break
    the video down into subclips.
    """
    num_frames = len(ids)

    rngs = []
    r0 = 0
    for i in range(0, num_frames, sub_window_size):
        r1 = min(i + sub_window_size + overlap, num_frames)
        unique_count = ids[r0:r1].unique().numel()
        if unique_count > unique_id_thr:
            rngs += [(r0, r1)]
            r0 = r1 - overlap

    if r0 != r1:
        rngs += [(r0, r1)]

    return rngs


def combine_refs(all_refs, rngs):
    combined = {}
    for track_ref, rng in zip(all_refs, rngs):
        offset = rng[0]
        for track_id, (i0, i1) in track_ref.items():
            i0, i1 = i0 + offset, i1 + offset
            if track_id not in combined:
                combined[track_id] = [i0, i1]
            else:
                i0_, i1_ = combined[track_id]
                combined[track_id] = [min(i0, i0_), max(i1, i1_)]
    return combined


def convert_to_idxs(track_ref, ids):
    all_idxs = []
    for curr_id, (i0, i1) in track_ref.items():
        sample_idxs = (ids[i0:i1] == curr_id).nonzero()
        sample_idxs[:, 0] += i0
        all_idxs.append(sample_idxs)

    all_idxs = torch.cat(all_idxs, 0)
    frame_idxs, track_idxs = all_idxs.unbind(1)

    return frame_idxs, track_idxs


def cleanup_tracks(
    preds,
    K,
    smpl_decoder,
    min_matched_frames=4,
    min_match_score=0.6,
    detect_conf_thr=0.2,
    return_ious=False,
    rng=None,
    use_tqdm=False,
    unique_id_thr=100,
):
    if rng is None:
        ids = preds["tracks"]["id"][0]
        rngs = calculate_cleanup_ranges(ids, unique_id_thr=unique_id_thr)
        rngs_ = tqdm(rngs) if use_tqdm else rngs
        all_refs = [
            cleanup_tracks(
                preds,
                K,
                smpl_decoder,
                min_matched_frames,
                min_match_score,
                detect_conf_thr,
                return_ious,
                rng=rng,
            )
            for rng in rngs_
        ]

        return combine_refs(all_refs, rngs)

    else:
        i0, i1 = rng
        detect_2d = torch.nn.utils.rnn.pad_sequence(
            [p[0] for p in preds["detections"]["pred_2d"][i0:i1]], batch_first=True
        )
        detect_conf = torch.nn.utils.rnn.pad_sequence(
            [p[0] for p in preds["detections"]["conf"][i0:i1]], batch_first=True
        )
        ids, betas, pose, trans = [
            preds["tracks"][k][0][i0:i1] for k in ["id", "betas", "pose", "trans"]
        ]

    # Zero out low confidence detections
    detect_conf = torch.sigmoid(detect_conf).squeeze(-1)
    detect_2d[detect_conf < detect_conf_thr] = 0

    pred_3d = smpl_decoder(betas, pose, trans, output_format="joints_face")
    track_2d = helper.project_to_2d(K, pred_3d)
    ids = ids.squeeze(-1).long()

    track_2d, id_lookup = rearrange_preds(track_2d, ids)
    smpl_params = {
        "betas": rearrange_preds(betas, ids)[0],
        "pose": rearrange_preds(pose, ids)[0],
        "trans": rearrange_preds(trans, ids)[0],
    }

    valid_tracks = (smpl_params["betas"] != 0).any(-1)
    track_2d[~valid_tracks] = 0

    p0 = track_2d.clone() / 1024
    c0 = (p0 != 0).any(-1).float().clone()
    p1 = detect_2d.clone() / 1024
    c1 = (p1 != 0).any(-1).float().clone()

    # Compare tracks with detections
    ious = helper.normalized_weighted_score(p0, c0, p1, c1)

    if return_ious:
        return ious

    track_ref = {}

    iou_sums = ious.max(-1)[0].sum(0).int()
    for i in range(ious.shape[1]):
        iou_sums = ious.max(-1)[0].sum(0).int()
        best_matched = iou_sums.argmax().item()
        max_iou, max_idxs = ious[:, best_matched].max(-1)
        matched_idxs = (max_iou > min_match_score).nonzero()

        if len(matched_idxs) < min_matched_frames:
            break

        i0, i1 = matched_idxs.min().item(), matched_idxs.max().item() + 1
        track_ref[id_lookup[best_matched]] = [i0, i1]

        idx_range = torch.arange(i0, i1)
        ious[idx_range, :, max_idxs[i0:i1]] = 0
        ious[idx_range, best_matched] = 0

    return track_ref


def bboxes_from_smpl(
    smpl_decoder,
    smpl_params,
    res,
    K,
    pad_x=0.05,
    pad_y=0.05,
    subsample_rate=20,
):
    """Use SMPL mesh to more precisely define bounding box boundary.

    Output bounding box format is x1, y1, x2, y2.
    """
    vertices = smpl_decoder(
        **smpl_params, output_format="mesh", subsample_rate=subsample_rate
    )
    vertices_2d = helper.project_to_2d(K, vertices)
    bboxes = helper.points_to_bbox2d(vertices_2d, pad_dims=[pad_x, pad_y])

    # Clamp bounding box to image boundaries
    image_height, image_width = res
    bboxes[..., 0, :].clamp_min_(0.01)
    bboxes[..., 1, 0].clamp_max_(image_width - 1)
    bboxes[..., 1, 1].clamp_max_(image_height - 1)

    return bboxes


def convert_to_mot(
    track_ids,
    frame_idxs,
    bboxes,
    valid_frames=None,
    frame_idx_ref=None,
    include_dict=False,
):
    mot_txt = ""
    mot_dict = {}

    if isinstance(track_ids, torch.Tensor):
        track_ids = track_ids.long().numpy()
    if isinstance(frame_idxs, torch.Tensor):
        frame_idxs = frame_idxs.long().numpy()
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.float().numpy()

    for track_id, frame_idx, bbox in zip(track_ids, frame_idxs, bboxes):
        img_frame_idx = frame_idx
        if frame_idx_ref is not None:
            img_frame_idx = frame_idx_ref[frame_idx]

        if valid_frames is None or img_frame_idx in valid_frames:
            bbox = list(bbox.flatten())

            if include_dict:
                if track_id not in mot_dict:
                    mot_dict[track_id] = {}
                mot_dict[track_id][img_frame_idx] = bbox

            # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
            str_bbox = [d for d in bbox]
            str_bbox[2] = str_bbox[2] - str_bbox[0]
            str_bbox[3] = str_bbox[3] - str_bbox[1]
            str_bbox = ",".join([f"{d + 1:.2f}" for d in str_bbox])
            mot_txt += f"{img_frame_idx + 1},{track_id},{str_bbox},1,0,0,0\n"

    if include_dict:
        return mot_txt, mot_dict
    else:
        return mot_txt
