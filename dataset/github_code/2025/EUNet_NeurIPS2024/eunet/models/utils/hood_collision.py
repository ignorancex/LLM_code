import torch
from eunet.utils.mesh3d import face_normals_batched
from eunet.datasets.utils.hood_common import gather
from mmcv.ops import QueryAndGroup


class CollisionPreprocessor:
    """
    Resolves garment-body collisions.
    """

    def __init__(self, push_eps=2e-3):
        self.push_eps = push_eps
        self.grouper = QueryAndGroup(
            None,
            1,
            min_radius=0.0,
            use_xyz=False,
            normalize_xyz=False,
            return_grouped_xyz=True,
            return_grouped_idx=False,
            return_unique_cnt=False,)

    def calc_direction(self, cloth_pos, obstacle_pos, obstacle_faces):
        pred_pos = cloth_pos
        device = pred_pos.device

        obstacle_face_pos = gather(obstacle_pos, obstacle_faces, 1, 2, 2).mean(dim=2)
        obstacle_face_normals = face_normals_batched(obstacle_pos, obstacle_faces)
        grouped_results = self.grouper(obstacle_face_pos.contiguous(), pred_pos.contiguous(), obstacle_face_normals.transpose(-1, -2))
        ## Calculate distance
        grouped_normals, grouped_xyz = grouped_results
        grouped_diff = pred_pos.transpose(1, 2).unsqueeze(-1) - grouped_xyz  # relative offsets
        grouped_normals = grouped_normals.permute(0, 2, 3, 1)
        grouped_diff = grouped_diff.permute(0, 2, 3, 1)
        distance = torch.sum(grouped_diff * grouped_normals, dim=-1, keepdim=True).squeeze(-2)

        interpenetration = torch.minimum(distance - self.push_eps, torch.FloatTensor([0]).to(device))
        direction_upd = interpenetration * grouped_normals.squeeze(-2)

        return direction_upd

    def solve(self, state, h_state, h_faces, state_dim, faces=None):
        B = len(state)
        for i in range(B):
            cloth_pos = state[i][None, :, :3]
            obstacle_pos = h_state[i][None, :, state_dim:state_dim+3] # For current pos
            obstacle_faces = h_faces[i][None, :]
            assert state[i].shape[-1] == 2 * state_dim
            cloth_prev_pos = state[i][None, :, state_dim:state_dim+3]
            pos_shift = self.calc_direction(cloth_pos, obstacle_pos, obstacle_faces)
            prev_pos_shift = self.calc_direction(cloth_prev_pos, obstacle_pos, obstacle_faces)

            new_pos = cloth_pos - pos_shift
            new_prev_pos = cloth_prev_pos - prev_pos_shift
            state[i] = torch.cat([new_pos[0], new_pos[0]-new_prev_pos[0], torch.zeros_like(state[i][:, 6:state_dim]).to(cloth_pos), new_prev_pos[0], state[i][:, state_dim+3:]], dim=-1)

        return state
