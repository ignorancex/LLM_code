# pylint: disable=not-callable
from pprint import pprint

import numpy as np
import torch
import copy
from manotorch.anchorlayer import AnchorLayer
from manotorch.axislayer import AxisLayer
from manotorch.manolayer import ManoLayer, MANOOutput
from manotorch.utils.quatutils import _normalize_quaternion, _quaternion_to_angle_axis
from lib.utils.transform import aa_to_rotmat, caculate_align_mat
from termcolor import colored
from tqdm import trange
from lib.models.honet.manobranch import ManoAdaptor
from lib.postprocess.geo_loss import FieldLoss, ObjectLoss, HandLoss
from lib.utils.net_utils import recurse_freeze
from lib.viztools.viz_o3d_utils import VizContext
from lib.viztools.utils import ColorMode, get_color_map
"""
MANO transform order (right hand)
              15-14-13-\
                        \
         3 --2 --1-------0
       6 --5 --4--------/
       12 --11 --10----/
         9 --8 --7----/
"""


class GeOptimizer:

    def __init__(
        self,
        device,
        lr=1e-2,
        n_iter=2500,
        verbose=False,
        runtime_viz=False,
        use_fhb_adaptor=False,
        compensate_tsl=False,
        lambda_contact_loss=10.0,
        lambda_repulsion_loss=0.5,
        repulsion_query=0.030,
        repulsion_threshold=0.080,
    ):
        self.device = device
        self.lr = lr
        self.n_iter = n_iter

        # options
        self.verbose = verbose
        self.compensate_tsl = compensate_tsl
        self.runtime_viz = runtime_viz
        if self.runtime_viz is True:
            self.viz_ctx = VizContext(non_block=True)
            self.viz_ctx.init(point_size=15.0)

        # layers and loss utils
        self.mano_layer = ManoLayer(
            rot_mode="quat",
            use_pca=False,
            mano_assets_root="assets/mano_v1_2",
            center_idx=9,
            flat_hand_mean=True,
        ).to(self.device)
        self.use_fhb_adaptor = use_fhb_adaptor
        if use_fhb_adaptor is True:
            self.adaptor = ManoAdaptor(self.mano_layer, "assets/mano/fhb_skel_centeridx9.pkl").to(self.device)
            recurse_freeze(self.adaptor)

        self.anchor_layer = AnchorLayer("assets/anchor").to(self.device)
        self.axis_layer = AxisLayer().to(self.device)

        # opt val dict, const val dict
        self.opt_val = {}
        self.const_val = {}
        self.ctrl_val = {}
        self.coef_val = {
            "lambda_contact_loss": lambda_contact_loss,
            "lambda_repulsion_loss": lambda_repulsion_loss,
            "repulsion_query": repulsion_query,
            "repulsion_threshold": repulsion_threshold,
        }

        # creating slots for optimizer and scheduler
        self.optimizer = None
        self.optimizing = True
        self.scheduler = None

    def reset(self):
        self.opt_val = {}
        self.const_val = {}
        self.ctrl_val = {}
        self.optimizer = None
        self.optimizing = False
        self.scheduler = None
        if self.runtime_viz is True:
            self.viz_ctx.reset()
            self.viz_ctx.deinit()
            self.runtime_viz = False

    def set_opt_val(
        self,
        mode,
        vertex_contact,
        contact_region_id,
        anchor_id,
        anchor_elasti,
        anchor_padding_mask,
        obj_faces,
        hand_shape_gt=None,
        hand_tsl_gt=None,
        hand_pose_gt=None,
        hand_shape_init=None,
        hand_tsl_init=None,
        hand_pose_init=None,
        obj_verts_can=None,
        obj_normals_can=None,
        obj_verts_gt=None,
        obj_normals_gt=None,
        obj_rot_init=None,
        obj_tsl_init=None,
        hand_compensate_root=None,
    ):
        """set the optimization variables.
        `_gt` means that the value is fixed during optimization, 
        `_init` means that the value is the initial value and subject to optimization.
        `_can` means the normals and vers are in object canonical space (without rot & tsl).

        Args:
            mode (str): "hand" | "obj" | "hand_obj"

            ### @NOTE: static val
            vertex_contact (Tensor): (NVERT, ) {0, 1} 
            contact_region_id (Tensor): (NVERT, 1), int
            anchor_id (Tensor): (NVERT, 4): int
            anchor_elasti (Tensor): (NVERT, 4)
            anchor_padding_mask, (Tensor): (NVERT, 4) {0, 1}
            obj_faces (Tensor): (N_FACES, 3)

            ###  @NOTE: dynamic val: hand, depending on mode
            hand_shape_gt (Tensor, optional): (10, ). Defaults to None.
            hand_tsl_gt (Tensor, optional): (3, ). Defaults to None.
            hand_pose_gt (tuple, optional): (LIST[NPROV, ]: int {0..16}, TENSOR[NPROV, 4]). Defaults to None.
            hand_shape_init (Tensor, optional): (10, ). Defaults to None.
            hand_tsl_init (Tensor, optional): (3, ). Defaults to None.
            hand_pose_init (tuple, optional): (LIST[NPROV, ]: int {0..16}, TENSOR[NPROV, 4]). Defaults to None.

            ###  @NOTE: dynamic val: obj, depending on mode
            obj_normals_can (Tensor, optional): (NVERT_O, 3). Defaults to None.
            obj_verts_gt (Tensor, optional): (NVERT_O, 3). Defaults to None.
            obj_normals_gt (Tensor, optional): (NVERT_O, 3). Defaults to None.
            obj_rot_init (Tensor, optional): (3, ). Defaults to None.
            obj_tsl_init (Tensor, optional): (3, ). Defaults to None.
            hand_compensate_root (Tensor, optional): (3, ). Provide when fhb adapt layer is not used, Defaults to None.
        """

        # ====== clear memory
        self.opt_val = {}
        self.const_val = {}
        self.ctrl_val = {
            "mode": mode,
            "optimize_hand_shape": False,
            "optimize_hand_tsl": False,
            "optimize_hand_pose": False,
            "optimize_obj": False,
            "use_fhb_adaptor": self.use_fhb_adaptor,
        }
        self.temp_val = {}

        # region ====== process static values >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        vertex_contact = vertex_contact.long()
        anchor_id = anchor_id.long()
        anchor_padding_mask = anchor_padding_mask.long()

        # boolean index contact_region, anchor_id, anchor_elasti && anchor_padding_mask
        obj_contact_region = contact_region_id[vertex_contact == 1]  # TENSOR[NCONT, ]
        anchor_id = anchor_id[vertex_contact == 1, :]  # TENSOR[NCONT, 4]
        anchor_elasti = anchor_elasti[vertex_contact == 1, :]  # TENSOR[NCONT, 4]
        anchor_padding_mask = anchor_padding_mask[vertex_contact == 1, :]  # TENSOR[NCONT, 4]

        # boolean mask indexing anchor_id, anchor_elasti && obj_vert_id
        indexed_anchor_id = anchor_id[anchor_padding_mask == 1]  # TENSOR[NVALID, ]
        self.const_val["indexed_anchor_id"] = indexed_anchor_id
        self.const_val["indexed_anchor_elasti"] = anchor_elasti[anchor_padding_mask == 1]  # TENSOR[NVALID, ]

        vertex_id = torch.arange(anchor_id.shape[0])[:, None].repeat_interleave(anchor_padding_mask.shape[1],
                                                                                dim=1)  # TENSOR[NCONT, 4]
        self.const_val["indexed_vertex_id"] = vertex_id[anchor_padding_mask == 1]  # TENSOR[NVALID, ]

        tip_anchor_mask = torch.zeros(indexed_anchor_id.shape[0]).bool().to(self.device)
        tip_anchor_list = [2, 3, 4, 9, 10, 11, 15, 16, 17, 22, 23, 24, 29, 30, 31]
        for tip_anchor_id in tip_anchor_list:
            tip_anchor_mask = tip_anchor_mask | (self.const_val["indexed_anchor_id"] == tip_anchor_id)
        self.const_val["indexed_elasti_k"] = torch.where(tip_anchor_mask,
                                                         torch.Tensor([1.0]).to(self.device),
                                                         torch.Tensor([0.1]).to(self.device)).to(self.device)

        # hand faces & edges
        self.const_val["hand_faces"] = self.mano_layer.th_faces
        self.const_val["obj_faces"] = obj_faces
        self.const_val["static_verts"] = self.get_static_hand_verts()
        self.const_val["hand_edges"] = HandLoss.get_edge_idx(self.const_val["hand_faces"])
        self.const_val["static_edge_len"] = HandLoss.get_edge_len(self.const_val["static_verts"],
                                                                  self.const_val["hand_edges"])
        self.const_val["contact_region_id"] = contact_region_id
        # endregion <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # region ====== dynamic val: hand >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # >>> hand_shape
        if mode in ["hand", "hand_obj"]:
            assert hand_shape_gt is None and hand_shape_init is not None, "hand shape init required"
            self.opt_val["hand_shape_var"] = hand_shape_init.detach().clone().requires_grad_(True)
            self.const_val["hand_shape_init"] = hand_shape_init
            self.ctrl_val["optimize_hand_shape"] = True
        else:  # mode == "obj"
            self.const_val["hand_shape_gt"] = hand_shape_gt
            self.ctrl_val["optimize_hand_shape"] = False

        # >>> hand_tsl
        if mode in ["hand", "hand_obj"]:
            assert hand_tsl_gt is None and hand_tsl_init is not None, "hand tsl init required"
            self.opt_val["hand_tsl_var"] = hand_tsl_init.detach().clone().requires_grad_(True)
            self.const_val["hand_tsl_init"] = hand_tsl_init
            self.ctrl_val["optimize_hand_tsl"] = True
        else:  # mode == "obj"
            self.const_val["hand_tsl_gt"] = hand_tsl_gt
            self.ctrl_val["optimize_hand_tsl"] = False

        # >>> hand pose
        if mode in ["hand", "hand_obj"]:
            gt_pose_idx, gt_pose_val = hand_pose_gt
            init_pose_idx, init_pose_val = hand_pose_init
            if len(set(gt_pose_idx).intersection(set(init_pose_idx))) > 0:
                raise RuntimeError("repeat hand_pose gt & init provided")
            if set(gt_pose_idx).union(set(init_pose_idx)) != set(range(16)):
                raise RuntimeError("hand_pose: not enough gt & init")
            self.const_val["hand_pose_gt_idx"] = gt_pose_idx
            self.const_val["hand_pose_gt_val"] = gt_pose_val
            self.const_val["hand_pose_var_idx"] = init_pose_idx
            self.opt_val["hand_pose_var_val"] = init_pose_val.detach().clone().requires_grad_(True)
            self.const_val["hand_pose_init_val"] = init_pose_val
            self.ctrl_val["optimize_hand_pose"] = True
        else:  # mode == "obj"
            gt_pose_idx, gt_pose_val = hand_pose_gt
            assert set(gt_pose_idx) == set(range(16)), "hand_pose: not enough gt"
            self.const_val["hand_pose_gt_idx"] = gt_pose_idx
            self.const_val["hand_pose_gt_val"] = gt_pose_val
            self.const_val["hand_pose_var_idx"] = []
            self.opt_val["hand_pose_var_val"] = torch.zeros((0, 4), dtype=torch.float, device=self.device)
            self.ctrl_val["optimize_hand_pose"] = False
        # endregion <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # region ====== dynamic val: obj >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        if mode in ["obj", "hand_obj"]:
            assert obj_rot_init is not None and obj_tsl_init is not None, "obj rot & tsl init required"
            self.const_val["obj_verts_can"] = obj_verts_can[vertex_contact == 1, :]
            self.const_val["obj_normals_can"] = obj_normals_can[vertex_contact == 1, :]
            self.const_val["full_obj_verts"] = obj_verts_can
            self.const_val["full_obj_normals"] = obj_normals_can
            self.opt_val["obj_rot_var"] = obj_rot_init.detach().clone().requires_grad_(True)
            self.opt_val["obj_tsl_var"] = obj_tsl_init.detach().clone().requires_grad_(True)
            self.const_val["obj_rot_init"] = obj_rot_init
            self.const_val["obj_tsl_init"] = obj_tsl_init
            self.ctrl_val["optimize_obj"] = True
        else:  # mode == "hand"
            self.const_val["obj_verts_gt"] = obj_verts_gt[vertex_contact == 1, :]
            self.const_val["obj_normals_gt"] = obj_normals_gt[vertex_contact == 1, :]
            self.const_val["full_obj_verts"] = obj_verts_gt
            self.const_val["full_obj_normals"] = obj_normals_gt
            self.ctrl_val["optimize_obj"] = False
        # endregion <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # compensate tsl (when fhb adapt layer is not used)
        if self.compensate_tsl and (hand_tsl_gt is not None or hand_tsl_init is not None):
            if hand_compensate_root is None:
                raise RuntimeError("if need to compensate hand root tsl, correct root pos is requried")
            _, curr_joints, _ = self.recover_hand()
            compensate_offset = hand_compensate_root - curr_joints[0, ...]
            if not self.ctrl_val["optimize_hand_tsl"]:
                self.const_val["hand_tsl_gt"] = hand_tsl_gt + compensate_offset
            else:
                self.opt_val["hand_tsl_var"] = (hand_tsl_init + compensate_offset).detach().clone().requires_grad_(True)
                self.const_val["hand_tsl_init"] = hand_tsl_init + compensate_offset

        # region ====== construct optimizer & scheduler >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        if (self.ctrl_val["optimize_hand_shape"] or self.ctrl_val["optimize_hand_tsl"] or
                self.ctrl_val["optimize_hand_pose"] or self.ctrl_val["optimize_obj"]):
            # dispatch lr to different param
            param = []
            if self.ctrl_val["optimize_hand_shape"]:
                param.append({"params": [self.opt_val["hand_shape_var"]]})
            if self.ctrl_val["optimize_hand_tsl"]:
                param.append({"params": [self.opt_val["hand_tsl_var"]], "lr": 0.1 * self.lr})
            if self.ctrl_val["optimize_hand_pose"]:
                param.append({"params": [self.opt_val["hand_pose_var_val"]]})
            if self.ctrl_val["optimize_obj"]:
                param.append({"params": [self.opt_val["obj_rot_var"]]})
                param.append({"params": [self.opt_val["obj_tsl_var"]], "lr": 0.1 * self.lr})

            self.optimizer = torch.optim.Adam(param, lr=self.lr)
            self.optimizing = True
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                        min_lr=1e-5,
                                                                        mode="min",
                                                                        factor=0.5,
                                                                        patience=20,
                                                                        verbose=False)
        else:
            self.optimizing = False
        # endregion <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # ====== verbose
        if self.verbose:
            print("Optimizing: ", self.optimizing)
            pprint(self.ctrl_val)
            pprint(list(self.opt_val.keys()))
            pprint(list(self.const_val.keys()))
            pprint(self.coef_val)

        return

    @staticmethod
    def get_var_pose_idx(sel_pose_idx):
        # gt has 16 pose
        all_pose_idx = set(range(16))
        sel_pose_idx_set = set(sel_pose_idx)
        var_pose_idx = all_pose_idx.difference(sel_pose_idx_set)
        return list(var_pose_idx)

    def get_static_hand_verts(self):
        init_val_pose = np.array([[1.0, 0.0, 0.0, 0.0]] * 16).astype(np.float32)
        vec_pose = torch.tensor(init_val_pose).reshape(-1).unsqueeze(0).float().to(self.device)
        vec_shape = torch.zeros(1, 10).float().to(self.device)
        mano_output: MANOOutput = self.mano_layer(vec_pose, vec_shape)
        v = mano_output.verts.squeeze(0)
        return v

    @staticmethod
    def assemble_pose_vec(gt_idx, gt_pose, var_idx, var_pose):
        idx_tensor = torch.cat((torch.Tensor(gt_idx).long(), torch.Tensor(var_idx).long()))
        pose_tensor = torch.cat((gt_pose, var_pose), dim=0)
        pose_tensor = pose_tensor[torch.argsort(idx_tensor)]
        return pose_tensor

    @staticmethod
    def transf_vectors(vectors, tsl, rot):
        """
        vectors: [K, 3], tsl: [3, ], rot: [3, ]
        return: [K, 3]
        """
        rot_matrix = aa_to_rotmat(rot.unsqueeze(0)).squeeze(0).reshape(3, 3)  # (3, 3)
        vec = (rot_matrix @ vectors.T).T
        vec = vec + tsl
        return vec

    def loss_fn(self, opt_val, const_val, ctrl_val, coef_val):
        var_hand_pose_assembled = self.assemble_pose_vec(
            const_val["hand_pose_gt_idx"],
            const_val["hand_pose_gt_val"],
            const_val["hand_pose_var_idx"],
            opt_val["hand_pose_var_val"],
        )

        # dispatch hand var
        vec_pose = var_hand_pose_assembled.unsqueeze(0)
        if ctrl_val["optimize_hand_shape"]:
            vec_shape = opt_val["hand_shape_var"].unsqueeze(0)
        else:
            vec_shape = const_val["hand_shape_gt"].unsqueeze(0)
        if ctrl_val["optimize_hand_tsl"]:
            vec_tsl = opt_val["hand_tsl_var"].unsqueeze(0)
        else:
            vec_tsl = const_val["hand_tsl_gt"].unsqueeze(0)

        # rebuild hand
        mano_output: MANOOutput = self.mano_layer(vec_pose, vec_shape)
        rebuild_verts = mano_output.verts
        rebuild_joints = mano_output.joints
        rebuild_transf = mano_output.transforms_abs
        rebuild_full_pose = mano_output.full_poses

        # skel adaption
        if ctrl_val["use_fhb_adaptor"]:
            adapt_joints, _ = self.adaptor(rebuild_verts)
            adapt_joints = adapt_joints.transpose(1, 2)
            rebuild_joints = rebuild_joints - adapt_joints[:, 9].unsqueeze(1)
            rebuild_verts = rebuild_verts - adapt_joints[:, 9].unsqueeze(1)
        rebuild_joints = rebuild_joints + vec_tsl
        rebuild_verts = rebuild_verts + vec_tsl
        rebuild_transf = rebuild_transf + torch.cat([
            torch.cat([torch.zeros(3, 3).to(self.device), vec_tsl.view(3, -1)], dim=1),
            torch.zeros(1, 4).to(self.device),
        ],
                                                    dim=0)
        rebuild_verts_squeezed = rebuild_verts.squeeze(0)

        # rebuild anchor
        rebuild_anchor = self.anchor_layer(rebuild_verts)
        rebuild_anchor = rebuild_anchor.contiguous()  # TENSOR[1, 32, 3]
        rebuild_anchor = rebuild_anchor.squeeze(0)  # TENSOR[32, 3]
        anchor_pos = rebuild_anchor[const_val["indexed_anchor_id"]]  # TENSOR[NVALID, 3]

        # dispatch obj var, depending on whether it is subjected to optimization.
        if ctrl_val["optimize_obj"]:
            obj_verts = self.transf_vectors(
                const_val["obj_verts_can"],
                opt_val["obj_tsl_var"],
                opt_val["obj_rot_var"],
            )
            full_obj_verts = self.transf_vectors(
                const_val["full_obj_verts"],
                opt_val["obj_tsl_var"],
                opt_val["obj_rot_var"],
            )
            full_obj_normals = self.transf_vectors(
                const_val["full_obj_normals"],
                torch.zeros(3, dtype=torch.float, device=self.device),
                opt_val["obj_rot_var"],
            )
        else:
            obj_verts = const_val["obj_verts_gt"]
            full_obj_verts = const_val["full_obj_verts"]
            full_obj_normals = const_val["full_obj_normals"]

        # contact loss
        contact_loss = FieldLoss.contact_loss(
            anchor_pos,
            obj_verts[const_val["indexed_vertex_id"]],
            const_val["indexed_anchor_elasti"],
            const_val["indexed_elasti_k"],
        )
        # repulsion loss
        repulsion_loss = FieldLoss.full_repulsion_loss(
            rebuild_verts_squeezed,
            full_obj_verts,
            full_obj_normals,
            query=coef_val["repulsion_query"],
            threshold=coef_val["repulsion_threshold"],
        )

        if ctrl_val["optimize_hand_pose"]:
            # get hand loss
            quat_norm_loss = HandLoss.pose_quat_norm_loss(var_hand_pose_assembled)
            var_hand_pose_normalized = _normalize_quaternion(var_hand_pose_assembled)
            pose_reg_loss = HandLoss.pose_reg_loss(var_hand_pose_normalized[const_val["hand_pose_var_idx"]],
                                                   const_val["hand_pose_init_val"])

            b_axis, u_axis, l_axis = self.axis_layer(rebuild_joints, rebuild_transf)

            angle_axis = _quaternion_to_angle_axis(var_hand_pose_normalized.reshape((16, 4)))
            angle_axis = angle_axis[1:, :]  # ignore global rot [15, 3]
            axis = angle_axis / torch.norm(angle_axis, dim=1, keepdim=True)
            angle = torch.norm(angle_axis, dim=1, keepdim=False)
            # limit angle
            angle_limit_loss = HandLoss.rotation_angle_loss(angle)

            joint_b_axis_loss = HandLoss.joint_b_axis_loss(b_axis, axis)
            joint_u_axis_loss = HandLoss.joint_u_axis_loss(u_axis, axis)
            joint_l_limit_loss = HandLoss.joint_l_limit_loss(l_axis, axis)

            edge_loss = HandLoss.edge_len_loss(rebuild_verts_squeezed, const_val["hand_edges"],
                                               const_val["static_edge_len"])
        else:
            quat_norm_loss = torch.Tensor([0.0]).to(self.device)
            pose_reg_loss = torch.Tensor([0.0]).to(self.device)
            angle_limit_loss = torch.Tensor([0.0]).to(self.device)
            joint_b_axis_loss = torch.Tensor([0.0]).to(self.device)
            joint_u_axis_loss = torch.Tensor([0.0]).to(self.device)
            joint_l_limit_loss = torch.Tensor([0.0]).to(self.device)
            edge_loss = torch.Tensor([0.0]).to(self.device)
            # pose_reg_loss_to_zero = torch.Tensor([0.0]).to(self.device)

        if ctrl_val["optimize_hand_shape"]:
            shape_reg_loss = HandLoss.shape_reg_loss(opt_val["hand_shape_var"], const_val["hand_shape_init"])
        else:
            shape_reg_loss = torch.Tensor([0.0]).to(self.device)

        if ctrl_val["optimize_hand_tsl"]:
            hand_tsl_loss = HandLoss.hand_tsl_loss(opt_val["hand_tsl_var"], const_val["hand_tsl_init"])
        else:
            hand_tsl_loss = torch.Tensor([0.0]).to(self.device)

        if ctrl_val["optimize_obj"]:
            obj_transf_loss = ObjectLoss.obj_transf_loss(opt_val["obj_tsl_var"], opt_val["obj_rot_var"],
                                                         const_val["obj_tsl_init"], const_val["obj_rot_init"])
        else:
            obj_transf_loss = torch.Tensor([0.0]).to(self.device)

        loss = (
            # ============= HAND ANATOMICAL LOSS
            1.0 * quat_norm_loss + 1.0 * angle_limit_loss + 1.0 * edge_loss + 0.1 * joint_b_axis_loss +
            0.1 * joint_u_axis_loss + 0.1 * joint_l_limit_loss
            # ============= ELAST POTENTIONAL ENERGY
            + coef_val["lambda_contact_loss"] * contact_loss + coef_val["lambda_repulsion_loss"] * repulsion_loss
            # ============= OFFSET LOSS
            + 1.0 * pose_reg_loss + 1.0 * shape_reg_loss + 1.0 * hand_tsl_loss + 1.0 * obj_transf_loss)

        # debug: runtime viz
        if self.runtime_viz:
            cr_for_viz = copy.deepcopy(self.const_val["contact_region_id"])
            contact_color = get_color_map(cr_for_viz, ColorMode.CONTACT_REGION)
            self.viz_ctx.update_by_mesh("obj_mesh_curr",
                                        verts=full_obj_verts,
                                        faces=self.const_val["obj_faces"],
                                        normals=full_obj_normals,
                                        vcolors=contact_color,
                                        update=self.ctrl_val["mode"] != "hand")
            self.viz_ctx.update_by_mesh("hand_mesh_curr",
                                        rebuild_verts.squeeze(0),
                                        faces=self.const_val["hand_faces"],
                                        update=self.ctrl_val["mode"] != "obj")
            if ctrl_val["optimize_hand_pose"]:
                u_axis = u_axis.squeeze(0).detach().cpu().numpy()
                b_axis = b_axis.squeeze(0).detach().cpu().numpy()
                l_axis = l_axis.squeeze(0).detach().cpu().numpy()
                rebuild_transf = rebuild_transf.squeeze(0).detach().cpu().numpy()

                b_color = np.array([255, 0, 0]) / 255.0  # twist dir
                u_color = np.array([0, 255, 0]) / 255.0  # spread dir
                l_color = np.array([0, 0, 255]) / 255.0  # bend dir

                for axi in range(1, 4):  # only show axis on index finger
                    # @NOTE: back axis, twist direction
                    b_rot = rebuild_transf[axi][:3, :3] @ caculate_align_mat(b_axis[axi - 1])  # left-multiply
                    b_tsl = rebuild_transf[axi][:3, 3].T
                    if self.temp_val.get(f"b_rot_{axi}_prev") is not None:
                        # transform previous axis back to its original pose
                        self.viz_ctx.update_by_arrow(f"b_axis_{axi}",
                                                     np.eye(3),
                                                     -self.temp_val[f"b_tsl_{axi}_prev"],
                                                     colors=b_color,
                                                     update=True)
                        self.viz_ctx.update_by_arrow(f"b_axis_{axi}",
                                                     self.temp_val[f"b_rot_{axi}_prev"].T,
                                                     np.zeros_like(b_tsl),
                                                     colors=b_color,
                                                     update=True)
                    self.viz_ctx.update_by_arrow(f"b_axis_{axi}", b_rot, b_tsl, colors=b_color, update=True)
                    self.temp_val[f"b_rot_{axi}_prev"] = b_rot
                    self.temp_val[f"b_tsl_{axi}_prev"] = b_tsl

                    # @NOTE: up axis, spread direction
                    u_rot = rebuild_transf[axi][:3, :3] @ caculate_align_mat(u_axis[axi - 1])  # left-multiply
                    u_tsl = rebuild_transf[axi][:3, 3].T
                    if self.temp_val.get(f"u_rot_{axi}_prev") is not None:
                        # transform previous axis back to its original pose
                        self.viz_ctx.update_by_arrow(f"u_axis_{axi}",
                                                     np.eye(3),
                                                     -self.temp_val[f"u_tsl_{axi}_prev"],
                                                     colors=u_color,
                                                     update=True)
                        self.viz_ctx.update_by_arrow(f"u_axis_{axi}",
                                                     self.temp_val[f"u_rot_{axi}_prev"].T,
                                                     np.zeros_like(u_tsl),
                                                     colors=u_color,
                                                     update=True)
                    self.viz_ctx.update_by_arrow(f"u_axis_{axi}", u_rot, u_tsl, colors=u_color, update=True)
                    self.temp_val[f"u_rot_{axi}_prev"] = u_rot
                    self.temp_val[f"u_tsl_{axi}_prev"] = u_tsl

                    # @NOTE: left axis, bend direction
                    l_rot = rebuild_transf[axi][:3, :3] @ caculate_align_mat(l_axis[axi - 1])
                    l_tsl = rebuild_transf[axi][:3, 3].T
                    if self.temp_val.get(f"l_rot_{axi}_prev") is not None:
                        # transform previous axis back to its original pose
                        self.viz_ctx.update_by_arrow(f"l_axis_{axi}",
                                                     np.eye(3),
                                                     -self.temp_val[f"l_tsl_{axi}_prev"],
                                                     colors=l_color,
                                                     update=True)
                        self.viz_ctx.update_by_arrow(f"l_axis_{axi}",
                                                     self.temp_val[f"l_rot_{axi}_prev"].T,
                                                     np.zeros_like(l_tsl),
                                                     colors=l_color,
                                                     update=True)
                    self.viz_ctx.update_by_arrow(f"l_axis_{axi}", l_rot, l_tsl, colors=l_color, update=True)
                    self.temp_val[f"l_rot_{axi}_prev"] = l_rot
                    self.temp_val[f"l_tsl_{axi}_prev"] = l_tsl

            self.viz_ctx.step()

        return (
            loss,
            {
                "quat_norm_loss": quat_norm_loss.detach().cpu().item(),
                "angle_limit_loss": angle_limit_loss.detach().cpu().item(),
                "edge_loss": edge_loss.detach().cpu().item(),
                "joint_b_axis_loss": joint_b_axis_loss.detach().cpu().item(),
                "joint_u_axis_loss": joint_u_axis_loss.detach().cpu().item(),
                "joint_l_limit_loss": joint_l_limit_loss.detach().cpu().item(),
                "contact_loss": contact_loss.detach().cpu().item(),
                "repulsion_loss": repulsion_loss.detach().cpu().item(),
                "pose_reg_loss": pose_reg_loss.detach().cpu().item(),
                "hand_tsl_loss": hand_tsl_loss.detach().cpu().item(),
                "obj_transf_loss": obj_transf_loss.detach().cpu().item(),
            },
        )

    def optimize(self, progress=False):
        if progress:
            bar = trange(self.n_iter, position=3)
            bar_hand = trange(0, position=2, bar_format="{desc}")
            bar_contact = trange(0, position=1, bar_format="{desc}")
            bar_axis = trange(0, position=0, bar_format="{desc}")
        else:
            bar = range(self.n_iter)

        loss = torch.Tensor([0.0]).to(self.device)
        loss_dict = {}
        for _ in bar:
            if self.optimizing:
                self.optimizer.zero_grad()

            loss, loss_dict = self.loss_fn(self.opt_val, self.const_val, self.ctrl_val, self.coef_val)

            if self.optimizing:
                loss.backward()
                self.optimizer.step()
                self.scheduler.step(loss)

            if progress:
                bar.set_description("TOTAL LOSS {:4e}".format(loss.item()))
                try:
                    bar_hand.set_description(
                        colored("HAND_REGUL_LOSS: ", "yellow") + "QN={:.3e} PR={:.3e} EG={:.3e}".format(
                            loss_dict["quat_norm_loss"],  # QN
                            loss_dict["pose_reg_loss"],  # PR
                            loss_dict["edge_loss"],  # Edge
                        ))
                except:
                    pass
                try:
                    bar_contact.set_description(
                        colored("HO_CONTACT_LOSS: ", "blue") + "Conta={:.3e}, Repul={:.3e}, OT={:.3e}".format(
                            loss_dict["contact_loss"],  # Conta
                            loss_dict["repulsion_loss"],  # Repul
                            loss_dict["obj_transf_loss"],  # OT
                        ))
                except:
                    pass
                try:
                    bar_axis.set_description(
                        colored("ANGLE_LOSS: ", "cyan") + "AL={:.3e} JB={:.3e} JU={:.3e} JL={:.3e}".format(
                            loss_dict["angle_limit_loss"],  # AL
                            loss_dict["joint_b_axis_loss"],  # JB
                            loss_dict["joint_u_axis_loss"],  # JU
                            loss_dict["joint_l_limit_loss"],  # JL
                        ))
                except:
                    pass

        if self.runtime_viz:
            self.viz_ctx.remove_all_geometry()

        self.temp_val = {}
        return loss.item(), loss_dict

    def recover_hand(self, squeeze_out=True):
        vars_hand_pose_assembled = self.assemble_pose_vec(
            self.const_val["hand_pose_gt_idx"],
            self.const_val["hand_pose_gt_val"],
            self.const_val["hand_pose_var_idx"],
            self.opt_val["hand_pose_var_val"],
        ).detach()
        vars_hand_pose_normalized = _normalize_quaternion(vars_hand_pose_assembled)
        vec_pose = vars_hand_pose_normalized.unsqueeze(0)
        if self.ctrl_val["optimize_hand_shape"]:
            vec_shape = self.opt_val["hand_shape_var"].detach().unsqueeze(0)
        else:
            vec_shape = self.const_val["hand_shape_gt"].unsqueeze(0)
        if self.ctrl_val["optimize_hand_tsl"]:
            vec_tsl = self.opt_val["hand_tsl_var"].detach().unsqueeze(0)
        else:
            vec_tsl = self.const_val["hand_tsl_gt"].unsqueeze(0)

        device = vec_pose.device
        # rebuild_verts, rebuild_joints, rebuild_transf, rebuild_full_pose = self.mano_layer(vec_pose, vec_shape)
        mano_output: MANOOutput = self.mano_layer(vec_pose, vec_shape)
        rebuild_verts = mano_output.verts
        rebuild_joints = mano_output.joints
        rebuild_transf = mano_output.transforms_abs
        rebuild_full_pose = mano_output.full_poses

        # skel adaption
        if self.ctrl_val["use_fhb_adaptor"]:
            adapt_joints, _ = self.adaptor(rebuild_verts)
            adapt_joints = adapt_joints.transpose(1, 2)
            rebuild_joints = rebuild_joints - adapt_joints[:, 9].unsqueeze(1)
            rebuild_verts = rebuild_verts - adapt_joints[:, 9].unsqueeze(1)
        rebuild_verts = rebuild_verts + vec_tsl
        rebuild_joints = rebuild_joints + vec_tsl
        rebuild_transf = rebuild_transf + torch.cat(
            [
                torch.cat((torch.zeros((3, 3), device=device), vec_tsl.T), dim=1),
                torch.zeros((1, 4), device=device),
            ],
            dim=0,
        )
        if squeeze_out:
            rebuild_verts, rebuild_joints, rebuild_transf = (
                rebuild_verts.squeeze(0),
                rebuild_joints.squeeze(0),
                rebuild_transf.squeeze(0),
            )
        return rebuild_verts, rebuild_joints, rebuild_transf

    def recover_hand_pose(self):
        vars_hand_pose_assembled = self.assemble_pose_vec(
            self.const_val["hand_pose_gt_idx"],
            self.const_val["hand_pose_gt_val"],
            self.const_val["hand_pose_var_idx"],
            self.opt_val["hand_pose_var_val"],
        ).detach()
        vars_hand_pose_normalized = _normalize_quaternion(vars_hand_pose_assembled)
        return vars_hand_pose_normalized

    def recover_obj(self):
        if self.ctrl_val["optimize_obj"]:
            obj_verts = self.transf_vectors(
                self.const_val["full_obj_verts"],
                self.opt_val["obj_tsl_var"].detach(),
                self.opt_val["obj_rot_var"].detach(),
            )
        else:
            obj_verts = self.const_val["full_obj_verts"]
        return obj_verts

    def recover_obj_rot(self):
        if self.ctrl_val["optimize_obj"]:
            res = self.opt_val["obj_rot_var"].detach().cpu()
            return res
        else:
            raise RuntimeError("not optimizing obj, cannot get obj_rot")

    def recover_obj_tsl(self):
        if "optimize_obj" not in self.ctrl_val:
            print(self.ctrl_val)
        if self.ctrl_val["optimize_obj"]:
            res = self.opt_val["obj_tsl_var"].detach().cpu()
            return res
        else:
            raise RuntimeError("not optimizing obj, cannot get obj_tsl")
