import numpy as np
import torch
import multiprocessing as mp
from PIL import Image
from skimage import io
from torch.utils.data import Dataset

from utils import (
    build_next_level,
    decode,
    density_estimation,
    fast_knns2spmat,
    l2norm,
    row_normalize,
    sparse_mx_to_indices_values,
    mark_same_camera_nbrs,
    build_knn_per_camera
)

import dgl

def worker(features, labels, xws, yws, coo2meter, cam_ids, k, levels, device):
    dataset = LanderDataset(
        features=features,
        labels=labels,
        xws=xws,
        yws=yws,
        coo2meter=coo2meter,
        cam_ids=cam_ids,
        k=k,
        levels=levels
    )
    return [g.to(device) for g in dataset.gs]

def prepare_dataset_graphs_mp(sequence,
                              k,
                              levels,
                              device,
                              num_workers):
    process_inputs = []
    for scene in sequence:
        process_inputs.append((
            scene['node_embeds'],
            scene['node_labels'],
            scene['xws'],
            scene['yws'],
            scene['coo2meter'],
            scene['cam_ids'],
            k,
            levels,
            device
        ))

    with mp.Pool(num_workers) as pool:
        gss = pool.starmap(worker, process_inputs)

    return gss

class GraphDataset(Dataset):
    def __init__(self, scene_seq, gss):
        self.scene_seq = scene_seq
        self.gss = gss

    def __len__(self):
        return len(self.scene_seq)
    
    def __getitem__(self, index):
        scene = self.scene_seq[index]
        scene['graphs'] = self.gss[index]
        return scene

class SceneDataset(Dataset):
    def __init__(
            self,
            sequence,
            coo2meter,
            feature_model,
            device,
            transform=None
    ):
        self.coo2meter = coo2meter
        self.sequence = sequence
        self.transform = transform
        self.feature_model = feature_model
        self.device = device
    
    def __getitem__(self, index):
        scene = self.sequence[index]
        crop_paths = scene['bboxes_paths']
        embeds = []

        img_batch = []
        for path in crop_paths:
            crop_image = io.imread(path)
            crop_image = Image.fromarray(crop_image)
            if self.transform:
                crop_image = self.transform(crop_image)

            img_batch.append(crop_image)
        img_batch = torch.stack(img_batch, dim=0)

        # Extract features
        with torch.no_grad():
            _, embeds = self.feature_model(img_batch.to(self.device))
            embeds = embeds.cpu().numpy()

        # Embed box world coordinates
        xws = scene['xws'][:, None] / self.coo2meter
        yws = scene['yws'][:, None] / self.coo2meter

        node_embeds = embeds
        sample = {
            'node_labels': scene['node_labels'],
            'node_embeds': node_embeds,
            'xws': xws,
            'yws': yws,
            'cam_ids': scene['cam_ids'],
            'coo2meter': self.coo2meter
        }
        return sample
    
    def __len__(self):
        return len(self.sequence)

class LanderDataset(object):
    def __init__(
        self,
        features,# (669560, 128)
        labels,# (669560,)
        xws,
        yws,
        coo2meter,
        cam_ids,
        cluster_features=None,
        k=1,
        levels=1
    ):
        self.k = k
        self.coo2meter = coo2meter
        self.gs = []
        self.nbrs = []
        self.sims = []
        self.levels = levels
        global_xws = xws.copy()
        global_yws = yws.copy()

        # Initialize features and labels
        features = l2norm(features.astype("float32"))
        self.k = min(features.shape[0], k)

        global_features = features.copy()
        if cluster_features is None:
            cluster_features = features
        global_num_nodes = features.shape[0]
        global_edges = ([], [])
        global_peaks = np.array([], dtype=np.int_)
        ids = np.arange(global_num_nodes)
        cam_ids_oh = np.zeros((cam_ids.shape[0], 4)).astype(np.int32)
        cam_ids_oh[np.arange(cam_ids.shape[0]), cam_ids] = 1
        global_cam_ids = cam_ids.copy()
        peak_cam_ids = cam_ids

        # Recursive graph construction
        for lvl in range(self.levels):
            knns = build_knn_per_camera(features, peak_cam_ids, xws, yws, self.k)
            knns = mark_same_camera_nbrs(knns, cam_ids_oh)

            nbrs, sims = knns[:, 0, :].astype(np.int32), knns[:, 1, :]

            self.nbrs.append(nbrs)
            self.sims.append(sims)
            density = density_estimation(sims, nbrs, labels)

            g = LanderDataset.build_graph(
                features, cluster_features, labels, xws, yws, peak_cam_ids, density, knns
            )

            if g.num_edges() == 0:
                break

            self.gs.append(g)

            if lvl >= self.levels - 1:
                break

            # Decode peak nodes
            (
                new_pred_labels,
                peaks,
                global_edges,
                global_pred_labels,
                global_peaks,
            ) = decode(
                g,
                -np.inf,
                "sim",
                True,
                ids,
                global_edges,
                global_num_nodes,
                global_peaks,
            )
            ids = ids[peaks]
            features, labels, peak_cam_ids, cluster_features, xws, yws, cam_ids_oh = build_next_level(
                features,
                labels,
                peaks,
                peak_cam_ids,
                global_features,
                global_pred_labels,
                global_peaks,
                global_cam_ids,
                global_xws,
                global_yws,
            )

            # If all peaks have same camera id, break
            if len(np.unique(peak_cam_ids)) == 1:
                break

    @staticmethod
    def build_graph(features, cluster_features, labels, xws, yws, cam_ids, density, knns):
        adj = fast_knns2spmat(knns)
        adj, adj_row_sum = row_normalize(adj)
        indices, values, shape = sparse_mx_to_indices_values(adj)
        g = dgl.graph((indices[1], indices[0]), num_nodes=len(knns))
        g.ndata["features"] = torch.FloatTensor(features)
        g.ndata["cluster_features"] = torch.FloatTensor(cluster_features)
        g.ndata["labels"] = torch.LongTensor(labels)
        g.ndata["density"] = torch.FloatTensor(density)
        g.ndata["xws"] = torch.FloatTensor(xws)
        g.ndata["yws"] = torch.FloatTensor(yws)
        g.ndata["cam_ids"] = torch.LongTensor(cam_ids)
        g.edata["affine"] = torch.FloatTensor(values)
        # A Bipartite from DGL sampler will not store global eid, so we explicitly save it here
        g.edata["global_eid"] = g.edges(form="eid")
        g.ndata["norm"] = torch.FloatTensor(adj_row_sum)
        g.apply_edges(
            lambda edges: {
                "raw_affine": edges.data["affine"] / edges.dst["norm"]
            }
        )
        g.apply_edges(
            lambda edges: {
                "labels_conn": (
                    edges.src["labels"] == edges.dst["labels"]
                ).long()
            }
        )

        return g
