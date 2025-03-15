import os
import argparse

from eunet.utils.visualization import MeshViewer
from eunet.datasets.utils import readPKL


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize output')
    parser.add_argument('--work_dir', help='the dir to the output rollout dir')
    parser.add_argument('--seq', type=int, help='the target sequence')
    parser.add_argument('--frame', type=int, help='the target frame')

    args = parser.parse_args()

    return args

def visualize_gt_and_pred(sample, frame, pred_dir, mesh_viewer=None):
    if mesh_viewer is None:
        mesh_viewer = MeshViewer()

    pred_path = os.path.join(pred_dir, sample, f"{frame}".zfill(3) + ".pkl")
    predictions = readPKL(pred_path)
    gt_V, pred_V, F = predictions['gt_vertices'], predictions['vertices'], predictions['faces']
    mesh_viewer.add_mesh(pred_V, F, name='pred')
    mesh_viewer.add_mesh(gt_V, F, name='gt')
    h_V, h_F = predictions['h_vertices'], predictions['h_faces']
    mesh_viewer.add_mesh(h_V, h_F, name='human')
    mesh_viewer.show()
    return

def main():
    args = parse_args()

    # Get dataset handler
    m_viewer = MeshViewer()
    # Give seq, num_frame
    sample = f"{args.seq}".zfill(5)
    frame = args.frame
    
    # Visualize prediction
    visualize_gt_and_pred(
        sample, frame, args.work_dir, mesh_viewer=m_viewer)


if __name__ == '__main__':
    main()