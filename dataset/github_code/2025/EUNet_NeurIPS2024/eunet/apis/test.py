import os.path as osp
import pickle
import shutil
import tempfile
import time
import numpy as np

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info
from mmcv.parallel import DataContainer
from eunet.datasets.utils import writePKL, to_numpy_detach, writeJSON
from copy import deepcopy
from collections import defaultdict

def single_gpu_rollout(model, data_loader, show=False, out_dir=None, **show_kwargs):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    rollout_pointer = None
    prev_pred = None
    history_inputs = None
    for i, data in enumerate(data_loader):
        states_dim = data['gt_label'][0]['vertices'].data[0][0].shape[1]
        assert len(data['inputs']['dynamic']) == 1

        seq_num = data['meta']['sequence']
        if rollout_pointer == seq_num:
            human_states = np.array(to_numpy_detach(data['inputs']['dynamic'][0]['h_state'].data[0][0]))
            assert history_inputs is not None
            current_state_inputs = history_inputs.astype(np.float32)
            data = dataset.prepare_rollout(current_garment=current_state_inputs, current_human=human_states, batch_data=data)
        else:
            print(f"Processing sequence: {seq_num}")
            # Begin a new rollout
            rollout_pointer = seq_num
            cuda_history_inputs = data['inputs']['dynamic'][0].get('state', None)
            assert cuda_history_inputs is not None
            history_inputs = np.array(to_numpy_detach(cuda_history_inputs.data[0][0]))

        with torch.no_grad():
            result = model(return_loss=False, **data)
        
        assert isinstance(result, dict)
        assert len(data['inputs']['static']['indices'].data[0]) == 1

        results.append({
            'rollout_idx': to_numpy_detach(rollout_pointer)[0],
            'acc': result['acc'],
            'indices': deepcopy(to_numpy_detach(data['inputs']['static']['indices'].data[0][0])),
            'indices_type': deepcopy(to_numpy_detach(data['inputs']['static']['indices_type'].data[0][0]))
            })

        prev_pred = result['pred']
        assert len(prev_pred) == 1
        prev_pred = np.array(prev_pred[0])
        prev_pred = prev_pred[:, :states_dim]
        assert history_inputs is not None
        history_inputs = np.concatenate([prev_pred, history_inputs[:, :-states_dim]], axis=-1)

        if show or out_dir:
            pred = result['pred']
            meta = data['meta']
            for i in range(len(pred)):
                seq_num = f"{meta['sequence'][i].item()}".zfill(5)
                frame_idx = f"{meta['frame'][i].item()}".zfill(3)
                V = np.array(pred[i])[:, :3]

                if out_dir is not None:
                    seq_dir = osp.abspath(osp.join(out_dir, seq_num))
                    mmcv.mkdir_or_exist(seq_dir)
                    faces = np.array(to_numpy_detach(data['inputs']['static']['faces'].data[0][0]))
                    h_state = np.array(to_numpy_detach(data['inputs']['dynamic'][0]['h_state'].data[0][0][:, :3]))
                    h_faces = np.array(to_numpy_detach(data['inputs']['static']['h_faces'].data[0][0]))
                    garment_gt_data = np.array(to_numpy_detach(data['gt_label'][0]['vertices'].data[0][0][:, :3]))
                    writePKL(osp.join(seq_dir, f"{frame_idx}.pkl"), dict(gt_vertices=garment_gt_data, vertices=V, faces=faces, h_vertices=h_state, h_faces=h_faces))

        assert isinstance(data['gt_label'], list)
        x = data['gt_label'][0]['vertices'].data[0]
        batch_size = len(x) if isinstance(x, list) else x.size(0)
        for _ in range(batch_size):
            prog_bar.update()
    
    return results

def single_gpu_test(model, data_loader, show=False, out_dir=None, **show_kwargs):
    if getattr(data_loader.dataset, 'rollout', False):
        return single_gpu_rollout(model, data_loader, show=show, out_dir=out_dir, **show_kwargs)
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        seq_num = data['meta']['sequence']

        with torch.no_grad():
            result = model(return_loss=False, **data)
        
        assert isinstance(result, dict)
        if isinstance(result['acc'], list):
            for i in result['acc']:
                assert isinstance(i, dict)
                results.append(i)
        else:
            assert isinstance(result['acc'], dict)
            results.append(result['acc'])

        if show or out_dir:
            pred = result['pred']
            meta = data['meta']
            for i in range(len(pred)):
                seq_num = f"{meta['sequence'][i].item()}".zfill(5)
                frame_idx = f"{meta['frame'][i].item()}".zfill(3)
                seq_dir = osp.abspath(osp.join(out_dir, seq_num))
                mmcv.mkdir_or_exist(seq_dir)
                V = pred[i]
                writePKL(osp.join(seq_dir, f"{frame_idx}.pkl"), dict(vertices=V))

        assert isinstance(data['gt_label'], list)
        x = data['gt_label'][0]['vertices'].data[0]
        batch_size = len(x) if isinstance(x, list) else x.size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.
    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        # Check if tmpdir is valid for cpu_collect
        if (not gpu_collect) and (tmpdir is not None and osp.exists(tmpdir)):
            raise OSError((f'The tmpdir {tmpdir} already exists.',
                           ' Since tmpdir will be deleted after testing,',
                           ' please make sure you specify an empty one.'))
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)
    dist.barrier()
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)
            
        assert isinstance(result, dict)
        if isinstance(result['acc'], list):
            # This is for neural sim prediction
            for i in result['acc']:
                assert isinstance(i, dict)
                results.append(i)
        else:
            assert isinstance(result['acc'], dict)
            results.append(result['acc'])

        if rank == 0:
            assert isinstance(data['gt_label'], list)
            x = data['gt_label'][0]['vertices'].data[0]
            batch_size = len(x) if isinstance(x, list) else x.size(0)

            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_result = mmcv.load(part_file)
            part_list.append(part_result)
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_result = pickle.loads(recv[:shape[0]].cpu().numpy().tobytes())
            part_list.append(part_result)
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results