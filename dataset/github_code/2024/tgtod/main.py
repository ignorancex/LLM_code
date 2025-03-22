import torch
import torch.nn.functional as F
import numpy as np
from pygod.utils import validate_device
from pygod.metric import eval_roc_auc, eval_average_precision, eval_recall_at_k
from torch_geometric.loader import ClusterData, ClusterLoader, DataLoader
from torch_geometric import seed_everything

from utils import arg_parser, load_data
from tgtod import TGTOD


def main():
    args = arg_parser()
    device = validate_device(args.device)
    dataset = load_data(args.dataset, args.station)

    out_channels = dataset.num_classes if dataset.num_classes > 2 else 1

    data = dataset[0]

    x = data.x
    x = (x - x.mean(0)) / x.std(0)
    data.x = x

    data.y = data.y.float()
    pos_weight = (data.y[data.train_mask] == 0).sum() / (data.y[data.train_mask] == 1).sum()

    num_nodes = data.x.shape[0]
    data.n_id = torch.arange(num_nodes, dtype=torch.long)

    data.edge_time -= data.edge_time.min()
    data.edge_time = data.edge_time // args.timeslot
    time_len = data.edge_time.max() + 1

    if args.num_parts > 1:
        cluster_data = ClusterData(data, num_parts=args.num_parts)
        dataloader = ClusterLoader(cluster_data, batch_size=1, shuffle=True, num_workers=4)
    else:
        dataloader = DataLoader([data])

    model = TGTOD(in_channels=data.x.size(-1),
                  hidden_channels=args.hid_dim,
                  out_channels=out_channels,
                  time_len=time_len,
                  num_parts=args.num_parts,
                  trans_dropout=args.dropout,
                  gnn_dropout=args.dropout,
                  use_cformer=args.use_cformer,
                  graph_weight=args.graph_weight,
                  station=args.station).to(device)

    print(f'Model {args.model} initialized')

    res_auc, res_apr, res_rec = [], [], []
    for run in range(args.runs):
        import gc
        gc.collect()
        print('Number of parameters: ', sum(p.numel() for p in model.parameters()))

        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
        max_valid_apr, best_auc, best_apr, best_rec, patience_cnt = 0., 0., 0., 0., 0
        for epoch in range(args.epochs):
            score = torch.zeros(num_nodes)
            for i, batch in enumerate(dataloader):
                x = batch.x.to(device)
                edge_index = [batch.edge_index[:, batch.edge_time == t].to(device) for t in range(time_len)]
                y = batch.y.to(device)
                train_idx = batch.train_mask.to(device)

                out = model(x, edge_index, i).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(out[train_idx],
                                                          y[train_idx],
                                                          pos_weight=pos_weight)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                score[batch.n_id] = out.detach().cpu()

            train_loss = F.binary_cross_entropy_with_logits(score[data.train_mask],
                                                            data.y[data.train_mask]).item()

            train_auc = eval_roc_auc(data.y[data.train_mask], score[data.train_mask])
            val_auc = eval_roc_auc(data.y[data.val_mask], score[data.val_mask])
            test_auc = eval_roc_auc(data.y[data.test_mask], score[data.test_mask])

            train_apr = eval_average_precision(data.y[data.train_mask], score[data.train_mask])
            val_apr = eval_average_precision(data.y[data.val_mask], score[data.val_mask])
            test_apr = eval_average_precision(data.y[data.test_mask], score[data.test_mask])

            train_rec = eval_recall_at_k(data.y[data.train_mask].long(), score[data.train_mask])
            val_rec = eval_recall_at_k(data.y[data.val_mask].long(), score[data.val_mask])
            test_rec = eval_recall_at_k(data.y[data.test_mask].long(), score[data.test_mask])

            if val_apr > max_valid_apr:
                max_valid_apr = val_apr
                patience_cnt = 0
                best_auc = test_auc
                best_apr = test_apr
                best_rec = test_rec
            else:
                patience_cnt += 1
                if patience_cnt > args.patience:
                    break

            if epoch % args.log_steps == 0:
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {train_loss:.4f}, '
                      f'| Train: AUC: {100 * train_auc:.1f}%, '
                      f'AP: {100 * train_apr:.1f}%, '
                      f'Rec: {100 * train_rec:.1f}%, '
                      f'| Valid: AUC: {100 * val_auc:.1f}% '
                      f'AP: {100 * val_apr:.1f}% '
                      f'Rec: {100 * val_rec:.1f}% '
                      f'| Test: AUC: {100 * test_auc:.1f}% '
                      f'AP: {100 * test_apr:.1f}% '
                      f'Rec: {100 * test_rec:.1f}%')

        res_auc.append(best_auc)
        res_apr.append(best_apr)
        res_rec.append(best_rec)

        print(f'AUC {100 * np.mean(res_auc):.1f}±{100 * np.std(res_auc):.1f} ({100 * np.max(res_auc):.1f}) '
            f'AP {100 * np.mean(res_apr):.1f}±{100 * np.std(res_apr):.1f} ({100 * np.max(res_apr):.1f}) '
            f'Rec {100 * np.mean(res_rec):.1f}±{100 * np.std(res_rec):.1f} ({100 * np.max(res_rec):.1f})')


if __name__ == "__main__":
    main()
    seed_everything(0)
