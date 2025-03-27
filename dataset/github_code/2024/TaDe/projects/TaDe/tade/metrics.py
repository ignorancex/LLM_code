import torch
from mmengine.registry import METRICS
from mmengine.evaluator import BaseMetric


@METRICS.register_module()
class BEVMetric(BaseMetric):
    def __init__(
        self,
        prefix=None,
        class_names=[
            "drivable_area",
            "ped_crossing ",
            "walkway      ",
            "carpark_area ",
            "car          ",
            "truck        ",
            "trailer      ",
            "bus          ",
            "cons_vehicle ",
            "bicycle      ",
            "motorcycle   ",
            "pedestrian   ",
            "traffic_cone ",
            "barrier      ",
        ],
        thresholds=[0.5],
    ):
        super().__init__(prefix=prefix)
        self.class_names = class_names
        self.thresholds = thresholds

    def process(self, data_batch, data_samples):
        with torch.no_grad():
            pred = torch.stack([data_samples[0] >= t for t in self.thresholds], dim=-1)
            gt = torch.unsqueeze(data_samples[1], dim=-1)
            mask = torch.unsqueeze(gt[:, -1], dim=1)
            h_cart, w_cart = 200, 200
            x1, z1, x2, z2 = -25, 0, 25, 50
            x = torch.arange(x1, x2, (x2 - x1) / w_cart)[None, :]
            z = torch.arange(z1, z2, (z2 - z1) / h_cart)[:, None]
            m = (torch.sqrt(x**2 + z**2) <= 25)[None, None, :, :, None].to(mask)
            gt = gt[:, :-1]
            i, u = pred & gt, pred | gt
            i, u, pred, gt = i & ~mask, u & ~mask, pred & ~mask, gt & ~mask
            i_n, u_n, pred_n, gt_n = i & m, u & m, pred & m, gt & m
            i_f, u_f, pred_f, gt_f = i & ~m, u & ~m, pred & ~m, gt & ~m
            i, u, pred, gt = i.sum((0, 2, 3)), u.sum((0, 2, 3)), pred.sum((0, 2, 3)), gt.sum((0, 2, 3))
            i_n, u_n, pred_n, gt_n = i_n.sum((0, 2, 3)), u_n.sum((0, 2, 3)), pred_n.sum((0, 2, 3)), gt_n.sum((0, 2, 3))
            i_f, u_f, pred_f, gt_f = i_f.sum((0, 2, 3)), u_f.sum((0, 2, 3)), pred_f.sum((0, 2, 3)), gt_f.sum((0, 2, 3))

            result = dict(
                i_u=torch.stack(
                    [
                        i,
                        u,
                        pred,
                        gt.expand_as(pred),
                        i_n,
                        u_n,
                        pred_n,
                        gt_n.expand_as(pred_n),
                        i_f,
                        u_f,
                        pred_f,
                        gt_f.expand_as(pred_f),
                    ]
                ).cpu()
            )

            if len(data_samples) > 2:
                result["correct"] = torch.sum(data_samples[2] == data_samples[3]).item()
                result["total"] = data_samples[3].numel()

        self.results.append(result)

    def compute_metrics(self, results):
        i_u = torch.zeros_like(results[0]["i_u"])
        correct, total = 0, 0

        for result in results:
            i_u += result["i_u"]
            if "correct" in result and "total" in result:
                correct += result["correct"]
                total += result["total"]

        ious = i_u[0] / torch.clamp(i_u[1], min=1e-7)
        precision = i_u[0] / torch.clamp(i_u[2], min=1e-7)
        recall = i_u[0] / torch.clamp(i_u[3], min=1e-7)
        ious_n = i_u[4] / torch.clamp(i_u[5], min=1e-7)
        precision_n = i_u[4] / torch.clamp(i_u[6], min=1e-7)
        recall_n = i_u[4] / torch.clamp(i_u[7], min=1e-7)
        ious_f = i_u[8] / torch.clamp(i_u[9], min=1e-7)
        precision_f = i_u[8] / torch.clamp(i_u[10], min=1e-7)
        recall_f = i_u[8] / torch.clamp(i_u[11], min=1e-7)

        metrics = dict()
        miou = 0.0
        for t, threshold in enumerate(self.thresholds):
            for n, name in enumerate(self.class_names):
                metrics[f"\n{name}/iou@{threshold:.2f}"] = ious[n, t].item()
                metrics[f"    {name}/pre@{threshold:.2f}"] = precision[n, t].item()
                metrics[f"    {name}/rec@{threshold:.2f}"] = recall[n, t].item()
                metrics[f"\n{name}/i_n@{threshold:.2f}"] = ious_n[n, t].item()
                metrics[f"    {name}/p_n@{threshold:.2f}"] = precision_n[n, t].item()
                metrics[f"    {name}/r_n@{threshold:.2f}"] = recall_n[n, t].item()
                metrics[f"\n{name}/i_f@{threshold:.2f}"] = ious_f[n, t].item()
                metrics[f"    {name}/p_f@{threshold:.2f}"] = precision_f[n, t].item()
                metrics[f"    {name}/r_f@{threshold:.2f}"] = recall_f[n, t].item()
            metrics[f"\nmean/iou@{threshold:.2f}"] = ious.mean(dim=0)[t].item()
            miou = max(ious.mean(dim=0)[t].item(), miou)

        if total > 0:
            metrics["index/accuracy"] = correct / total
        metrics["mIOU"] = miou
        return metrics
