import datetime
import time

import presets
import torch
import torch.utils.data
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn
import utils
from coco_utils import get_coco, get_coco_api_from_dataset
from coco_eval import CocoEvaluator
from engine import evaluate


@torch.inference_mode()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    coco_evaluator = CocoEvaluator(coco, ["bbox"])

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)
        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)
        breakpoint()
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    map_scores = coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return map_scores


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Detection Training", add_help=add_help)

    parser.add_argument("--data-path", default="../../../data/coco/", type=str, help="dataset path")
    parser.add_argument("--model", default="retinanet_resnet50_fpn", type=str, help="model name")
    parser.add_argument("-b", "--batch-size", default=1, type=int, help="images per gpu")
    parser.add_argument("--num_samples", default=None, type=int, help="number of val samples")
    parser.add_argument("-j", "--workers", default=4, type=int, help="number of data loading workers (default: 4)")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    return parser


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    print(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("\nLoading data")
    dataset_test = get_coco(
        args.data_path, 
        image_set="val", 
        transforms=presets.DetectionPresetEval(), 
        num_samples=args.num_samples
        )

    print("\nCreating data loaders")
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, 
        batch_size=args.batch_size, 
        sampler=test_sampler, 
        num_workers=args.workers, 
        collate_fn=utils.collate_fn
    )

    print("\nCreating model")    
    model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
    
    model.to(device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        
    print("\n\nStart Evaluation\n\n")
    start_time = time.time()
    #torch.backends.cudnn.deterministic = True
    map_scores = evaluate(model, data_loader_test, device=device)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"\n\nEvaluation time {total_time_str}\n\n")
    print(f'\n\nmAP: {map_scores[0]:.3f}\n\n\n')

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
