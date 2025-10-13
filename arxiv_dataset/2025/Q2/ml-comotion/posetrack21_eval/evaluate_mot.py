if __name__ == "__main__":
    from datasets.pt_warper import PTWrapper
    import motmetrics as mm
else:
    from .datasets.pt_warper import PTWrapper
    from . import motmetrics as mm

import numpy as np
mm.lap.default_solver = 'lap'
from shapely.geometry import box, Polygon, MultiPolygon
import argparse
import os
from tqdm import tqdm

def get_mot_accum(results, seq, ignore_iou_thres=0.1, use_ignore_regions=False):
    def get_frame_idx(impath):
        return int(impath.split("/")[-1].replace(".jpg", ""))

    valid_frames = [get_frame_idx(d["im_path"]) for d in seq.data if len(d["gt"]) > 0]

    mot_accum = mm.MOTAccumulator(auto_id=True)

    ignore_regions_x = seq.ignore['ignore_x']
    ignore_regions_y = seq.ignore['ignore_y']
    for i, data in enumerate(seq):
        if i in valid_frames:
            # i corresponds to image index
            ignore_x = ignore_regions_x[i + 1]
            ignore_y = ignore_regions_y[i + 1]

            if len(ignore_x) > 0:
                if not isinstance(ignore_x[0], list):
                    ignore_x = [ignore_x]
            if len(ignore_y) > 0:
                if not isinstance(ignore_y[0], list):
                    ignore_y = [ignore_y]

            ignore_regions = []
            for r_idx in range(len(ignore_x)):
                region = []
                if len(ignore_x[r_idx]) == 0 or len(ignore_y[r_idx]) == 0:
                    continue

                for x, y in zip(ignore_x[r_idx], ignore_y[r_idx]):
                    region.append([x, y])

                try:
                    ignore_region = Polygon(region)
                    ignore_regions.append(ignore_region)
                except:
                    assert False

            gt = data['gt']
            gt_ids = []
            gt_boxes = []

            if gt:
                for gt_id, box in gt.items():
                    gt_ids.append(gt_id)
                    gt_boxes.append(box)

                gt_boxes = np.stack(gt_boxes, axis=0)
                # x1, y1, x2, y2 --> x1, y1, width, height
                gt_boxes = np.stack((gt_boxes[:, 0],
                                    gt_boxes[:, 1],
                                    gt_boxes[:, 2] - gt_boxes[:, 0],
                                    gt_boxes[:, 3] - gt_boxes[:, 1]),
                                    axis=1)
            else:
                gt_boxes = np.array([])

            track_ids = []
            track_boxes = []
            det_boxes = []
            ignore_candidates = []

            for track_id, frames in results.items():
                if i in frames:
                    track_ids.append(track_id)
                    # frames = x1, y1, x2, y2, score
                    track_boxes.append(frames[i][:4])
                    box = frames[i][:4]

                    if use_ignore_regions:
                        x1 = box[0]
                        y1 = box[1]
                        x2 = box[2]
                        y2 = box[3]

                        box_poly = Polygon([
                            [x1, y1],
                            [x2, y1],
                            [x2, y2],
                            [x1, y2],
                            [x1, y1],
                        ])

                        det_boxes.append(box_poly)

            if use_ignore_regions:
                # obtain candidate detections for ignore regions
                region_ious = np.zeros((len(det_boxes), len(ignore_regions)), dtype=np.float32)
                for i in range(len(ignore_regions)):
                    for j in range(len(det_boxes)):
                        if ignore_regions[i].is_valid:
                            poly_intersection = ignore_regions[i].intersection(det_boxes[j]).area
                            poly_union = ignore_regions[i].union(det_boxes[j]).area
                        else:
                            multi_poly = ignore_regions[i].buffer(0)
                            poly_intersection = 0
                            poly_union = 0

                            if isinstance(multi_poly, Polygon):
                                poly_intersection = multi_poly.intersection(det_boxes[j]).area
                                poly_union = multi_poly.union(det_boxes[j]).area
                            elif isinstance(multi_poly, MultiPolygon):
                                poly_intersection = multi_poly.intersection(det_boxes[j]).area
                                poly_union = multi_poly.union(det_boxes[j]).area
                            else:
                                for poly in multi_poly:
                                    poly_intersection += poly.intersection(det_boxes[j]).area
                                    poly_union += poly.union(det_boxes[j]).area

                        # For reference: original IOU calculation
                        # - region_ious[j, i] = poly_intersection / poly_union

                        # Updated IOU calculation
                        region_ious[j, i] = poly_intersection / det_boxes[j].area

                if len(ignore_regions) > 0 and len(det_boxes) > 0:
                    ignore_candidates = np.where((region_ious > ignore_iou_thres).max(axis=1))[0].tolist()

            if track_ids:
                track_boxes = np.stack(track_boxes, axis=0)
                # x1, y1, x2, y2 --> x1, y1, width, height
                track_boxes = np.stack((track_boxes[:, 0],
                                        track_boxes[:, 1],
                                        track_boxes[:, 2] - track_boxes[:, 0],
                                        track_boxes[:, 3] - track_boxes[:, 1]),
                                    axis=1)
            else:
                track_boxes = np.array([])

            distance = mm.distances.iou_matrix(gt_boxes, track_boxes, max_iou=0.5)

            mot_accum.update(
                gt_ids,
                track_ids,
                distance,
                ignore_candidates=ignore_candidates)

    return mot_accum


def evaluate_mot_accums(accums, names, generate_overall=False):
    mh = mm.metrics.create()
    summary = mh.compute_many(
        accums,
        metrics=mm.metrics.motchallenge_metrics + ['num_matches', 'num_objects'],
        names=names,
        generate_overall=generate_overall, )

    str_summary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names, )
    print(str_summary)

    return str_summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mot_path', required=True, help='Path to GT MOT files')
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--result_path', required=True)
    parser.add_argument('--ignore_iou_thres', default=0.1)
    parser.add_argument('--sequence_choice', type=str, nargs="+", default=[])
    args = parser.parse_args()

    mot_accums = []

    mot_path = args.mot_path
    dataset_path = args.dataset_path
    result_path = args.result_path
    use_ignore_regions = True
    ignore_iou_thres = args.ignore_iou_thres

    dataset = PTWrapper(mot_path, dataset_path, vis_threshold=0.1)
    sequence_names = [str(s) for s in dataset if not s.no_gt]
    if len(args.sequence_choice) > 0:
        sequence_names = [f"{s}_mpii_test" for s in args.sequence_choice]

    for seq_idx, seq in enumerate(tqdm(dataset)):
        if seq._seq_name not in sequence_names:
            continue

        if not os.path.isdir(result_path):
            raise FileNotFoundError(f"result path {result_path} does not exist")
        results = seq.load_results(result_path)
        if len(results) == 0:
            print(f"Results not provided for gt sequence {seq._seq_name}")
        mot_accums.append(get_mot_accum(results,
                                        seq,
                                        ignore_iou_thres=ignore_iou_thres,
                                        use_ignore_regions=use_ignore_regions))

    if mot_accums:
        evaluate_mot_accums(mot_accums, sequence_names, generate_overall=True)

if __name__ == '__main__':
    main()
