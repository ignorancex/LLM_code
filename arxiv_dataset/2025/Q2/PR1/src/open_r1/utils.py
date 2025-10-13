import json
import re
import os
import torch
import math
from scipy.optimize import linear_sum_assignment
from torch import nn
from torchvision.ops.boxes import box_area
from scipy.optimize import linear_sum_assignment

def get_qa_pairs(conversation: list):
    qa_pairs = []
    for i in range(0, len(conversation), 2):
        question = conversation[i]['value']
        answer = conversation[i+1]['value']
        qa_pairs.append((question, answer))
    return qa_pairs

def load_image(image_path):
    from PIL import Image
    import megfile
    from io import BytesIO
    import os
    os.environ['OSS_ENDPOINT'] = 'http://oss.i.basemind.com'
    if 's3://' in image_path:
        with megfile.smart_open(image_path, "rb") as f:
            bytes_data = f.read()
        image = Image.open(BytesIO(bytes_data), "r").convert('RGB')
    else:
        image = Image.open(image_path).convert('RGB')
    return image

# make conversation for text hf-dataset
def make_conversation(
        example,
        system_prompt,
        question_template=None,
        answer_template=None
    ):
    question = example["problem"] if question_template is None else question_template.format(question=example["problem"])
    answer = example["solution"] if answer_template is None else answer_template.format(answer=example["solution"])
    return {
        "prompt": json.dumps([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]),
    }

# make conversation for multi-modal hf-dataset
def make_conversation_image(
        example,
        system_prompt,
        question_template=None,
        answer_template=None
    ):
    question = example["problem"] if question_template is None else question_template.format(question=example["problem"])
    question = question + "." if not question.endswith(".") else question
    return {
        "prompt": json.dumps([
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            },
        ]),
    }

# make conversation for json dataset
def json_map(
        example,
        system_prompt,
        question_template=None,
        answer_template=None
    ):
    '''
    1. example:
        {
            "problem": <str>,
            "image": <image_path>,
            "solution": <int/str>
        }
    2. system_prompt: <str>
    3. question_template: <str>, e.g. "xx {question}?"
    4. answer_template: <str>, e.g. "<answer> {answer} </answer>."
    '''
    image_path = example['image']
    question = example['problem'] if question_template is None else question_template.format(question=example['problem'])
    solution = example['solution'] if answer_template is None else answer_template.format(answer=example['solution'])
    
    rst = {
        "image": load_image(image_path),
        "prompt": json.dumps([
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            },
        ]),
        "solution": solution,
        "problem": question,
    }
    if 'clicks' in example:
        rst['clicks'] = example['clicks']
    return rst

def parse_float_sequence_within(input_str):
    """
    Extract the first sequence of four floating-point numbers within square brackets from a string.

    Args:
    input_str (str): A string that may contain a sequence of four floats within square brackets.

    Returns:
    list: A list of four floats if the pattern is found, or a list of four zeros if the pattern is not found.
    """
    # Define the regex pattern to find the first instance of four floats within square brackets
    # TODO: add more patterns to support various formats
    # pattern1 [num, num, num, num]
    pattern = r"\[\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\s*\]"

    # Use re.search to find the first match of the pattern in the input string
    match = re.search(pattern, input_str)

    # If a match is found, convert the captured groups into a list of floats
    if match:
        return [float(match.group(i)) for i in range(1, 5)]
    # pattern2 (num, num, num, num)
    pattern = r"\(\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\s*\)"
    match = re.search(pattern, input_str)
    if match:
        return [float(match.group(i)) for i in range(1, 5)]
    # pattern3 (num, num), (num, num)
    pattern = r"\(\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\),\s*\(\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)"
    match = re.search(pattern, input_str)
    if match:
        return [float(match.group(i)) for i in range(1, 5)]
    # If the input does not contain the pattern, return the null float sequence
    return [0, 0, 0, 0]

def extract_bbox_answer(content):
    is_qwen2vl = False
    if "<|box_start|>" in content:
        is_qwen2vl = True
    bbox = parse_float_sequence_within(content)
    if not is_qwen2vl:
        bbox = [int(x * 1000) for x in bbox]
    return bbox, is_qwen2vl

def compute_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / union_area
    return iou

def save_dict_to_json(dict_data, filename):
    """
    Save a dictionary to a JSON file.
    
    Args:
        dict_data (dict): Dictionary to be saved
        filename (str): Path to the output JSON file
    """
    try:
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(json.dumps(dict_data, indent=4))
    except Exception as e:
        print(f"error saving dictionary to {filename}: {e}")

def save_args_to_txt(args, filename):
    """
    Save the parsed arguments to a txt file.
    
    Args:
        args (argparse.Namespace): The parsed arguments
        filename (str): The path to the output txt file
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")

@torch.jit.script
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

@torch.jit.script
def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-8)  # 增加数值稳定性
    return iou, union

def generalized_box_iou(boxes1, boxes2):
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    
    iou, union = box_iou(boxes1, boxes2)
    
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1] + 1e-8  # 防止除零
    
    return iou - (area - union) / area

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1, cost_bbox: float = 5, cost_giou: float = 2, threshold: float = 0.5):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.threshold = threshold
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)
        
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        cost_class = -out_prob[:, tgt_ids]
        cost_bbox = torch.cdist(box_xyxy_to_cxcywh(out_bbox), 
                              box_xyxy_to_cxcywh(tgt_bbox), p=1)
        cost_giou = -generalized_box_iou(out_bbox, tgt_bbox)
        
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()
        
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        final_indices = []
        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            pred_boxes = outputs["pred_boxes"][batch_idx]
            tgt_boxes = targets[batch_idx]["boxes"]
            matched_ious, _ = box_iou(pred_boxes[src_idx], tgt_boxes[tgt_idx])
            keep = matched_ious.diag() > self.threshold
            final_indices.append((torch.as_tensor(src_idx, dtype=torch.int64)[keep],
                                  torch.as_tensor(tgt_idx, dtype=torch.int64)[keep]))
            
        return final_indices

def get_box_size(boxes):
    # small: 32*32, medium: 96*96
    weights = []
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    for area in areas:
        if area < 32*32:
            weights.append(3)
        elif area < 96*96:
            weights.append(2)
        else:
            weights.append(1)
    return torch.tensor(weights)

def calculate_reward(pred_boxes, target_boxes, pred_ids, target_ids):
    cls_reward = (pred_ids == target_ids).float().mean()
    iou_reward, _ = box_iou(pred_boxes, target_boxes)
    iou_reward = iou_reward.diag()
    is_size_weight = os.getenv("IS_SIZE_WEIGHT", "False")
    if is_size_weight == "True":
        iou_reward = (get_box_size(target_boxes) * iou_reward).sum() / (iou_reward.sum() * len(iou_reward) + 1e-8)
    else:
        iou_reward = iou_reward.mean()
    return cls_reward, iou_reward

def parse_json(contents, cate_to_id):
    pred_id = []
    pred_box = []
    wrong_format = 0
    for content in contents:
        try:
            bbox = content["bbox_2d"]
            label_id = int(cate_to_id[content["label"]])
            pred_id.append(label_id)
            pred_box.append(bbox)
        except:
            wrong_format += 1
    if not pred_id:
        return torch.empty(0), torch.empty((0,4))
    format_reward = 1 - wrong_format / len(contents)
    return torch.tensor(pred_id), torch.tensor(pred_box), format_reward

def get_cate_to_id():
    return {"person": 1, "bicycle": 2, "car": 3, "motorcycle": 4, "airplane": 5, "bus": 6, "train": 7, "truck": 8, "boat": 9, "traffic light": 10, "fire hydrant": 11, "stop sign": 13, "parking meter": 14, "bench": 15, "bird": 16, "cat": 17, "dog": 18, "horse": 19, "sheep": 20, "cow": 21, "elephant": 22, "bear": 23, "zebra": 24, "giraffe": 25, "backpack": 27, "umbrella": 28, "handbag": 31, "tie": 32, "suitcase": 33, "frisbee": 34, "skis": 35, "snowboard": 36, "sports ball": 37, "kite": 38, "baseball bat": 39, "baseball glove": 40, "skateboard": 41, "surfboard": 42, "tennis racket": 43, "bottle": 44, "wine glass": 46, "cup": 47, "fork": 48, "knife": 49, "spoon": 50, "bowl": 51, "banana": 52, "apple": 53, "sandwich": 54, "orange": 55, "broccoli": 56, "carrot": 57, "hot dog": 58, "pizza": 59, "donut": 60, "cake": 61, "chair": 62, "couch": 63, "potted plant": 64, "bed": 65, "dining table": 67, "toilet": 70, "tv": 72, "laptop": 73, "mouse": 74, "remote": 75, "keyboard": 76, "cell phone": 77, "microwave": 78, "oven": 79, "toaster": 80, "sink": 81, "refrigerator": 82, "book": 84, "clock": 85, "vase": 86, "scissors": 87, "teddy bear": 88, "hair drier": 89, "toothbrush": 90}

def match_and_compute_distances(pred, gt, return_indices=False):
    """
    Matches points from pred to gt based on the minimum L2 distance using the Hungarian algorithm.
    Returns the distances between the matched points. Optionally returns the indices of matched points.

    Parameters:
    - pred: List of prediction points (list of tuples)
    - gt: List of ground truth points (list of tuples)
    - return_indices: Whether to return the indices of matched points (default: False)

    Returns:
    - distances: Tensor of distances between matched points
    - (row_ind, col_ind): Indices of matched points (if return_indices is True)
    """
    pred_tensor = torch.tensor(pred, dtype=torch.float32)
    gt_tensor = torch.tensor(gt, dtype=torch.float32)
    
    cost_matrix = torch.cdist(pred_tensor, gt_tensor, p=2)
    row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().numpy())
    
    # Extract matched points
    matched_pred = pred_tensor[torch.from_numpy(row_ind)]
    matched_gt = gt_tensor[torch.from_numpy(col_ind)]
    
    # Compute distances
    distances = torch.norm(matched_pred - matched_gt, p=2, dim=1)
    
    if return_indices:
        return distances, (torch.as_tensor(row_ind, dtype=torch.int64), torch.as_tensor(col_ind, dtype=torch.int64))
    else:
        return distances

def format_reward(completions, pattern, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def pr1_counting_format_reward(completions, **kwargs):
    pattern = r"<\|object_ref_start\|>[^<]*<\|object_ref_end\|>(?:<\|box_start\|>[^<]*<\|box_end\|>)+<\|im_end\|>"
    return format_reward(completions, pattern)

# verify by points
def extract_boxes(s):
    res = []
    pattern = r'<\|box_start\|>(.*?)<\|box_end\|>'
    bboxs = re.findall(pattern, s)
    for bbox in bboxs:
        try:
            (x1, y1), (x2, y2) = eval(bbox)
            res.append(((x1+x2)/2, (y1+y2)/2))
        except:
            pass
    return res

def num_scale(pred, gt):
    scale = 1000 * math.sqrt(2)
    dist_score = 1 - (match_and_compute_distances(pred, gt) / scale).mean()
    return (min(len(pred), len(gt)) / max(len(pred), len(gt))) * dist_score