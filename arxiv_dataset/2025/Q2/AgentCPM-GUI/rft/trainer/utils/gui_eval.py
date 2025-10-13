import re
import json
import json5
import jsonschema
import difflib
import math
from concurrent.futures import ProcessPoolExecutor
from .dataset import SCHEMA

def load_and_validate_action(res:str,):
    action_str = re.search(r'```json(.*?)```', res, re.DOTALL)
    if action_str:
        action_str = action_str.group(1).strip()
    else:
        action_str = res
    action = json5.loads(action_str,allow_duplicate_keys=False)
    # if isinstance(res, str):
    #     action_str = res
    #     action = json5.loads(action_str,allow_duplicate_keys=False)
    # else:
    #     action = res
    
    # action = json5.loads(res,allow_duplicate_keys=False)
    jsonschema.validate(action, SCHEMA)
    return action

global_executor = ProcessPoolExecutor(max_workers=8)

def _action_schema_check(res:str,solution: dict):
    try:
        action:dict = load_and_validate_action(res)
        if "```json" in res:
            return 0.5
        return 1.0
    except jsonschema.ValidationError as e:
        return 0.3
    except Exception as e:
        return 0.0

def action_schema_check(completions, **kwargs):
    global global_executor
    futures = [global_executor.submit(_action_schema_check,completion[0]["content"],) for completion in completions]
    scores = []
    for future in futures:
        try:
            scores.append(future.result(timeout=5)*0.3)
        except TimeoutError as e:
            print("Timeout while checking schema.")
            scores.append(0.0)

    return scores

def _action_type_check(res:str, solution: dict):
    if isinstance(solution,str):
        return difflib.SequenceMatcher(None, res,solution).ratio()
    try:
        action = load_and_validate_action(res)
        # if not ("thought" in action or "think" in action or res.startswith("//") or res.startswith("/*")):
        #     raise Exception("No think.")
        action_keys = set(action.keys())
        solution_keys = set(solution.keys())
        if "think" in action_keys:
            action_keys.remove("think")
        if "think" in solution_keys:
            solution_keys.remove("think")
        if "thought" in action_keys:
            action_keys.remove("thought")
        if "thought" in solution_keys:
            solution_keys.remove("thought")
        # jaccard_index = len(action_keys & solution_keys) / len(solution_keys.union(action_keys))
        # if jaccard_index < 1:
            # print("Mismatched keys in action, Expected: ", solution_keys, " Got: ", action_keys)
        # score = jaccard_index
        # score = max(0,score)
        assert len(action_keys) > 0, "No action found"

        score = len(action_keys & solution_keys)  == len(solution_keys.union(action_keys))
        
        # if "```json" in res:
        #     score = score * 0.9
        
        # here we have to punish extra keys
        # score -= len(action_keys - solution_keys)*0.2
        
        return score
    
    except Exception as e:
        return -1
    

def action_type_check(completions, solution: list[dict], **kwargs):
    global global_executor
    futures = [global_executor.submit(_action_type_check,completion[0]["content"],sol) for completion,sol in zip(completions,solution)]
    scores = []
    for future in futures:
        try:
            scores.append(future.result(timeout=5))
        except TimeoutError as e:
            print("Timeout while checking type.")
            scores.append(0.0)

    return scores

def _action_args_check(res:str, solution: dict, reso: tuple, bbox: list[list]):
    if isinstance(solution,str):
        return difflib.SequenceMatcher(None, res,solution).ratio()
    try:
        action = load_and_validate_action(res)
        # if not ("thought" in action or "think" in action or res.startswith("//") or res.startswith("/*")):
        #     raise Exception("No think.")
        action_keys = set(action.keys())
        solution_keys = set(solution.keys())
        if "think" in action_keys:
            action_keys.remove("think")
        if "think" in solution_keys:
            solution_keys.remove("think")
        if "thought" in action_keys:
            action_keys.remove("thought")
        if "thought" in solution_keys:
            solution_keys.remove("thought")
        
        assert len(action_keys) > 0, "No action found"
        
        # if len(action_keys & solution_keys)  == len(solution_keys.union(action_keys)):
        #     score = 1.0
        # else:
        #     score = 0.0
        score = 0.0
        if "```json" in res:
            score -= 0.05
    except Exception as e:
        return -1

    if len(action_keys - solution_keys) > 0:
        # here we have to punish extra keys
        score -= len(action_keys - solution_keys)*0.1
        # print("Get Extra Action, Expected: ", solution_keys, " Got: ", action_keys, " Score: ", score)
    if len(solution_keys - action_keys) > 0:
        # here we should punish missing keys
        score -= len(solution_keys - action_keys)*0.3
        # print("No Expected Action: ", solution_keys, " Got: ", action_keys, " Score: ", score)

    
    sub_scores = []
    for k in solution.keys():
        if k in ["think","thought"]:
            continue
        if k not in action:
            sub_scores.append(0)
            continue
        sub_score = 0
        match k:
            case "POINT":
                sub_score += calculate_dist_score(action[k], solution[k], reso, bbox[0])
            
            case "duration":
                if action[k] > 150 and action[k] <= 5000:
                    sub_score += 1.0
                else:
                    sub_score -= 0
                    # print("Invalid duration: ", action[k])
            
            case "TYPE":
                similarity = difflib.SequenceMatcher(None, action[k], solution[k]).ratio()
                sub_score += similarity
                # print("Text: ",solution[k],", Got: ", action[k],". Similarity: ", similarity)
                
            case "to":
                if isinstance(solution[k], list):
                    # point direction
                    if isinstance(action[k],list):
                        sub_score += calculate_dist_score(action[k], solution[k], reso, bbox[1])
                    else:
                        sub_score -= 0
                        # print(f"Invalid to for direction {solution[k]}: ", action[k])
                    
                else:
                    # text direction
                    if isinstance(action[k],list):
                        sub_score -= 0
                        # print(f"Invalid to for direction {solution[k]}: ", action[k])
                    else:
                        if action[k] == solution[k]:
                            sub_score += 1.0
                        else:
                            sub_score -= 0
                            # print("Invalid to: ", action[k])
            
            case _:
                if solution[k] is None:
                    if action[k] is None:
                        sub_score += 1.0
                    else:
                        sub_score -= 0
                        # print("Required ", solution[k], ", got: ", action[k])
                else:
                    if action[k] == solution[k]:
                        sub_score += 1.0
                    else:
                        sub_score -= 0
                        # print("Required ", solution[k], ", got: ", action[k])
                        
        sub_scores.append(sub_score)
    if not sub_scores:
        return score
    else:
        return score + sum(sub_scores) / len(sub_scores)
    

def action_args_check(completions, solution: list[dict], resolution, bboxs,**kwargs):
    global global_executor
    futures = [global_executor.submit(_action_args_check,completion[0]["content"],sol,reso,bbox) for completion,sol,reso,bbox in zip(completions,solution,resolution,bboxs)]

    scores = []
    for future in futures:
        try:
            scores.append(future.result(timeout=5))
        except TimeoutError as e:
            print("Timeout while checking type.")
            scores.append(0.0)

    return scores

def _react_check(res:str, solution: dict, reso: tuple, bbox: list[list], step_id):
    if isinstance(solution,str):
        return difflib.SequenceMatcher(None, res,solution).ratio()
    
    res = res.strip()
    if (
        res.count("<think>")   != 1 or
        res.count("</think>")  != 1 or
        res.count("<act>")     != 1 or
        res.count("</act>")    != 1 or
        not res.endswith("</act>")
    ):
        return -1.0
    
    if step_id == 0:
        if not res.startswith("<plan>") or res.count("</plan>") != 1:
            return -1.0
    else:
        if not (res.startswith("<reflection>") or res.startswith("<think>")):
            return -1.0
    
    if not ("<think>" in res and "</think>" in res) or res.startswith("<act>"):
        return - 1.0
    # extract the think content
    think_str = re.search(r'<think>(.*?)</think>', res, re.DOTALL)
    if think_str:
        think_str = think_str.group(1).strip()
    else:
        return - 1.0
    
    # check the action
    try:
        act_str = re.search(r'<act>(.*?)</act>', res, re.DOTALL).group(1).strip()
        action = load_and_validate_action(act_str)
    except:
        return - 1.0
    
    action_keys = set(action.keys())
    solution_keys = set(solution.keys())
    
    if action_keys != solution_keys:
        return len(action_keys & solution_keys) / len(solution_keys.union(action_keys)) - 1
    
    score = 0.0
    
    sub_scores = []
    for k in solution.keys():
        if k not in action:
            sub_scores.append(0)
            continue
        sub_score = 0
        match k:
            case "POINT":
                sub_score += calculate_dist_score(action[k], solution[k], reso, bbox[0])
            
            case "duration":
                if action[k] > 150 and action[k] <= 5000:
                    sub_score += 1.0
                else:
                    sub_score -= 0
                    # print("Invalid duration: ", action[k])
            
            case "TYPE":
                similarity = difflib.SequenceMatcher(None, action[k], solution[k]).ratio()
                sub_score += similarity
                # print("Text: ",solution[k],", Got: ", action[k],". Similarity: ", similarity)
                
            case "to":
                if isinstance(solution[k], list):
                    # point direction
                    if isinstance(action[k],list):
                        sub_score += calculate_dist_score(action[k], solution[k], reso, bbox[1])
                    else:
                        sub_score -= 0
                        # print(f"Invalid to for direction {solution[k]}: ", action[k])
                    
                else:
                    # text direction
                    if isinstance(action[k],list):
                        sub_score -= 0
                        # print(f"Invalid to for direction {solution[k]}: ", action[k])
                    else:
                        if action[k] == solution[k]:
                            sub_score += 1.0
                        else:
                            sub_score -= 0
                            # print("Invalid to: ", action[k])
            
            case _:
                if solution[k] is None:
                    if action[k] is None:
                        sub_score += 1.0
                    else:
                        sub_score -= 0
                        # print("Required ", solution[k], ", got: ", action[k])
                else:
                    if action[k] == solution[k]:
                        sub_score += 1.0
                    else:
                        sub_score -= 0
                        # print("Required ", solution[k], ", got: ", action[k])
                        
        sub_scores.append(sub_score)
    if not sub_scores:
        return score
    else:
        return score + sum(sub_scores) / len(sub_scores)
    
    

def react_check(completions, solution: list[dict], resolution, bboxs, step_id, **kwargs):
    global global_executor
    futures = [global_executor.submit(_react_check,completion[0]["content"],sol,reso,bbox,step) for completion,sol,reso,bbox,step in zip(completions,solution,resolution,bboxs,step_id)]

    scores = []
    for future in futures:
        try:
            scores.append(future.result(timeout=5))
        except TimeoutError as e:
            print("Timeout while checking type.")
            scores.append(-1.0)

    return scores

def calculate_manhattan_distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

def calculate_dist_score(pred_loc: list[list[int,int]], gt_loc: list[int,int], res: tuple[int,int], bbox: list[int]):    
    x_ratio = pred_loc[0]/1000
    y_ratio = pred_loc[1]/1000
    
    gt_x, gt_y = gt_loc
    gt_x_ratio = gt_x /1000
    gt_y_ratio = gt_y /1000
    
    origin_res, now_res = res
    origin_w, origin_h = origin_res
    now_w, now_h = now_res
    
    abs_x = int(x_ratio * origin_w)
    abs_y = int(y_ratio * origin_h)
    gt_abs_x = int(gt_x_ratio * origin_w)
    gt_abs_y = int(gt_y_ratio * origin_h)
    tolerance = 0.05
    
    if bbox is None or not isinstance(bbox, list):
        # print("No bbox provided.")
        # let assume the bbox is 1%x1% windows
        if ((gt_x_ratio - 1e-2) <= x_ratio <= (gt_x_ratio + 1e-2)) and ((gt_y_ratio - 1e-2) <= y_ratio <= (gt_y_ratio + 1e-2)):
            dist_score =  1.0
        # elif ((gt_x_ratio - tolerance) <= x_ratio <= (gt_x_ratio + tolerance)) and ((gt_y_ratio - tolerance) <= y_ratio <= (gt_y_ratio + tolerance)):
        #     dist_score =  0.3
        else:
            # dist_score = -1
            dist_score = 1 - calculate_manhattan_distance(x_ratio, y_ratio, gt_x_ratio, gt_y_ratio) / 2
    
    else:
        left_top = bbox[0]
        right_bottom = bbox[1]
        
        if (left_top[0] <= abs_x <= right_bottom[0]) and (left_top[1] <= abs_y <= right_bottom[1]):
            dist_score = 0.95
            # remain 0.1 for centering
            max_delta = max(abs(abs_x - (left_top[0] + right_bottom[0]) / 2), abs(abs_y - (left_top[1] + right_bottom[1]) / 2))
            dist_score += 0.05 * ((1 - max_delta / 1000)**3)
        # elif ((left_top[0] - tolerance*origin_w) <= abs_x <= (right_bottom[0] + tolerance*origin_w )) and ((left_top[1] - tolerance*origin_h) <= abs_y <= right_bottom[1] + tolerance*origin_h):
        #     dist_score = 0.3
        else:
            # print(f"Point {(x_ratio,y_ratio)} {[abs_x,abs_y]} out of Bbox {[left_top, right_bottom]}, GT: {(gt_x_ratio,gt_y_ratio)} {[gt_abs_x,gt_abs_y]}")
            # dist_score = -1
            dist_score = 1 - calculate_manhattan_distance(x_ratio, y_ratio, gt_x_ratio, gt_y_ratio) / 2
    
    return dist_score
    
    # origin_res, now_res = res
    # origin_w, origin_h = origin_res
    # now_w, now_h = now_res
    
    # x, y = pred_loc
    # gt_x, gt_y = gt_loc
    # gt_x_ratio = gt_x /1000
    # gt_y_ratio = gt_y /1000
    # x_ratio = x / now_w
    # y_ratio = y / now_h
    
    # if x_ratio > 1 or y_ratio > 1:
    #     print("Invalid prediction coordinate: ", pred_loc)
    #     return -1.0
    
    # abs_x = int(x_ratio * origin_w)
    # abs_y = int(y_ratio * origin_h)
    
    
    # if bbox is None or not isinstance(bbox, list):
    #     # print("No bbox provided.")
    #     dist_score = - calculate_manhattan_distance(x_ratio, y_ratio, gt_x_ratio, gt_y_ratio) / 2
        
    # else:
    #     left_top, right_bottom = bbox
    #     if left_top[0] <= abs_x <= right_bottom[0] and left_top[1] <= abs_y <= right_bottom[1]:
    #         dist_score = 0.9
    #         # remain 0.1 for centering
    #         max_delta = max(abs(abs_x - (left_top[0] + right_bottom[0]) / 2), abs(abs_y - (left_top[1] + right_bottom[1]) / 2))
    #         dist_score += 0.1 * ((1 - max_delta / 1000)**3)
    #     else:
    #         print(f"Point {(x_ratio,y_ratio)} {[abs_x,abs_y]} out of Bbox {[left_top, right_bottom]}, GT: {(gt_x_ratio,gt_y_ratio)} {gt_loc}")
    #         dist_score = - calculate_manhattan_distance(x_ratio, y_ratio, gt_x_ratio, gt_y_ratio) / 2
    
    # return dist_score
    
    # 绝对坐标iou
    
    # left = min(pred_loc[0][0], pred_loc[1][0])
    # top = min(pred_loc[0][1], pred_loc[1][1])
    # right = max(pred_loc[0][0], pred_loc[1][0])
    # bottom = max(pred_loc[0][1], pred_loc[1][1])
    
    # origin_res, now_res = res
    # origin_w, origin_h = origin_res
    # now_w, now_h = now_res
    
    # pred_left_top = [int(left/now_w*origin_w),int(top/now_h*origin_h)]
    # pred_right_bottom = [int(right/now_w*origin_w),int(bottom/now_h*origin_h)]
    
    # if pred_left_top[0] >= pred_right_bottom[0] or pred_left_top[1] >= pred_right_bottom[1]:
    #     print("Invalid prediction box: ", pred_left_top, pred_right_bottom)
    #     return -1.0
    
    # if bbox is None or not isinstance(bbox, list):
    #     print("No bbox provided.")
    #     gt_x, gt_y = gt_loc
        
    #     delta_x = abs(gt_x/1000 - (left + right) / (now_w * 2))
    #     delta_y = abs(gt_y/1000 - (top + bottom) / (2 * now_h))
    #     max_delta = max(delta_x,delta_y)
    #     dist_score = - max_delta
    #     return dist_score

    # # calculate CIoU score
    # left_top, right_bottom = bbox
    
    # # Intersection area
    # x1 = max(left_top[0], pred_left_top[0])
    # y1 = max(left_top[1], pred_left_top[1])
    # x2 = min(right_bottom[0], pred_right_bottom[0])
    # y2 = min(right_bottom[1], pred_right_bottom[1])
    # inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # # Compute areas of ground truth and predicted boxes
    # gt_area = max(right_bottom[0] - left_top[0], 0) * max(right_bottom[1] - left_top[1], 0)
    # pred_area = max(pred_right_bottom[0] - pred_left_top[0], 0) * max(pred_right_bottom[1] - pred_left_top[1], 0)
    
    # # IoU calculation with smooth term to avoid division by zero
    # iou = inter_area / (gt_area + pred_area - inter_area + 1e-6)
    
    # # Centers of ground truth and predicted boxes
    # gt_center_x = (left_top[0] + right_bottom[0]) / 2.0
    # gt_center_y = (left_top[1] + right_bottom[1]) / 2.0
    # pred_center_x = (pred_left_top[0] + pred_right_bottom[0]) / 2.0
    # pred_center_y = (pred_left_top[1] + pred_right_bottom[1]) / 2.0
    
    # # Squared distance between the centers
    # center_distance_sq = (pred_center_x - gt_center_x) ** 2 + (pred_center_y - gt_center_y) ** 2
    
    # # Smallest enclosing box
    # enc_left = min(left_top[0], pred_left_top[0])
    # enc_top = min(left_top[1], pred_left_top[1])
    # enc_right = max(right_bottom[0], pred_right_bottom[0])
    # enc_bottom = max(right_bottom[1], pred_right_bottom[1])
    # c_diag_sq = (enc_right - enc_left) ** 2 + (enc_bottom - enc_top) ** 2 + 1e-6  # add smooth term
    
    # # Widths and heights for aspect ratio consistency calculation
    # gt_w = right_bottom[0] - left_top[0]
    # gt_h = right_bottom[1] - left_top[1]
    # pred_w = pred_right_bottom[0] - pred_left_top[0]
    # pred_h = pred_right_bottom[1] - pred_left_top[1]
    
    # # Compute the aspect ratio penalty term v
    # if gt_h == 0 or pred_h == 0:
    #     v = 0.0
    # else:
    #     angle_gt = math.atan(gt_w / (gt_h + 1e-6))
    #     angle_pred = math.atan(pred_w / (pred_h + 1e-6))
    #     v = (4 / (math.pi ** 2)) * (angle_gt - angle_pred) ** 2
    
    # alpha = v / (1 - iou + v + 1e-6)
    # ciou = iou - (center_distance_sq / c_diag_sq) - alpha * v
    
    # return ciou

