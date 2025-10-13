import json
import random
from ai2thor_engine.RocAgent import RocAgent
from utils import *
from prompt import *
import argparse
from tqdm import tqdm
import os
import time
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering

try:
    from web_ui import start_dashboard_server, log_task_start, log_task_complete, log_interaction
    WEB_DASHBOARD_AVAILABLE = True
    print("Web Dashboard is available")
except ImportError as e:
    WEB_DASHBOARD_AVAILABLE = False
    print(f"Web Dashboard is not available: {e}")
    def start_dashboard_server(*args, **kwargs): return None
    def log_task_start(data): pass
    def log_task_complete(success, data=None): pass 
    def log_interaction(data): pass
MODE = "LOCAL" # choose ["LOCAL","API"]
PLATFORM_TYPE="GPU" 

MAX_MODEL_INFER_COUNT=3
def load_data(args):
    cache = {}
    prefix_path = f"./data/{args.model_name}"
    if os.path.exists(prefix_path):
        for pre in os.listdir(prefix_path):
            pre_path = os.path.join(prefix_path, pre)
            if os.path.isdir(pre_path) and "result.json" in os.listdir(pre_path):
                cache[pre] = 1
    with open(args.input_path) as f:
        data = json.load(f)
    
    # 如果指定了task_ids，按数组索引加载这些任务
    if hasattr(args, 'task_ids') and args.task_ids:
        task_indices = [int(id.strip()) for id in args.task_ids.split(',')]
        filtered_data = []
        for idx in task_indices:
            if 0 <= idx < len(data):
                filtered_data.append(data[idx])
            else:
                print(f"--Warning: Task index {idx} out of range (0-{len(data)-1})")
        data = filtered_data
        print(f"--Loading tasks by array indices: {task_indices}")
        print(f"--Found {len(data)} valid tasks")
    
    print(f"--total task count:{len(data)}")
    last_data = []
    for line in data:
        identity = f"""{line["identity"]}_{line['tasktype']}_{line["scene"]}_{line['instruction_idx']}"""
        if identity not in cache:
            last_data.append(line)
    print(f"--cache:{len(data)-len(last_data)}---remaining evaluation tasks:{len(last_data)}")
    # random.shuffle(last_data)
    per_group_count = len(last_data)//args.total_count
    group_data = [last_data[i*per_group_count: (i+1)*per_group_count if i !=args.total_count-1 else len(last_data)] for i in range(args.total_count)]
    print(f"--Current process evaluation data:{len(group_data[args.cur_count-1])}")
    group_data[args.cur_count-1].reverse()
    return group_data[args.cur_count-1]

def get_trajectory(controller, task, model, max_step=10, port=-1):
    autogn = None  # 初始化为None，避免异常处理中的未定义错误
    try:
        scene = task["scene"]
        task_name = task["taskquery"]
        index = task["identity"]
        if task["tasktype"].startswith("ordered_pickup_two_object_and_put"):
            tasktype="ordered_pickup_two_object_and_put"
        else:tasktype=task["tasktype"]
        max_step=get_max_steps(tasktype)

        save_path=f"./data/{model}/{index}_{task['tasktype']}_{scene}_{task['instruction_idx']}"
        
        print(f"******** Task Name: {task_name} *** Max Steps: {max_step} ********")
        print(f"******** Task Record: {save_path} ********")
        
        # 记录任务开始 - Web Dashboard
        log_task_start({
            'identity': index,
            'taskquery': task_name,
            'scene': scene,
            'tasktype': tasktype,
            'max_steps': max_step,
            'save_path': save_path
        })
        autogn = RocAgent(controller, save_path, scene, visibilityDistance=20, gridSize=0.1, fieldOfView=90, 
                            target_objects=task["target_objects"],
                            related_objects=task["related_objects"],
                            navigable_objects=task["navigable_objects"],
                            taskid=task["identity"],
                            platform_type=PLATFORM_TYPE)
        print("RoctAgent Initialization successful!!!")
        
        # ENHANCEMENT (9.4): Set task context for improved VLM prompts
        autogn.set_task_context(
            task_description=task_name,
            task_type=tasktype
        )
        
        objects = autogn.eventobject.get_objects_type(autogn.controller.last_event)
        action, pre_action = "init", "init"
        item, pre_item = None, None
        trajectory = []
        legal_locations = []
        legal_objects = []
        images = []
        response = ""
        messages = [{"role": "system","content": EMBODIED_SYSTEM_PROMPT}]
        call_model_count = 0
        con_same_action = 0
        last_step_count = autogn.step_count
        while action != "end" and autogn.step_count < max_step and call_model_count<MAX_MODEL_INFER_COUNT:
            last_step_count = autogn.step_count
            if action==pre_action and item==pre_item:
                con_same_action+=1
                if con_same_action == MAX_MODEL_INFER_COUNT:
                    dic = {
                        "response": response,
                        "action": "end",
                        "object": None,
                        "legal_locations": [],
                        "legal_objects": [],
                        "success": 0,
                        "errorInfo": "",
                        "images": []
                    }
                    trajectory.append(dic)
                    # Don't stop the controller here as it's shared between tasks
                    # autogn.controller.stop()
                    result_dir = autogn.result_dir
                    del autogn
                    return trajectory, messages, result_dir
            else:
                con_same_action = 0
                pre_action=action
                pre_item=item
            if invalid_action(action):
                user_text = INVALID_ACTION_PROMPT.format() # action=temp_action
                dic = {
                        "response": response,
                        "action": action,
                        "object": item,
                        "legal_locations": legal_locations,
                        "legal_objects": legal_objects,
                        "success": 0,
                        "errorInfo": user_text,
                        "images": []
                    }
                trajectory.append(dic)
                messages.append({
                        "role": "user",
                        "content": user_text+USER_IMAGE_PREFIX_ERROR
                    })
            
            else:
            
                print(autogn.step_count,"****** begin exec action:",action, item ,"***")
                success, image_fp, legal_locations, legal_objects = autogn.exec(action, item)
                print(autogn.step_count,"****** end exec action:",action, item ,"***")
                user_text = ""
            
                if not success or image_fp is None or image_fp == []:
                    if "navigate to" in action:                    
                        if item=="No Suitable Object":
                            user_text = f"""<|feedback|>Action: "{action}" is illegal, the name of the navigated object doesn't quite match the obejct in the image, please try navigating to another object first.\n"""
                        else:                           
                            user_text = f"""<|feedback|>Action: "{action}" is illegal, "{item}" is the most relevant item in this room and "{raw_action}". Object: "{item}" is not currently navigable, you can try "navigate to <object>" to reach nearby, larger objects for closer observation.\n"""

                    else:
                        if item=="No Suitable Object":    
                            user_text = f"""<|feedback|>Action: "{action}" is illegal, the name of the object doesn't quite match the obejct in the image, Please try interacting with another object or navigating to another object.\n"""
                        else:                             
                            user_text = f"""<|feedback|>Action: {raw_action} is illegal, Object: {item} is currently unavailable for interaction. Possible situations include: {item} does not exist in your current view; you are too far away from {item}; the {item} cannot perform operation {action}.\nYou can try \"move forward\" to approach the target object or \"navigate to <object>\" to reach nearby, larger objects for closer inspection."""
                        
                    dic = {
                        "response": response,
                        "action": action,
                        "object": item,
                        "legal_locations": legal_locations,
                        "legal_objects": legal_objects,
                        "success": 0,
                        "errorInfo": user_text,
                        "images": image_fp
                    }
                    trajectory.append(dic)
                    
                    messages.append({
                        "role": "user",
                        "content": user_text+USER_IMAGE_PREFIX_ERROR
                    })
            
                
                else:
                    dic = {
                        "response": response,
                        "action": action,
                        "object": item,
                        "legal_locations": legal_locations,
                        "legal_objects": legal_objects,
                        "success": 1,
                        "errorInfo": "",
                        "images": image_fp
                    }
                    trajectory.append(dic)
                    if isinstance(image_fp, list):
                        for i in image_fp:
                            images.append(i)
                            user_text += "<image>"
                    else:
                        images.append(image_fp)
                        user_text += "<image>"
                
                    
                    if action == "init":
                        if action == "init":
                            if MODE=="LOCAL":
                                TASK_PREFIX=TASK_PREFIX_PUT
                            elif MODE=="API":
                                TASK_PREFIX=TASK_PREFIX_PUT_IN
                        messages.append({"role":"user",
                                        "content":user_text + TASK_PREFIX.format(
                                            task_name=task_name, )})
                                            
                    
                    elif "move forward" in action:
                        messages.append({
                            "role": "user",
                            "content": user_text+USER_IMAGE_PREFIX_MOVE_FORWARD.format(
                                action=action
                            )
                        })
                    else:
                        temp_action = action if item is None else action + " " + item
                        messages.append({"role":"user",
                                        "content":user_text+USER_IMAGE_PREFIX.format(
                                            action=temp_action,
                                            )})
                
            inputs = {"messages": messages, "images": images}
            
            if MODE=="API":
                api_messages = prepare_api_messages(inputs)
                response = call_llm(api_messages, model)
                call_model_count += 1
            elif MODE=="LOCAL":
                local_messages = prepare_deploy_messages(inputs)
                response = local_model(local_messages, port) #local model predict
                call_model_count += 1
            
            if response == "":
                print(f"--task{task['identity']}Trajectory acquisition failed -- request timed out, model is not output, end the current evaluation task!!!")
                return None, None, None

            if autogn.step_count!=last_step_count:
                call_model_count = 0
            else:
                print(f"******** Action_Execute_Count: {autogn.step_count} *** Call_VLM_Count: {call_model_count} ********")  
            raw_action, action, item = macth_action_item(response, autogn.action_space, objects,MODE)

            messages.append({"role":"assistant","content":response})
        
        dic = {
            "response": response,
            "action": "end",
            "object": None,
            "legal_locations": legal_locations,
            "legal_objects": legal_objects,
            "success": 1,
            "errorInfo": "",
            "images": []
        }
        trajectory.append(dic)
        # Don't stop the controller here as it's shared between tasks
        # autogn.controller.stop()
        del autogn
        return trajectory, messages, save_path
    except Exception as e:
        print(e)
        if autogn is not None:
            try:
                # Don't stop the controller here as it's shared between tasks
                # autogn.controller.stop()  # Commented out to fix multi-task testing
                del autogn
            except:
                pass  # 如果清理失败，继续执行
        print(f"--task{task['identity']}Track acquisition failed -- emulator /api exception, end the current evaluation task!!!--")
        return None, None, None

def test(controller, test_data, model="Qwen2.5-VL-3B-Instruct", port=-1):
    save_path=f"./data/{model}/{test_data['identity']}_{test_data['tasktype']}_{test_data['scene']}_{test_data['instruction_idx']}"
    if os.path.exists(f"{save_path}/result.json"):
        print(f"""--task{test_data["identity"]}It has been evaluated successfully, skip it.---""")
        return
    
    test_start_time = time.time()
    id = test_data['instruction_idx']
    if 'task_metadata' in test_data:
        scene_metadata = test_data['task_metadata']
        key_actions = [(a['action']+" "+ a["objectType"]).strip() for a in scene_metadata['actions']]
        
    else:
        with open(f"./data/single_search_task_metadata/{test_data['scene']}.json") as f:
            scene_metadata = json.load(f)[0]
        key_actions = [(a['action']+" "+ a["objectType"]).strip() for a in scene_metadata[id]['actions']]
    
    
    trajectory, messages, result_dir = get_trajectory(controller, test_data, model, port=port)
    
    if trajectory is None:
        print(f"--task{test_data['identity']}failed--")
        return
    metric_dic = metric(test_data, trajectory, key_actions)
    test_end_time = time.time()
    elapsed_time = int(test_end_time - test_start_time)
    
    # Extract success status from calculated metrics
    task_success = metric_dic["success"] == 1
    completeness = metric_dic["completeness"]
    
    with open(f"{result_dir}/result.json","w") as f:
        f.write(json.dumps({
            "identity":test_data["identity"],
            "scene": test_data["scene"],
            "tasktype": test_data["tasktype"],
            "instruction_idx": test_data["instruction_idx"],
            "model": model,
            "taskname":test_data["taskname"],
            "trajectory": trajectory,
            "messages": messages,
            "key_actions": key_actions,
            "metrics": metric_dic,
            "time": elapsed_time,
            "maxstep": get_max_steps(test_data["tasktype"]),
        }, indent=4))
    
    # Use actual metrics instead of always printing success
    if task_success:
        print(f"""--task{test_data["identity"]}evaluate SUCCEEDED--- (completeness: {completeness:.2f})""")
    else:
        print(f"""--task{test_data["identity"]}evaluate FAILED--- (completeness: {completeness:.2f})""")

if __name__ == "__main__":
    
    if MODE=="LOCAL":
        parser = argparse.ArgumentParser()

        parser.add_argument("--input_path", type=str, default="./data/test_809.json", help="input file path")
        parser.add_argument("--model_name", type=str, default="Qwen2.5-VL-3B-Instruct", help="")
        parser.add_argument("--batch_size", type=int, default=200, help="")
        parser.add_argument("--port", type=int, default=10000, help="")
        parser.add_argument("--cur_count", type=int, default=1, help="")
        parser.add_argument("--total_count", type=int, default=4, help="")
        parser.add_argument("--task_ids", type=str, default=None, help="Comma-separated task indices to run")
        parser.add_argument("--dashboard_port", type=int, default=8080, help="Web dashboard port")
        parser.add_argument("--no_dashboard", action="store_true", help="Disable web dashboard")
        args = parser.parse_args()
        print(args)
        data = load_data(args)

        # Initialize Web Dashboard for LOCAL mode
        dashboard_thread = None
        dialogue_enabled = False

        # Check if dialogue system is enabled
        try:
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), 'ai2thor_engine'))
            from ai2thor_engine.RocAgent import RocAgent

            # Check RocAgent default settings
            dialogue_enabled = getattr(RocAgent, '_default_enable_dialogue_system', False)

            # Fallback: check source code for dialogue system setting
            if not dialogue_enabled:
                try:
                    import inspect
                    source = inspect.getsource(RocAgent.__init__)
                    if 'self.enable_dialogue_system = True' in source:
                        dialogue_enabled = True
                except:
                    pass
        except Exception as e:
            print(f"Warning: Failed to check dialogue system status: {e}")
            dialogue_enabled = False

        # Launch Web Dashboard if conditions are met
        if dialogue_enabled and not args.no_dashboard and WEB_DASHBOARD_AVAILABLE:
            print(f"\n[INFO] Starting Web Dashboard on port {args.dashboard_port}...")
            dashboard_thread = start_dashboard_server(port=args.dashboard_port, auto_open=False)
            if dashboard_thread:
                print(f"[INFO] Web Dashboard successfully started at: http://localhost:{args.dashboard_port}")
                import time
                time.sleep(2)  # Allow dashboard initialization
            else:
                print("[ERROR] Web Dashboard failed to start")
        elif not dialogue_enabled:
            print("[INFO] Dialogue system is disabled - Web Dashboard not started")
        elif args.no_dashboard:
            print("[INFO] Web Dashboard disabled via --no_dashboard flag")
        elif not WEB_DASHBOARD_AVAILABLE:
            print("[WARNING] Web Dashboard module not available for import")

        success_count = 0
        # controller = None
        controller = Controller(
            platform=CloudRendering,
            snapToGrid=False,
            quality='Medium',
            agentMode="default",
            massThreshold=None,
            scene='FloorPlan1',
            visibilityDistance=20,
            gridSize=0.1,
            renderDepthImage=False,
            renderInstanceSegmentation=True,
            width=800,
            height=450,
            fieldOfView=90,
        )
        for test_data in tqdm(data):
            try:
                result_file = f"./data/{args.model_name}/{test_data['identity']}_{test_data['tasktype']}_{test_data['scene']}_{test_data['instruction_idx']}/result.json"
                existed_before = os.path.exists(result_file)
                
                test(controller, test_data, args.model_name, args.port)
                
                # Check if task actually succeeded by reading the metrics from result.json
                if os.path.exists(result_file) and not existed_before:
                    try:
                        with open(result_file, 'r') as f:
                            result_data = json.load(f)
                            # Use actual success metric instead of just file existence
                            if result_data.get('metrics', {}).get('success', 0) == 1:
                                success_count += 1
                    except (json.JSONDecodeError, KeyError):
                        # If we can't read metrics, fall back to file existence (backward compatibility)
                        print(f"Warning: Could not read metrics from {result_file}, using file existence as success indicator")
                        success_count += 1
            except Exception as e:
                print(e)
                print(f"--task{test_data['identity']}failed, End the current evaluation task!!!--")
                continue
        print(f"--The current process evaluation task end--total task count:{len(data)}successed task count:{success_count}")
        # Stop controller after all tasks are completed
        controller.stop()
    
    
    elif MODE=="API":
        match_item_model="gpt-4o-mini"
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--input_path", type=str, default="./data/test_809.json", help="input file path")
        parser.add_argument("--model_name", type=str, default="gpt-4o-mini", help="")
        parser.add_argument("--batch_size", type=int, default=200, help="")
        parser.add_argument("--port", type=int, default=10000, help="")
        parser.add_argument("--cur_count", type=int, default=1, help="")
        parser.add_argument("--total_count", type=int, default=4, help="")
        parser.add_argument("--task_ids", type=str, default=None, help="Comma-separated task array indices to run (e.g., '0,1,2' for first 3 tasks)")
        parser.add_argument("--dashboard_port", type=int, default=8888, help="Web dashboard port")
        parser.add_argument("--no_dashboard", action="store_true", help="Disable web dashboard (only applies to dialogue mode)")
        args = parser.parse_args()
        print(args)
        
        data = load_data(args)
        
        # Launch Web Dashboard - Only for dialogue mode
        dashboard_thread = None
        
        # Check if dialogue system is enabled
        dialogue_enabled = False
        try:
            # We need to check the actual RocAgent class default value
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), 'ai2thor_engine'))
            from ai2thor_engine.RocAgent import RocAgent
            
            # Create a temporary instance to check the default setting
            # We can't instantiate fully without controller, so check class defaults
            dialogue_enabled = getattr(RocAgent, '_default_enable_dialogue_system', False)
            
            # Fallback: create minimal instance to check instance defaults
            if not dialogue_enabled:
                try:
                    # Check what the __init__ sets as default
                    import inspect
                    source = inspect.getsource(RocAgent.__init__)
                    if 'self.enable_dialogue_system = True' in source:
                        dialogue_enabled = True
                except:
                    pass
        except:
            dialogue_enabled = False
        
        # Only launch dashboard if dialogue system is enabled AND --no_dashboard not specified
        if dialogue_enabled and not args.no_dashboard and WEB_DASHBOARD_AVAILABLE:
            print(f"\nLaunching Web dashboard for dialogue mode...")
            dashboard_thread = start_dashboard_server(port=args.dashboard_port, auto_open=True)
            if dashboard_thread:
                print(f"Web dashboard: http://localhost:{args.dashboard_port}")
            else:
                print("Web dashboard failed.")
        elif not dialogue_enabled:
            print("Dialogue system disabled - no dashboard needed")
        elif args.no_dashboard:
            print("Dashboard disabled by --no_dashboard flag")
        
        # If no tasks to run, keep dashboard running
        if len(data) == 0:
            print("No tasks to run, but keeping dashboard active for monitoring...")
            if dashboard_thread:
                try:
                    import time
                    time.sleep(5)  # Keep it running for a bit
                    print(f"Dashboard is running at http://localhost:{args.dashboard_port}")
                except KeyboardInterrupt:
                    print("Dashboard stopped by user")
            exit(0)
        success_count = 0
        
        controller = Controller(
            platform=CloudRendering,
            snapToGrid=False,
            quality='Medium',
            agentMode="default",
            massThreshold=None,
            scene='FloorPlan1',
            visibilityDistance=20,
            gridSize=0.1,
            renderDepthImage=False,
            renderInstanceSegmentation=True,
            width=800,
            height=450,
            fieldOfView=90,
        )
        
        for test_data in tqdm(data):
            try:
                result_file = f"./data/{args.model_name}/{test_data['identity']}_{test_data['tasktype']}_{test_data['scene']}_{test_data['instruction_idx']}/result.json"
                existed_before = os.path.exists(result_file)
                
                test(controller, test_data, args.model_name, args.port)
                
                # Check if task actually succeeded by reading the metrics from result.json
                if os.path.exists(result_file) and not existed_before:
                    try:
                        with open(result_file, 'r') as f:
                            result_data = json.load(f)
                            # Use actual success metric instead of just file existence
                            if result_data.get('metrics', {}).get('success', 0) == 1:
                                success_count += 1
                    except (json.JSONDecodeError, KeyError):
                        # If we can't read metrics, fall back to file existence (backward compatibility)
                        print(f"Warning: Could not read metrics from {result_file}, using file existence as success indicator")
                        success_count += 1
            except Exception as e:
                print(e)
                print(f"--task{test_data['identity']}failed, End the current evaluation task!!!--")
                continue
        print(f"--The current process evaluation task end--total task count:{len(data)}successed task count:{success_count}")
        # Stop controller after all tasks are completed
        controller.stop()
    
    # from concurrent.futures import ThreadPoolExecutor
    # from tqdm import tqdm
    # with ThreadPoolExecutor(5) as executor:
    #     for match in tqdm(
    #         executor.map(test, data), total=len(data)
    #         ):
    #             pass
    # with ThreadPoolExecutor(2) as executor:
    #     futures = []
    #     for test_data, index in zip(data, [i for i in range(len(data))]):
    #         futures.append(executor.submit(test, test_data, index, args.model_name))
        
    #     for future in tqdm(futures, total=len(data)):
    #         future.result()