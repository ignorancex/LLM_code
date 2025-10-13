import math
try:
    from utils import *
except Exception as e:
    print(e)
try:
    from .utils import add_text_to_image, add_border, EventObject
except Exception as e:
    print(e)

from .baseAgent import BaseAgent
from tqdm import tqdm
import numpy as np
import cv2, json

class RocAgent(BaseAgent):
    STATE_OBSERVATION = "observation"
    STATE_PLANNING = "planning"
    STATE_THINKING = "thinking"
    STATE_REFLECTION = "reflection"
    STATE_DECISION_MAKING_STATE = "decision_making"
    STATE_VERIFICATION = "verification"
    STATE_END = "end"
    def __init__(self, controller, save_path="./data/", scene="FloorPlan203", 
                 visibilityDistance=1.5, gridSize=0.25, fieldOfView=90, target_objects=[], related_objects=[], navigable_objects=[], taskid=0,platform_type="GPU"):
        super().__init__(controller, scene, visibilityDistance, gridSize, fieldOfView,platform_type)
        
        # Web Dashboard logging integration
        try:
            import sys
            import os
            # Add evaluate directory to sys.path for web_dashboard import
            evaluate_dir = os.path.join(os.path.dirname(__file__), '..')
            if evaluate_dir not in sys.path:
                sys.path.append(evaluate_dir)
            
            from web_ui import log_interaction, log_vlm_call
            self._log_interaction = log_interaction
            self._log_vlm_call = log_vlm_call
            self._web_logging_enabled = True
            print("Web Dashboard logging integration successful")
        except ImportError as e:
            self._log_interaction = lambda x: None
            self._log_vlm_call = lambda x: None
            self._web_logging_enabled = False
            print(f"Web Dashboard logging integration failed: {e}")
            
        self.env, self.executor, self.monitor, self.planner = self.build_agent()
        self.pre_navigate_location=""
        self.agent_state = []
        self.object_state = {}
        self.target_objects = []
        self.navigale_objects = {}
        self.state = ""
        self.result_dir = f"{save_path}"
        self.navigable_objects = {}
        self.legal_interactions = {}
        self.current_container = None
        self.objecttype2object={} # navigable object type to object mapping
        self.action_space = {
            "init": self.init_agent_corner,
            "navigate to": self.navigate,
            "pickup": self.pick_up,
            "put": self.put_in,
            "put in":self.put_in,   # for MODE=API
            "toggle": self.toggle,
            "open": self.open,
            "close": self.close,
            "observe": self.observe,
            "move forward": self.move_forward,
            "end": "end",
        }
        self.related_objects=related_objects
        self.target_item_type2obj_id = {}
        for target_obj in target_objects:
            if target_obj.split("|")[0] not in self.target_item_type2obj_id:
                self.target_item_type2obj_id[target_obj.split("|")[0]] = []
            self.target_item_type2obj_id[target_obj.split("|")[0]].append(target_obj)
        
        for obj in self.controller.last_event.metadata['objects']:
            if obj['objectType'] not in self.objecttype2object:
                self.objecttype2object[obj['objectType']]=[]
            self.objecttype2object[obj['objectType']].append(obj)
        
        for navigable_obj in navigable_objects:
            if navigable_obj not in self.navigable_objects:
                self.navigable_objects[navigable_obj] = 0
            self.navigable_objects[navigable_obj] += 1
        self.taskid = str(taskid)
        self.objid2position={}
        with open("./data/agent_positions.json") as f:
            custom_position_data = json.load(f)
        for taskid in custom_position_data:
            temp_data = custom_position_data[taskid]
            for objid in temp_data:
                if objid != "scene" and objid != "tasktype" and objid != "taskname":
                    self.objid2position[objid] = temp_data[objid]

        # if self.taskid in custom_position_data:
        #     self.objid2position = custom_position_data[self.taskid]
        # self.init_agent_corner()
        
        # Enhanced navigation configuration
        self.enable_object_indexing = True      # For object indexing
        self.enable_dialogue_system = True      # For VLM-based disambiguation
        self.enable_multi_view = True           # For multi-view observation
        self.confidence_gap_threshold = 30      # Auto-select if confidence gap > 30%
        self.current_task_description = ""      # Set by task context
        self.current_gpt4o_reasoning = ""       # Store GPT-4o reasoning for VLM prompts
        
        # Disambiguation mode configuration
        self.disambiguation_mode = "human_only_random_fallback"
        # Options:
        #   "human_first_vlm_fallback" - Human first, VLM analysis as fallback
        #   "vlm_first_human_choice"   - VLM analysis first, human choice with confidence scores
        #   "human_only_random_fallback" - Human only, random selection as fallback
        self.human_selection_timeout = 60        # Seconds to wait for human selection
        
        # Initialize object indexing
        if self.enable_object_indexing:
            self.init_object_indexing()
        
    def build_agent(self):
        return None, None, None, None
    
    def predict_next_action(self, task):
        if self.state==RocAgent.STATE_OBSERVATION:
            # Initial state, transition to planning/thinking state based on observation and task
            pass
        if self.state==RocAgent.STATE_PLANNING:
            # After planning, transition to decision-making state
            pass
        if self.state==RocAgent.STATE_THINKING:
            # 1. After thinking, re-plan and transition to planning state
            # 2. After thinking, no planning needed, transition to decision-making state
            # 3. After thinking, task completed, transition to verification state
            pass
        if self.state==RocAgent.STATE_REFLECTION:
            # 1. After reflection, can make decision directly, transition to decision-making state
            # 2. After reflection, continue planning, transition to planning state
            pass
        if self.state==RocAgent.STATE_DECISION_MAKING_STATE:
            # 1. If decision fails, transition to reflection state
            # 2. If decision succeeds, transition to thinking state
            pass
        if self.state==RocAgent.STATE_VERIFICATION:
            # 1. After successful verification, transition to end state
            # 2. If verification fails, transition to reflection state
            pass
        if self.state==RocAgent.STATE_END:
            # End state
            pass
    
    # Move a few steps towards the target object
    def move_observation(self, target_item):
        # 1. Adjust agent's direction so target object is in front
        self.adjust_view(target_item)
        # 2. Calculate distance between object and agent, adjust agent's position
        distance = target_item["distance"]
        # 2. Move several steps towards target object
        # Move 1/3 of the distance
        self.action.action_mapping["moveAhead"](self.controller, round(distance/3, 1))
        self.update_event()
        # 3. Adjust agent's field of view
        self.adjust_agent_fieldOfView(120)
        self.update_event()
        self.update_legal_location()


    def init_agent_corner(self):
        scene_bounds2 = self.controller.last_event.metadata['sceneBounds']['cornerPoints'][2]
        scene_bounds3 = self.controller.last_event.metadata['sceneBounds']['cornerPoints'][3]
        scene_bounds6 = self.controller.last_event.metadata['sceneBounds']['cornerPoints'][6]
        scene_bounds7 = self.controller.last_event.metadata['sceneBounds']['cornerPoints'][7]

        # 3. Get agent's reachable positions
        event = self.controller.step(dict(action='GetReachablePositions'))
        reachable_positions = event.metadata['actionReturn']
        pre_target_positions = []
        # 4. Calculate closest reachable positions to the four corner points
        min_distance = float("inf")
        for i, scene_bounds in enumerate([scene_bounds2, scene_bounds3, scene_bounds6, scene_bounds7]):
            for position in reachable_positions:
                distance = math.sqrt((position['x']-scene_bounds[0])**2 + (position['z']-scene_bounds[2])**2)
                if distance < min_distance:
                    min_distance = distance
                    target_position = position
                    index = i
        # 5. Set agent's rotation angle
        if index == 0:
            # 180, 270
            target_rotation = dict(x=0, y=225, z=0)
        elif index == 1:
            # 270, 360
            target_rotation = dict(x=0, y=315, z=0)
        elif index == 2:
            # 90,180
            target_rotation = dict(x=0, y=135, z=0)
        else:
            # 0,90
            target_rotation = dict(x=0, y=45, z=0)
        
        # 6. Navigate agent to reachable position
        while True:
            event = self.action.action_mapping["teleport"](self.controller, position=target_position, rotation=target_rotation, horizon=0)
            self.update_event()
            if event.metadata['lastActionSuccess']:
                break
            else:
                pre_target_positions.append(target_position)
                event = self.controller.step(dict(action='GetReachablePositions'))
                reachable_positions = event.metadata['actionReturn']
                
                # 4. Calculate closest reachable positions to the four corner points
                min_distance = float("inf")
                for i, scene_bounds in enumerate([scene_bounds2, scene_bounds3, scene_bounds6, scene_bounds7]):
                    for position in reachable_positions:
                        if position in pre_target_positions:
                            continue
                        distance = math.sqrt((position['x']-scene_bounds[0])**2 + (position['z']-scene_bounds[2])**2)
                        if distance < min_distance:
                            min_distance = distance
                            target_position = position
                            index = i
                # 5. Set agent's rotation angle
                if index == 0:
                    # 180, 270
                    target_rotation = dict(x=0, y=225, z=0)
                elif index == 1:
                    # 270, 360
                    target_rotation = dict(x=0, y=315, z=0)
                elif index == 2:
                    # 90,180
                    target_rotation = dict(x=0, y=135, z=0)
                else:
                    # 0,90
                    target_rotation = dict(x=0, y=45, z=0)
                print("Teleport failed, retrying...")
        self.action.action_mapping["teleport"](self.controller, position=target_position, rotation=target_rotation, horizon=0)
        self.update_event()
        # self.save_frame({"action": "init_agent_view"}, prefix_save_path="./data/init_scene_image")
        # self.action.action_mapping["rotate_right"](self.controller, 30)
        # self.update_legal_location()
        # self.save_frame({"action": "init_view2"}, prefix_save_path="./data/init_scene_image")
        image_fp, legal_navigations, legal_interactions = None, None, None
        image_fp = self.save_frame({"step_count": str(self.step_count),
                                    "action": "init",},
                                    prefix_save_path=self.result_dir)
        legal_navigations = self.get_legal_navigations()
        legal_interactions = self.get_legal_interactions()
        return image_fp, legal_navigations, legal_interactions

    def navigate(self, itemtype):
        # Store GPT-4o reasoning from previous response if available
        if hasattr(self, '_last_vlm_response'):
            reasoning = self.extract_reasoning_from_response(self._last_vlm_response)
            self.set_gpt4o_reasoning(reasoning)
        
        # Record navigation action start
        if self._web_logging_enabled:
            self._log_interaction({
                'type': 'navigate',
                'action': f'Navigate to {itemtype}',
                'content': f'Start navigation to {itemtype}',
                'step': getattr(self, 'step_count', 0)
            })
            
        image_fp, legal_navigations, legal_interactions = None, None, None
    
        # ORIGINAL CODE - PRESERVED FOR A/B TESTING AND COMPARISON
        # if itemtype in self.target_item_type2obj_id:
        #     if self.taskid=="84" or self.taskid=="85":
        #         if self.controller.last_event.metadata["inventoryObjects"] == []:
        #             obj_id = self.target_item_type2obj_id[itemtype][0]
        #         else:
        #             obj_id = self.target_item_type2obj_id[itemtype][1]
        #     else:
        #         obj_id = self.target_item_type2obj_id[itemtype][0]
        #     item = self.eventobject.get_object_by_id(self.controller.last_event, obj_id)
        # else:
        #     item = self.objecttype2object[itemtype][0]
        
        # NEW ENHANCED CODE WITH DISAMBIGUATION SYSTEM
        # First check if indexed name is provided (e.g., "Sofa_1")
        if hasattr(self, 'objecttype2indexed') and itemtype in self.objecttype2indexed:
            item = self.objecttype2indexed[itemtype]
        else:
            # Check for multiple objects with position-based deduplication
            if itemtype in self.objecttype2object:
                objects = self.objecttype2object[itemtype]
                
                # Apply same deduplication logic as in indexing
                position_to_obj = {}
                for obj in objects:
                    pos_key = (round(obj['position']['x'], 2), round(obj['position']['z'], 2))
                    if pos_key not in position_to_obj:
                        position_to_obj[pos_key] = obj
                
                unique_objects = list(position_to_obj.values())
                
                if len(unique_objects) > 1 and hasattr(self, 'enable_dialogue_system') and self.enable_dialogue_system:
                    # ENHANCED (9.4): Use improved VLM-based dialogue for disambiguation
                    try:
                        item = self.request_user_disambiguation_improved(itemtype, unique_objects)
                        # Robust fallback if disambiguation fails
                        if item is None:
                            print(f"[WARNING] Disambiguation returned None for {itemtype}, using first object")
                            item = unique_objects[0]
                    except KeyError as e:
                        print(f"[ERROR] Missing key in disambiguation for {itemtype}: {e}")
                        print(f"  This is likely due to missing 'target_found' or 'image_path' in analysis")
                        item = unique_objects[0]
                    except Exception as e:
                        print(f"[ERROR] Unexpected disambiguation error for {itemtype}: {type(e).__name__}: {e}")
                        import traceback
                        traceback.print_exc()  # Print full traceback for debugging
                        item = unique_objects[0]
                elif len(unique_objects) == 1:
                    item = unique_objects[0]
                    if len(objects) > 1:
                        print(f"Deduplication: {len(objects)} {itemtype} objects at same position, using single representative")
                else:
                    # Fallback to first object if dialogue disabled
                    item = unique_objects[0] if unique_objects else objects[0]
                    if len(objects) > len(unique_objects):
                        print(f"Multiple {itemtype} found (deduplicated {len(objects)}→{len(unique_objects)}), using first one (dialogue disabled)")
            else:
                # Handle original target_item_type2obj_id logic
                if itemtype in self.target_item_type2obj_id:
                    if self.taskid=="84" or self.taskid=="85":
                        if self.controller.last_event.metadata["inventoryObjects"] == []:
                            obj_id = self.target_item_type2obj_id[itemtype][0]
                        else:
                            obj_id = self.target_item_type2obj_id[itemtype][1]
                    else:
                        obj_id = self.target_item_type2obj_id[itemtype][0]
                    item = self.eventobject.get_object_by_id(self.controller.last_event, obj_id)
                else:
                    print(f"Error: No objects of type {itemtype} found")
                    return None, None, None
        
        # CRITICAL SAFETY CHECK: Ensure item is always defined
        if 'item' not in locals() or item is None:
            print(f"CRITICAL: item is undefined for {itemtype}, attempting emergency fallback")
            if itemtype in self.objecttype2object and self.objecttype2object[itemtype]:
                item = self.objecttype2object[itemtype][0]
                print(f"[SUCCESS] Emergency fallback successful: using {item['objectType']}")
            else:
                print(f"[ERROR] Emergency fallback failed: no objects available")
                return None, None, None
        
        # Store itemtype - items not directly navigated to
        navigate_obj_type=item["objectType"]
        
        # If item is a container, if there are related objects on the container, and the container is not closed, navigate directly to that object "openable": 0, "isOpen": 0, 
        if item.get("receptacle", False) and (not item["openable"]):
            for related_object in self.related_objects:
                if related_object in item['receptacleObjectIds']:
                    item = self.eventobject.get_object_by_id(self.controller.last_event, related_object)
                    break
        
        # while(item['name'] == self.pre_navigate_location and len(self.objecttype2object[item['objectType']])>1):
        #     item = random.choice(self.objecttype2object[item['objectType']])
        # self.pre_navigate_location = item['name']
        # If container is not open and has target objects inside, cannot navigate directly to target objects
        if item["objectId"] in self.objid2position:
            target_position = self.objid2position[item["objectId"]]["agent_teleport_position"]
            target_rotation = self.objid2position[item["objectId"]]["agent_rotation"]
            horizon = self.objid2position[item["objectId"]]["agent_cameraHorizon"]
            print("Set position", self.objid2position)
        else:
            target_position, target_rotation = self.compute_position_8(item, pre_target_positions=[])
            horizon = 60
        # self.arm_reset()
        if target_position is None:
            print("teleport failed, no reachable positions")
            return image_fp, legal_navigations, legal_interactions
        event = self.action.action_mapping["teleport"](self.controller, position=target_position, rotation=target_rotation, horizon=horizon)
        # Check if successful
        pre_target_positions = []
        index = 0
        while not event.metadata['lastActionSuccess']:
            index += 1
            print(f"teleport failed, retrying...{index}")
            pre_target_positions.append(target_position)
            target_position, target_rotation = self.compute_position_8(item, pre_target_positions)
            event = self.action.action_mapping["teleport"](self.controller, position=target_position, rotation=target_rotation)
            self.update_event()
        
        if item["objectId"] not in self.objid2position:
            self.adjust_height(item)
            self.adjust_view(item)

        image_fp = self.save_frame({"step_count": str(self.step_count),
                                    "action": "navigate",
                                    "item": navigate_obj_type},
                                    prefix_save_path=self.result_dir)
        
        # Record navigation completion
        if self._web_logging_enabled:
            success = image_fp is not None
            self._log_interaction({
                'type': 'navigate',
                'action': f'Navigate to {navigate_obj_type} - {"Success" if success else "Failed"}',
                'content': f'Navigation to {navigate_obj_type} {"successful" if success else "failed"}',
                'image_path': image_fp if image_fp else '',
                'step': getattr(self, 'step_count', 0)
            })
        
        if item.get("receptacle", False) and "receptacleObjectIds" in item and (item['receptacleObjectIds'] != [] or item['receptacleObjectIds'] is not None):
            self.current_container = item
        
        legal_navigations = self.get_legal_navigations()
        legal_interactions = self.get_legal_interactions()

        # Check if multi-view observation is needed for large objects
        if self.enable_multi_view and hasattr(self, 'needs_multi_view_observation') and self.needs_multi_view_observation(item):
            print(f"[DEBUG] Large object detected: {item['objectType']}, switching to multi-view observation")
            return self.navigate_complete_view(itemtype)

        # self.update_legal_location()
        return image_fp, legal_navigations, legal_interactions
    
    def observe(self):
        image_fp, legal_navigations, legal_interactions = [], None, None
        for i in range(3):
            self.action.action_mapping["rotate_left"](self.controller, 90)
            
            image_fp.append(self.save_frame({"step_count": str(self.step_count),
                                        "i": str(i),
                                        "action": "observe"},
                                        prefix_save_path=self.result_dir))
            legal_navigations = self.get_legal_navigations()

        for i in range(3):
            images = [cv2.imread(path) for path in image_fp]
            img1 = add_text_to_image(images[0], "left view", (10, images[0].shape[0] - 20))
            img2 = add_text_to_image(images[1], "back view", (10, images[1].shape[0] - 20))
            img3 = add_text_to_image(images[2], "right view", (10, images[2].shape[0] - 25))
            # Add border to image (note: only add left and right borders to the middle image)
            img2_with_border = add_border(img2, 5, (0, 0, 0))
            # Horizontal concatenation
            img_h_concat = np.concatenate((img1, img2_with_border, img3), axis=1)
            # Save result
            output_path = self.save_frame({"step_count": str(self.step_count),
                                            # "i": str(i),
                                            "action": "observe"},
                                            prefix_save_path=self.result_dir)
            try:
                cv2.imwrite(output_path, img_h_concat)
                break
            except Exception as e:
                print("try_save_image")
                print(e)
        
        self.action.action_mapping["rotate_left"](self.controller, 90)
        legal_interactions = self.get_legal_interactions()
        
        return output_path, legal_navigations, legal_interactions
        
    def move_forward(self, distance=0.5):
        
        image_fp, legal_navigations, legal_interactions = None, None, None
        # Find which of the 8 surrounding directions has the most explorable positions
        # reachablePositions=self.controller.step(action="GetReachablePositions")
        
        # Turn back to 0
        # current_rotate=self.controller.last_event.metadata["agent"]["rotation"]["y"]
        # if current_rotate<0:
        #     self.action.action_mapping["rotate_right"](self.controller,degrees=abs(current_rotate))
        # if current_rotate>0:
        #     self.action.action_mapping["rotate_left"](self.controller,degrees=current_rotate)
        
        self.action.action_mapping["move_ahead"](self.controller, distance)
        print("RocAgent",self.controller.last_event)
        if self.controller.last_event.metadata["errorMessage"]=="":
            image_fp = self.save_frame({"step_count": str(self.step_count),
                                        "action": "move_forward"},
                                        prefix_save_path=self.result_dir)
            legal_navigations = self.get_legal_navigations()
            legal_interactions = self.get_legal_interactions()
            return image_fp, legal_navigations, legal_interactions
        else:
            # Left or right movement, random?
            # Based on which position is closer to the target object
            # import pdb;pdb.set_trace()
            if self.related_objects:
                distance_right_list = []
                distance_left_list = []
                
                # move_r_or_l=random.choice(["move_right","move_left"])
                self.action.action_mapping["move_right"](self.controller, distance)
                print("RocAgent",self.controller.last_event)
                errorMessage1=self.controller.last_event.metadata["errorMessage"]
                agentxright=self.controller.last_event.metadata["agent"]["position"]["x"]
                agentzright=self.controller.last_event.metadata["agent"]["position"]["z"]  

                if errorMessage1=="":
                    self.action.action_mapping["move_left"](self.controller, distance) # Return to original position
                    
                self.action.action_mapping["move_left"](self.controller, distance) # Move left
                print("RocAgent",self.controller.last_event)
                errorMessage2=self.controller.last_event.metadata["errorMessage"]
                agentxleft=self.controller.last_event.metadata["agent"]["position"]["x"]
                agentzleft=self.controller.last_event.metadata["agent"]["position"]["z"] 
                
                if errorMessage2=="":
                    self.action.action_mapping["move_right"](self.controller, distance) # Return to original position
                
                for obj_id in self.related_objects:
                    item = self.eventobject.get_object_by_id(self.controller.last_event,obj_id)
                    if item["visible"]==True:
                        itemx=item["position"]["x"]
                        itemz=item["position"]["z"]
                        
                        # Calculate distance after moving right
                        distance_right = math.sqrt((agentxright - itemx) ** 2 + (agentzright - itemz) ** 2)
                        distance_right_list.append(distance_right)
                        # Calculate distance after moving left
                        distance_left = math.sqrt((agentxleft - itemx) ** 2 + (agentzleft - itemz) ** 2)
                        distance_left_list.append(distance_left)
                   
                if errorMessage1=="" and errorMessage2=="" and distance_right_list and distance_left_list: # Both left and right can move, choose the direction that gets closest to target object after moving
                    # 1. Choose direction with smallest average distance to all target objects
                    # avg_distance_right = sum(distance_right_list) / len(distance_right_list)
                    # avg_distance_left = sum(distance_left_list) / len(distance_left_list)
                    # if avg_distance_right < avg_distance_left:
                    #     direction = "move_right"
                    # else:
                    #     direction = "move_left"
                    
                    # 2. Choose direction that minimizes distance to closest object
                    
                    min_distance_right = min(distance_right_list)
                    min_distance_left = min(distance_left_list)
                    
                    if min_distance_right < min_distance_left:
                        direction = "move_right"
                    else:
                        direction = "move_left"
                    
                    # After moving to direction side, distance to closest among n target objects 
                    self.action.action_mapping[direction](self.controller, distance)
                    if self.controller.last_event.metadata["errorMessage"]=="":
                        image_fp = self.save_frame({"step_count": str(self.step_count),
                                                "action": "move_forward"},
                                                prefix_save_path=self.result_dir)
                        legal_navigations = self.get_legal_navigations()
                        legal_interactions = self.get_legal_interactions()
                        return image_fp, legal_navigations, legal_interactions  
                    
                elif errorMessage1=="" or errorMessage2=="":  # One of left/right directions can move, choose the movable direction
                    if errorMessage1=="":
                        self.action.action_mapping["move_right"](self.controller, distance)
                        
                    elif errorMessage2=="":
                        self.action.action_mapping["move_left"](self.controller, distance)
                    
                    print("RocAgent",self.controller.last_event)                  
                    if self.controller.last_event.metadata["errorMessage"]=="":
                        image_fp = self.save_frame({"step_count": str(self.step_count),
                                                "action": "move_forward"},
                                                prefix_save_path=self.result_dir)
                        legal_navigations = self.get_legal_navigations()
                        legal_interactions = self.get_legal_interactions()
                        return image_fp, legal_navigations, legal_interactions
                
                else:
                    self.action.action_mapping["move_back"](self.controller, distance)  # Move backward
                    print("RocAgent",self.controller.last_event)
                    if self.controller.last_event.metadata["errorMessage"]=="":
                        image_fp = self.save_frame({"step_count": str(self.step_count),
                                        "action": "move_forward"},
                                        prefix_save_path=self.result_dir)
                        legal_navigations = self.get_legal_navigations()
                        legal_interactions = self.get_legal_interactions()
                        return image_fp, legal_navigations, legal_interactions
                    
                    else:
                        self.action.action_mapping["rotate_right"](self.controller,degrees=90)
                        errorMessage_rotate_right=self.controller.last_event.metadata["errorMessage"]
                        self.action.action_mapping["move_ahead"](self.controller, distance)
                        print("RocAgent",self.controller.last_event)
                        if self.controller.last_event.metadata["errorMessage"]=="":
                            image_fp = self.save_frame({"step_count": str(self.step_count),
                                        "action": "move_forward"},
                                        prefix_save_path=self.result_dir)
                            legal_navigations = self.get_legal_navigations()
                            legal_interactions = self.get_legal_interactions()
                            return image_fp, legal_navigations, legal_interactions
                        else:
                            if errorMessage_rotate_right=="": # Turn left
                                self.action.action_mapping["rotate_left"](self.controller,degrees=180)
                            self.action.action_mapping["move_ahead"](self.controller, distance)
                            print("RocAgent",self.controller.last_event)
                            if self.controller.last_event.metadata["errorMessage"]=="":
                                image_fp = self.save_frame({"step_count": str(self.step_count),
                                        "action": "move_forward"},
                                        prefix_save_path=self.result_dir)
                                legal_navigations = self.get_legal_navigations()
                                legal_interactions = self.get_legal_interactions()
                                return image_fp, legal_navigations, legal_interactions
                    
            else:
                self.action.action_mapping["move_right"](self.controller, distance)
                                    
                if self.controller.last_event.metadata["errorMessage"]=="":
                    image_fp = self.save_frame({"step_count": str(self.step_count),
                                            "action": "move_forward"},
                                            prefix_save_path=self.result_dir)
                    legal_navigations = self.get_legal_navigations()
                    legal_interactions = self.get_legal_interactions()
                    return image_fp, legal_navigations, legal_interactions

                else:
                    self.action.action_mapping["move_left"](self.controller, distance)
                                        
                    if self.controller.last_event.metadata["errorMessage"]=="":
                        image_fp = self.save_frame({"step_count": str(self.step_count),
                                                "action": "move_forward"},
                                                prefix_save_path=self.result_dir)
                        legal_navigations = self.get_legal_navigations()
                        legal_interactions = self.get_legal_interactions()
                        return image_fp, legal_navigations, legal_interactions
                    else:
                        # # Left movement
                        # self.action.action_mapping["move_left"](self.controller, distance)
                        # print("RocAgent",self.controller.last_event)
                        # if self.controller.last_event.metadata["errorMessage"]=="":
                        #     image_fp = self.save_frame({"step_count": str(self.step_count),
                        #                             "action": "move_forward"},
                        #                             prefix_save_path=self.result_dir)
                        #     legal_navigations = self.get_legal_navigations()
                        #     legal_interactions = self.get_legal_interactions()
                        #     return image_fp, legal_navigations, legal_interactions
                        self.action.action_mapping["move_back"](self.controller, distance)  # Move backward
                        print("RocAgent",self.controller.last_event)
                        if self.controller.last_event.metadata["errorMessage"]=="":
                            image_fp = self.save_frame({"step_count": str(self.step_count),
                                            "action": "move_forward"},
                                            prefix_save_path=self.result_dir)
                            legal_navigations = self.get_legal_navigations()
                            legal_interactions = self.get_legal_interactions()
                            return image_fp, legal_navigations, legal_interactions
                        
                        else:
                            self.action.action_mapping["rotate_right"](self.controller,degrees=90)
                            errorMessage_rotate_right=self.controller.last_event.metadata["errorMessage"]
                            self.action.action_mapping["move_ahead"](self.controller, distance)
                            print("RocAgent",self.controller.last_event)
                            if self.controller.last_event.metadata["errorMessage"]=="":
                                image_fp = self.save_frame({"step_count": str(self.step_count),
                                            "action": "move_forward"},
                                            prefix_save_path=self.result_dir)
                                legal_navigations = self.get_legal_navigations()
                                legal_interactions = self.get_legal_interactions()
                                return image_fp, legal_navigations, legal_interactions
                            else:
                                if errorMessage_rotate_right=="": # Turn left
                                    self.action.action_mapping["rotate_left"](self.controller,degrees=180)
                                self.action.action_mapping["move_ahead"](self.controller, distance)
                                print("RocAgent",self.controller.last_event)
                                if self.controller.last_event.metadata["errorMessage"]=="":
                                    image_fp = self.save_frame({"step_count": str(self.step_count),
                                            "action": "move_forward"},
                                            prefix_save_path=self.result_dir)
                                    legal_navigations = self.get_legal_navigations()
                                    legal_interactions = self.get_legal_interactions()
                                    return image_fp, legal_navigations, legal_interactions
                    
        print("RocAgent",self.controller.last_event)
        return image_fp, legal_navigations, legal_interactions

    def pick_up(self, itemtype):
        
        if itemtype in self.target_item_type2obj_id:
            obj_id = self.target_item_type2obj_id[itemtype][0]
            item = self.eventobject.get_object_by_id(self.controller.last_event, obj_id)
        else:
            item = self.objecttype2object[itemtype][0]
        
        image_fp, legal_navigations, legal_interactions = None, None, None
        self.action.action_mapping["pick_up"](self.controller, item['objectId'])
        image_fp = self.save_frame({"step_count": str(self.step_count),
                                    "action": "pick_up",
                                    "item": item["objectType"]},
                                    prefix_save_path=self.result_dir)
        legal_navigations = self.get_legal_navigations()
        legal_interactions = self.get_legal_interactions()
        return image_fp, legal_navigations, legal_interactions

    def put_in(self, itemtype):
        if itemtype in self.target_item_type2obj_id:
            obj_id = self.target_item_type2obj_id[itemtype][0]
            item = self.eventobject.get_object_by_id(self.controller.last_event, obj_id)
        else:
            item = self.objecttype2object[itemtype][0]
        
        image_fp, legal_navigations, legal_interactions = None, None, None
        self.action.action_mapping["put_in"](self.controller, item['objectId'])
        image_fp = self.save_frame({"step_count": str(self.step_count),
                                    "action": "put_in",
                                    "item": item["objectType"]},
                                    prefix_save_path=self.result_dir)
        legal_navigations = self.get_legal_navigations()
        legal_interactions = self.get_legal_interactions()
        return image_fp, legal_navigations, legal_interactions

    def toggle(self, itemtype):
        if itemtype in self.target_item_type2obj_id:
            obj_id = self.target_item_type2obj_id[itemtype][0]
            item = self.eventobject.get_object_by_id(self.controller.last_event, obj_id)
        else:
            item = self.objecttype2object[itemtype][0]
        # 本身是打开的状态
        if item["isToggled"]==True:
            image_fp, legal_navigations, legal_interactions = None, None, None
            self.action.action_mapping["toggle_off"](self.controller, item['objectId'])
            image_fp = self.save_frame({"step_count": str(self.step_count),
                                    "action": "toggle",
                                    "item": item["objectType"]},
                                    prefix_save_path=self.result_dir)
            legal_navigations = self.get_legal_navigations()
            legal_interactions = self.get_legal_interactions()
            return image_fp, legal_navigations, legal_interactions
        else:
            image_fp, legal_navigations, legal_interactions = None, None, None
            self.action.action_mapping["toggle_on"](self.controller, item['objectId'])
            image_fp = self.save_frame({"step_count": str(self.step_count),
                                    "action": "toggle",
                                    "item": item["objectType"]},
                                    prefix_save_path=self.result_dir)
            legal_navigations = self.get_legal_navigations()
            legal_interactions = self.get_legal_interactions()
            return image_fp, legal_navigations, legal_interactions

    def open(self, itemtype):
        if itemtype in self.target_item_type2obj_id:
            obj_id = self.target_item_type2obj_id[itemtype][0]
            item = self.eventobject.get_object_by_id(self.controller.last_event, obj_id)
        else:
            item = self.objecttype2object[itemtype][0]
        
        image_fp, legal_navigations, legal_interactions = None, None, None
        self.action.action_mapping["open"](self.controller, item['objectId'])
        image_fp = self.save_frame({"step_count": str(self.step_count),
                                    "action": "open",
                                    "item": item["objectType"]},
                                    prefix_save_path=self.result_dir)
        legal_navigations = self.get_legal_navigations()
        legal_interactions = self.get_legal_interactions()
        return image_fp, legal_navigations, legal_interactions
    
    def close(self, itemtype):
        if itemtype in self.target_item_type2obj_id:
            obj_id = self.target_item_type2obj_id[itemtype][0]
            item = self.eventobject.get_object_by_id(self.controller.last_event, obj_id)
        else:
            item = self.objecttype2object[itemtype][0]
        
        image_fp, legal_navigations, legal_interactions = None, None, None
        self.action.action_mapping["close"](self.controller, item['objectId'])
        image_fp = self.save_frame({"step_count": str(self.step_count),
                                    "action": "close",
                                    "item": item["objectType"]},
                                    prefix_save_path=self.result_dir)
        legal_navigations = self.get_legal_navigations()
        legal_interactions = self.get_legal_interactions()
        return image_fp, legal_navigations, legal_interactions


    def get_all_item_image(self):
        res = []
        for item in tqdm(self.eventobject.get_objects(self.controller.last_event)[0]):
            # print(item["name"],self.eventobject.get_item_surface_area(item['name']))
            if item["name"] == "DiningTable_806ce8fd":#Book_e173324d Box_8e5b2c6b CellPhone_b8be2958
            # # print(item["name"],":",round(item["rotation"]['y']))
                succeess, _ ,_ = self.navigate(item)
                self.save_frame({"item": item["name"]})
                dic = {
                    "scene": self.scene,
                    "item": item["name"],
                    "agent":{
                        "agentMode": "arm",
                        "position": self.get_agent_position(),
                        "rotation": self.get_agent_rotation(),
                    },
                    "camera":{
                        "position": self.get_camera_position(),
                        "rotation": self.get_camera_rotation(),
                    },
                    "fieldOfView":90,
                    "gridSize":0.1,
                    "visibilityDistance": 10,
                    "image_path": f"./data/item_image/{self.scene}_{item['name']}.png"
                }
                res.append(dic)
                self.controller.reset(self.scene)
        with open(f"./data/{self.scene}_objects.jsonl", "w") as f:
            import json
            for item in res:
                f.write(json.dumps(item, ensure_ascii=False)+"\n")

    def get_navigate_path(self):
        res = []
        
        for item in tqdm(self.eventobject.get_objects(self.controller.last_event)[0]):
            # print(item["name"],self.eventobject.get_item_surface_area(item['name']))
            # if item["name"] == "DiningTable_806ce8fd":#Book_e173324d Box_8e5b2c6b CellPhone_b8be2958
            # # print(item["name"],":",round(item["rotation"]['y']))
            # if item["name"] in self.legal_location:
                import copy
                legal_location = copy.deepcopy(self.legal_location)
                succeess, _ ,_ = self.navigate(item)
                visible_objects = []
                obj_names, objs = self.eventobject.get_visible_objects(self.controller.last_event)
                for obj_name, obj in zip(obj_names, objs):
                    volume = self.eventobject.get_item_volume(self.controller.last_event, obj_name)
                    if volume <= 0.1:
                        if obj["distance"] <= 1.5:
                            visible_objects.append(obj["name"])
                    elif volume <= 0.5:
                        if obj["distance"] <= 2.5:
                            visible_objects.append(obj["name"])
                    elif volume <= 1:
                        if obj["distance"] <= 5.0:
                            visible_objects.append(obj["name"])
                    else:
                        if obj["distance"] <= 10.0:
                            visible_objects.append(obj["name"])
                for obj_name in visible_objects:
                    if obj_name not in legal_location.keys():
                        legal_location[obj_name] = 1
                    else:
                        legal_location[obj_name] += 1

                self.save_frame({"item": item["name"]})
                dic = {
                    "scene": self.scene,
                    "init_legal_location": legal_location,
                    "object": {
                        "name":item["name"],
                        "position": item["position"],
                        "rotation": item["rotation"],
                    },
                    "agent":{
                        "agentMode": "default",
                        "position": self.get_agent_position(),
                        "rotation": self.get_agent_rotation(),
                    },
                    "camera":{
                        "position": self.get_camera_position(),
                        "rotation": self.get_camera_rotation(),
                    },
                    "fieldOfView":90,
                    "gridSize":0.1,
                    "visibilityDistance": 10,
                    "image_path": f"./data/item_image/{self.scene}_{item['name']}.png"
                }
                res.append(dic)
                self.controller.reset(self.scene)

        with open(f"./data/{self.scene}_objects.jsonl", "w") as f:
            import json
            for line in res:
                f.write(json.dumps(line, ensure_ascii=False)+"\n")

    def example(self):
        for item in tqdm(self.eventobject.get_objects(self.controller.last_event)[0]):
            if item["name"] == "DiningTable_0beb798c": # Book_e173324d Box_8e5b2c6b CellPhone_b8be2958
                self.navigate(item)
                self.move_observation(item)
                self.adjust_agent_fieldOfView(150)
                self.save_frame({"item": item["name"], "action": "pick_up"})
        
        pass
    
    def test_visibility(self):
        self.init()
        volumes = {
            "0-0.01": [],
            "0.01-0.05": [],
            "0.05-0.1": [],
            "0.1-0.5": [],
            "0.5-1.0": [],
            "1.0+": []
        }
        import json
        with open("./data/visible_objects_f.jsonl") as f:
            data = [json.loads(line) for line in f.readlines()]
        visible_objects = []
        for d in data:
            visible_objects.extend(d["visible_objects"])
            for obj in d["objects"]:
                if not obj["name"].startswith("Floor_"):
                    volume = obj.get("volum")  # Use get method to avoid KeyError
                    if volume is not None:  # Ensure volume exists
                        if 0 <= volume < 0.01:
                            volumes["0-0.01"].append(obj["name"])   
                        elif 0.01 <= volume < 0.05: 
                            volumes["0.01-0.05"].append(obj["name"]) 
                        elif 0.05 <= volume < 0.1:
                            volumes["0.05-0.1"].append(obj["name"])
                        elif 0.1 <= volume < 0.5:
                            volumes["0.1-0.5"].append(obj["name"])
                        elif 0.5 <= volume < 1.0:
                            volumes["0.5-1.0"].append(obj["name"])
                        else:
                            volumes["1.0+"].append(obj["name"])
        
        with open("./data/visible_objects.jsonl", "a") as f:
            import json
            visible_objects = []
            item_names, items = self.eventobject.get_visible_objects(self.controller.last_event)
            for item_name, item in zip(item_names, items):
                volume = self.eventobject.get_item_volume(self.controller.last_event, item_name)
                surface_area = self.eventobject.get_item_surface_area(self.controller.last_event, item_name)
                if volume <= 0.1:
                    if item["distance"] <= 1.5:
                        visible_objects.append(item["name"])
                    elif surface_area > 1:
                        visible_objects.append(item["name"])
                elif volume <= 0.5:
                    if item["distance"] <= 2.5:
                        visible_objects.append(item["name"])
                    elif surface_area > 1:
                        visible_objects.append(item["name"])
                elif volume <= 1:
                    if item["distance"] <= 5.0:
                        visible_objects.append(item["name"])
                    elif surface_area > 1:
                        visible_objects.append(item["name"])
                else:
                    if item["distance"] <= 10.0:
                        visible_objects.append(item["name"])
                        
            dic = {
                "scene": self.scene,
                "objects":[
                    {"name": item["name"], 
                     "visible": item["visible"],
                     "volum": self.eventobject.get_item_volume(self.controller.last_event, item['name']),
                     "surface_area": self.eventobject.get_item_surface_area(self.controller.last_event, item['name']),
                     "distance": item["distance"],
                    }
                for item in self.eventobject.get_objects(self.controller.last_event)[0]
                ],
                "visible_objects": visible_objects,
            }
            f.write(json.dumps(dic, ensure_ascii=False)+"\n")
    
    def get_navigate_location(self):
        metadata = self.controller.last_event.metadata
        volumes = []
        objectid2object={}
        for obj in metadata["objects"]:
            objectid2object[obj["objectId"]]=obj
            if obj["objectType"]!="Floor":#去掉地板
                size=obj["axisAlignedBoundingBox"]["size"]
                # print(size)
                v=size["x"]*size["y"]*size["z"]
                dx=obj["axisAlignedBoundingBox"]["center"]["x"]
                dz=obj["axisAlignedBoundingBox"]["center"]["z"]
                agentx=metadata["agent"]["position"]["x"]
                agentz=metadata["agent"]["position"]["z"]
                d=math.sqrt((dx-agentx)**2+(dz-agentz)**2)
                
                # 计算横面积
                sxz = size["x"] * size["z"]

                # 计算纵向1面积
                sxy = size["x"] * size["y"]

                # 计算纵向2面积
                szy = size["y"] * size["z"]

                # 选择最大的面积作为 s
                s = max(sxz, sxy, szy)
                
                # 计算体积与距离的比率
                if d != 0:  # 防止除以零
                    rate = v / d
                else:
                    rate = 0
                    print(obj["objectId"],"d=0")  
                rate=v/d
                isnavigable=False
                if obj["visible"]==True:#没有被挡住的情况下
                    if v<0.01:#物体体积很小
                        isnavigable=False
                        #体积虽然小，但面积较大，且距离足够近
                        # 1. s>0.5 10米内
                        # 2. s>0.15 4米内
                        # 3. s>0.08 2.5米内
                        # 4. v>0.005 2米内
                        if s>0.5 and d<10:
                            isnavigable=True
                        elif s>0.15 and d<4:
                            isnavigable=True
                        elif s>0.08 and d<2.5:
                            isnavigable=True 
                        elif v>0.005 and d<2:
                            isnavigable=True 
                        elif v>0.001 and d<1.5:
                            isnavigable=True
                        elif d<1: # Volume is small but distance is close enough
                            isnavigable=True
                    else:
                        isnavigable=True
                        if rate<=0.02: # Volume is large but distance is too far v/d
                            isnavigable=False
                            if s>0.5 and d<10: # Exclude area influence
                                isnavigable=True
                            elif s>0.15 and d<4:
                                isnavigable=True
                            elif s>0.08 and d<2.5:
                                isnavigable=True 
                            elif v>0.005 and d<2:
                                isnavigable=True 
                            elif v>0.001 and d<1.5:
                                isnavigable=True
                            elif d<1: # Volume is small but distance is close enough
                                isnavigable=True
                                
                volumes.append({
                    "objectId":obj["objectId"],
                    "objectType":obj["objectType"],
                    "visible":obj["visible"],
                    "volume":v,
                    "s":s,
                    "distance":d,
                    "rate":rate,
                    "isnavigable":isnavigable
                })
                sorted_volumes = sorted(volumes, key=lambda v: v["rate"])

        res = {}
        for item in sorted_volumes:
            res[item["objectId"]] = item
        #     if item['isnavigable']:
        #         if item["objectType"] not in self.objecttype2object:
        #             self.objecttype2object[item["objectType"]] = [objectid2object[item["objectId"]]]
        #         else:
        #             itemname = objectid2object[item["objectId"]]['name']
        #             if itemname not in [obj['name'] for obj in self.objecttype2object[item["objectType"]]]:
        #                 self.objecttype2object[item["objectType"]].append(objectid2object[item["objectId"]])
                
        return res
    
    # Global reachable positions
    def get_legal_navigations(self):
        objects = self.get_navigate_location()
        for objectId, obj in objects.items():
            if obj["isnavigable"]:
                if obj["objectType"] not in self.navigable_objects:
                    self.navigable_objects[obj["objectType"]] = 0
                self.navigable_objects[obj["objectType"]] += 1
        
        return list(self.navigable_objects.keys())

    def get_current_container_obj(self):
        if self.current_container is not None:
            objects = [obj.split("|")[0] for obj in self.current_container["receptacleObjectIds"]]
            # print(objects)
            return objects
        else:
            return []

    # Global interactive positions
    def get_legal_interactions(self):
        legal_interactions = {}
        objects = self.get_navigate_location()
        for objectId, obj in objects.items():
            if (obj["visible"] and obj["objectType"] in self.get_current_container_obj()) or obj["isnavigable"]:
                if obj["objectType"] not in legal_interactions:
                    legal_interactions[obj["objectType"]] = 0
                legal_interactions[obj["objectType"]] += 1
        
        self.legal_interactions = legal_interactions
        return list(self.legal_interactions.keys())

    def action_meta(self, navigate_locations, item, action="obervation"):
        if action =="init":
            self.init_agent_corner()
            navigate_location = self.get_navigate_location()
            for k, item in navigate_location.items():
                if item["objectId"] not in navigate_locations:
                    navigate_locations[item["objectId"]] = item
        
        elif action == "obervation":
            for i in range(3):
                self.action.action_mapping["rotate_left"](self.controller, 90)
                navigate_location = self.get_navigate_location()
                for k, item in navigate_location.items():
                    if item["objectId"] not in navigate_locations:
                        navigate_locations[item["objectId"]] = item
        
        elif action == "navigate":
            self.navigate(item)
            navigate_location = self.get_navigate_location()
            for k, item in navigate_location.items():
                if item["objectId"] not in navigate_locations:
                    navigate_locations[item["objectId"]] = item
        
        elif action == "move":
            self.move_forward(0.5)
            navigate_location = self.get_navigate_location()
            for k, item in navigate_location.items():
                if item["objectId"] not in navigate_locations:
                    navigate_locations[item["objectId"]] = item
        
        
        return navigate_locations, navigate_location
    
    def exec(self, action, item=None):
        # for itemtype in self.eventobject.get_objects_type(self.controller.last_event):
        #     if itemtype not in self.navigable_objects:
        #         self.navigable_objects[itemtype] = 0
        #     self.navigable_objects[itemtype] += 1
        #     if itemtype not in self.legal_interactions:
        #         self.legal_interactions[itemtype] = 0
        #     self.legal_interactions[itemtype] += 1
        self.navigable_objects
        image_fp, legal_locations, legal_objects = None, list(self.navigable_objects.keys()), list(self.legal_interactions.keys())
        # image_fp, legal_locations, legal_objects = None, self.eventobject.get_objects_type(self.controller.last_event), list(self.legal_interactions.keys())
        self.step_count += 1
        for action_name in self.action_space:
            if action_name in action:
                if action_name == "observe" or action_name == "init":
                    image_fp, legal_locations, legal_objects = self.action_space[action_name]()
                    if self.controller.last_event.metadata["errorMessage"]!="":
                        success=False
                    else:
                        success=True
                    return success, image_fp, legal_locations, legal_objects
                elif action_name == "move forward":
                    image_fp, legal_locations, legal_objects = self.action_space[action_name](distance=0.5)
                    if self.controller.last_event.metadata["errorMessage"]!="":
                        success=False
                    else:
                        success=True
                    return success, image_fp, legal_locations, legal_objects
                else:
                    if item is None:
                        return False, None, list(self.navigable_objects.keys()), list(self.legal_interactions.keys())
                    else:
                        # Navigation action
                        if action_name == "navigate to" and item in self.navigable_objects:
                            image_fp, legal_locations, legal_objects = self.action_space[action_name](item)
                            return True, image_fp, legal_locations, legal_objects
                        # Interactive action # "put in" for MODE=API
                        if action_name in ["pickup", "put", "put in","toggle", "open", "close"] and item in self.legal_interactions:
                            image_fp, legal_locations, legal_objects = self.action_space[action_name](item)
                            if self.controller.last_event.metadata["errorMessage"]!="":
                                success=False
                            else:
                                success=True
                            return success, image_fp, legal_locations, legal_objects
                
        
        return False, image_fp, legal_locations, legal_objects

    # ======= ENHANCED NAVIGATION: Multi-object Disambiguation System =======
    
    def init_object_indexing(self):
        """Create indexed mapping for duplicate objects with position-based deduplication"""
        self.objecttype2indexed = {}
        
        for obj_type, objects in self.objecttype2object.items():
            if len(objects) > 1:
                # Deduplicate objects at the same position (AI2-THOR bug workaround)
                position_to_obj = {}
                for obj in objects:
                    # Round position to handle floating point precision issues
                    pos_key = (round(obj['position']['x'], 2), round(obj['position']['z'], 2))
                    if pos_key not in position_to_obj:
                        position_to_obj[pos_key] = obj
                
                # Sort deduplicated objects by position for consistent ordering
                unique_objects = list(position_to_obj.values())
                sorted_objs = sorted(unique_objects, key=lambda o: (o['position']['x'], o['position']['z']))
                
                # Only index if we still have multiple unique positions after deduplication
                if len(sorted_objs) > 1:
                    for i, obj in enumerate(sorted_objs):
                        indexed_name = f"{obj_type}_{i+1}"
                        self.objecttype2indexed[indexed_name] = obj
                        print(f"Object indexing: {indexed_name} at position ({obj['position']['x']:.2f}, {obj['position']['z']:.2f})")
                else:
                    # All objects were at same position - just use the first one
                    print(f"Deduplication: {len(objects)} {obj_type} objects collapsed to 1 unique position")
            else:
                # Single object - still add to indexed mapping for consistency
                obj = objects[0]
                indexed_name = f"{obj_type}_1"
                self.objecttype2indexed[indexed_name] = obj
    
    def generate_spatial_description(self, obj, idx, all_objects):
        """Generate human-readable spatial description"""
        descriptions = []
        
        # Basic position description
        if obj['position']['x'] < 0:
            descriptions.append("on the left side of the room")
        else:
            descriptions.append("on the right side of the room")
        
        # Find nearby landmarks
        nearby_landmarks = self.find_nearby_landmarks(obj)
        if nearby_landmarks:
            descriptions.append(f"near the {nearby_landmarks[0]}")
        
        return ", ".join(descriptions)
    
    def find_nearby_landmarks(self, obj, radius=2.0):
        """Find nearby landmark objects"""
        landmarks = []
        for other in self.controller.last_event.metadata['objects']:
            if other['objectType'] in ['Window', 'Door', 'Sink', 'Stove']:
                distance = ((obj['position']['x'] - other['position']['x'])**2 + 
                           (obj['position']['z'] - other['position']['z'])**2)**0.5
                if distance < radius:
                    landmarks.append(other['objectType'].lower())
        return landmarks
    
    def vlm_call_with_logging(self, image_path, prompt, analysis_type="general"):
        """Enhanced VLM call with comprehensive logging and monitoring"""
        import time
        start_time = time.time()
        
        # Preprocess log record
        log_data = {
            'type': 'vlm_analysis',
            'analysis_type': analysis_type,
            'image_path': image_path,
            'prompt_length': len(prompt),
            'prompt_preview': prompt[:200] + "..." if len(prompt) > 200 else prompt,
            'full_prompt': prompt,  # Complete prompt for debugging
            'step': getattr(self, 'step_count', 0),
            'start_time': start_time,
            'timestamp': time.strftime('%H:%M:%S')
        }
        
        # print(f"Prompt Preview: {log_data['prompt_preview']}")
        
        if hasattr(self, '_log_vlm_call') and self._log_vlm_call:
            self._log_vlm_call(log_data)
        
        try:
            response = self._execute_vlm_request(image_path, prompt)
            
            end_time = time.time()
            duration = end_time - start_time
            
            log_data.update({
                'success': True,
                'response_length': len(response),
                'response_preview': response[:200] + "..." if len(response) > 200 else response,
                'full_response': response,  # Complete response for debugging
                'duration': round(duration, 2),
                'end_time': end_time,
                'timestamp': time.strftime('%H:%M:%S')
            })
            
            # print(f"VLM analysis completed - duration {duration:.2f}s")
            # print(f"Response preview: {log_data['response_preview']}")
            
            # Send completion log to Dashboard
            if hasattr(self, '_log_vlm_call') and self._log_vlm_call:
                self._log_vlm_call(log_data)
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            
            log_data.update({
                'success': False,
                'error': error_msg,
                'duration': round(duration, 2),
                'timestamp': time.strftime('%H:%M:%S')
            })
            
            print(f"VLM analysis failed - {error_msg}")
            
            # Send error log to Dashboard
            if hasattr(self, '_log_vlm_call') and self._log_vlm_call:
                self._log_vlm_call(log_data)
            
            # Return structured failure response
            return f"Reasoning: VLM analysis failed due to technical error: {error_msg}\nConfidence: 25"

    def _execute_vlm_request(self, image_path, prompt):
        """Pure VLM execution without logging - core functionality with retry"""
        import sys
        import os
        import time
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from VLMCall import VLMAPI
        
        vlm = VLMAPI("Qwen/Qwen2-VL-7B-Instruct")
        
        img_url = vlm.encode_image_2(image_path)
        messages = [
            {"role": "system", "content": "You are a household navigation agent analyzing scenes for task completion."},
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": img_url}}
                ]
            }
        ]
        
        # Three retry mechanism with smart error handling
        for attempt in range(3):
            try:
                if attempt > 0:
                    print(f"Retry: {attempt+1} times)")
                    time.sleep(2)
                return vlm.vlm_request(messages)
            except Exception as e:
                # Check if it's a 400 error (client error) - don't retry
                error_str = str(e)
                if "400" in error_str or "invalid_parameter" in error_str.lower():
                    print(f"VLM analysis failed with client error (400) - not retrying: {e}")
                    raise e
                    
                # For other errors, retry up to 3 times
                if attempt == 2:
                    print(f"VLM anaylsis failed after 3 attempts - {e}")
                    raise e
    
    # Maintain backward compatible old interface
    def vlm_call(self, image_path, prompt):
        """Legacy VLM call interface - redirects to new architecture"""
        return self.vlm_call_with_logging(image_path, prompt, "legacy_call")
    
    def set_gpt4o_reasoning(self, reasoning_text):
        """Store GPT-4o's reasoning for use in disambiguation"""
        self.current_gpt4o_reasoning = reasoning_text
        print(f"[GPT-4O REASONING] {reasoning_text}")

    def extract_reasoning_from_response(self, response):
        """Extract reasoning text before <DecisionMaking> tag"""
        if '<DecisionMaking>' in response:
            reasoning = response.split('<DecisionMaking>')[0].strip()
            return reasoning
        return response.strip()
    
    def navigate_to_object_for_analysis(self, obj):
        """Navigate to specific object for VLM analysis"""
        view_position = self.calculate_analysis_position(obj)
        
        if view_position is None:
            print(f"  -> No reachable position found for {obj['objectType']}")
            return False
        
        event = self.action.action_mapping["teleport"](
            self.controller,
            position=view_position['position'],
            rotation=view_position['rotation'],
            horizon=view_position.get('horizon', 60)  # Use 60-degree overhead view
        )

        if event.metadata['lastActionSuccess']:
            # Apply vision optimization like standard navigation
            self.adjust_height(obj)
            self.adjust_view(obj)

        return event.metadata['lastActionSuccess']

    def return_to_position(self, original_position):
        """Return agent to original position after VLM analysis"""
        event = self.action.action_mapping["teleport"](
            self.controller,
            position=original_position['position'],
            rotation=original_position['rotation'],
            horizon=original_position['horizon']
        )
        
        if not event.metadata['lastActionSuccess']:
            print("[WARNING] Failed to return to original position!")

    def calculate_analysis_position(self, obj):
        """Calculate optimal viewing position for object analysis"""
        obj_pos = obj['position']
        
        # Get all reachable positions
        event = self.controller.step(dict(action='GetReachablePositions'))
        reachable_positions = event.metadata['actionReturn']
        
        # Find closest reachable position to the object
        min_distance = float('inf')
        best_position = None
        
        for pos in reachable_positions:
            distance = math.sqrt(
                (pos['x'] - obj_pos['x'])**2 + 
                (pos['z'] - obj_pos['z'])**2
            )
            if distance < min_distance:
                min_distance = distance
                best_position = pos
        
        if best_position is None:
            return None
            
        # Calculate rotation to face the object
        dx = obj_pos['x'] - best_position['x']
        dz = obj_pos['z'] - best_position['z']
        angle = math.degrees(math.atan2(dx, dz))
        
        analysis_pos = {
            'position': best_position,
            'rotation': {'x': 0, 'y': angle, 'z': 0},
            'horizon': 0
        }
        
        return analysis_pos
    
    def take_candidate_photos_only(self, task_description, candidates):
        """Navigate to ALL candidates and take photos without VLM analysis"""
        analyses = []
        
        # Store original position to return to
        original_position = {
            'position': self.controller.last_event.metadata['agent']['position'],
            'rotation': self.controller.last_event.metadata['agent']['rotation'],
            'horizon': self.controller.last_event.metadata['agent']['cameraHorizon']
        }
        
        print(f"                 Taking photos of {len(candidates)} candidates individually")
        successful_navigations = 0
        
        for i, obj in enumerate(candidates):
            print(f"[PHOTO CAPTURE] Processing candidate {i+1}/{len(candidates)}: {obj['objectType']}_{i+1}")
            
            # Navigate to this specific object
            nav_success = self.navigate_to_object_for_analysis(obj)

            if nav_success:
                successful_navigations += 1

                # Check if this is an openable container (like Cabinet, Drawer, etc.)
                should_open = (
                    obj.get("openable", False) and
                    not obj.get("isOpen", False) and
                    obj.get("receptacle", False)
                )

                container_was_opened = False
                if should_open:
                    try:
                        print(f"  -> Opening {obj['objectType']}_{i+1} for interior observation...")
                        open_event = self.action.action_mapping["open"](self.controller, obj['objectId'])
                        if open_event.metadata['lastActionSuccess']:
                            container_was_opened = True
                            self.update_event()
                            # Take a photo after opening
                            image_path_opened = self.save_frame({
                                "analyzing": f"{obj['objectType']}_{i+1}_opened",
                                "task": task_description,
                                "candidate": i+1,
                                "container_state": "opened",
                                "total_candidates": len(candidates)
                            })
                        else:
                            print(f"  -> Failed to open {obj['objectType']}_{i+1}: {open_event.metadata.get('errorMessage', 'Unknown error')}")
                    except Exception as e:
                        print(f"  -> Error opening container: {e}")

                # Take photo from this position (either with container open or as-is)
                image_path = self.save_frame({
                    "analyzing": f"{obj['objectType']}_{i+1}",
                    "task": task_description,
                    "candidate": i+1,
                    "container_opened": container_was_opened,
                    "total_candidates": len(candidates)
                })

                # Close the container if we opened it
                if container_was_opened:
                    try:
                        print(f"  -> Closing {obj['objectType']}_{i+1} after observation...")
                        close_event = self.action.action_mapping["close"](self.controller, obj['objectId'])
                        if not close_event.metadata['lastActionSuccess']:
                            print(f"  -> Warning: Failed to close {obj['objectType']}_{i+1}")
                        self.update_event()
                    except Exception as e:
                        print(f"  -> Error closing container: {e}")

                # No VLM analysis - just store photo info
                confidence = 50  # Neutral confidence for human selection
                if container_was_opened:
                    vlm_response = f"Photo captured with container opened for human selection: {obj['objectType']}_{i+1}"
                else:
                    vlm_response = f"Photo captured for human selection: {obj['objectType']}_{i+1}"
                analysis_quality = "photo_only"
                
            else:
                # Navigation failed - no photo
                print(f"  -> Navigation failed for {obj['objectType']}_{i+1}")
                confidence = 1  # Very low confidence for navigation failure
                vlm_response = f"Navigation to {obj['objectType']}_{i+1} failed. No photo available."
                analysis_quality = "navigation_failed"
                image_path = ""
            
            analyses.append({
                'object': obj,
                'index': i + 1,
                'confidence': confidence,
                'analysis': vlm_response,
                'analysis_quality': analysis_quality,
                'spatial_desc': self.generate_spatial_description(obj, i, candidates),
                'target_found': False,
                'image_path': image_path
            })
        
        # Return to original position
        print(f"[PHOTO CAPTURE] Returning to original position after capturing {len(candidates)} photos")
        self.return_to_position(original_position)
        
        print(f"[PHOTO CAPTURE] Photo capture complete: {successful_navigations}/{len(candidates)} successful navigations")
        
        # Return all analyses for human selection (don't sort by confidence)
        return analyses
    
    def get_object_by_choice(self, choice, analyses):
        """Get object by user choice number"""
        try:
            choice_num = int(choice) - 1
            if 0 <= choice_num < len(analyses):
                return analyses[choice_num]['object']
        except:
            pass
        return analyses[0]['object']
    
    def run_vlm_analysis_on_photos(self, photo_analyses, task_description):
        """Run VLM analysis on already captured photos"""
        print("[VLM FALLBACK] Analyzing captured photos with VLM...")
        vlm_analyses = []
        
        for analysis in photo_analyses:
            if analysis['image_path'] and analysis['analysis_quality'] != 'navigation_failed':
                obj = analysis['object']
                i = analysis['index'] - 1
                
                # Enhanced VLM prompt with GPT-4o context
                gpt4o_reasoning = getattr(self, 'current_gpt4o_reasoning', 'Agent decided to navigate to this object type.')
                
                prompt = f"""TASK: {task_description}
AGENT'S REASONING: {gpt4o_reasoning}

CURRENT ANALYSIS: {obj['objectType']} #{analysis['index']}
Position: x={obj['position']['x']:.1f}, z={obj['position']['z']:.1f}
Spatial Description: {analysis['spatial_desc']}

Analyze this {obj['objectType']} for the given task:
1. What objects/items are visible on or near this {obj['objectType']}?
2. Based on the task and agent's reasoning, how suitable is this {obj['objectType']}?
3. Provide a confidence score (0-100) for task completion relevance.

Response Format:
Visible Objects: [list what you see]
Task Suitability: [explain relevance to task]
Confidence: [0-100]"""
                
                try:
                    vlm_response = self.vlm_call_with_logging(analysis['image_path'], prompt)
                    confidence = self.extract_confidence_from_response(vlm_response)
                    analysis['confidence'] = confidence
                    analysis['analysis'] = vlm_response
                    analysis['analysis_quality'] = "vlm_fallback"
                except Exception as e:
                    print(f"  -> VLM fallback failed for {obj['objectType']}_{analysis['index']}: {e}")
                    analysis['confidence'] = 5  # Very low confidence
                    analysis['analysis_quality'] = "vlm_fallback_failed"
            
            vlm_analyses.append(analysis)
        
        return sorted(vlm_analyses, key=lambda x: x['confidence'], reverse=True)
    
    def select_best_candidate_from_vlm(self, analyses, itemtype):
        """Select best candidate from VLM analyses"""
        if not analyses:
            print(f"[ERROR] No analyses available for {itemtype}")
            return None

        # Get the highest confidence score
        max_confidence = analyses[0]['confidence']

        # Find all candidates with the highest confidence
        best_candidates = [a for a in analyses if a['confidence'] == max_confidence]

        # If multiple candidates have the same highest confidence, randomly select one
        if len(best_candidates) > 1:
            import random
            best_analysis = random.choice(best_candidates)
            print(f"[VLM SELECTED] Multiple candidates with same confidence ({max_confidence}%), randomly selected: {itemtype}_{best_analysis['index']}")
        else:
            best_analysis = best_candidates[0]
            print(f"[VLM SELECTED] Best candidate: {itemtype}_{best_analysis['index']} (confidence: {best_analysis['confidence']}%)")

        return best_analysis['object']
    
    def extract_target_detection(self, vlm_response):
        """Extract whether target object was found from VLM response"""
        response_lower = vlm_response.lower()
        positive_indicators = ['found', 'detected', 'visible', 'see', 'present', 'available']
        return any(indicator in response_lower for indicator in positive_indicators)
    
    def analyze_candidates_with_vlm(self, task_description, candidates):
        """Navigate to ALL candidates for individual analysis"""
        analyses = []
        
        # Store original position to return to
        original_position = {
            'position': self.controller.last_event.metadata['agent']['position'],
            'rotation': self.controller.last_event.metadata['agent']['rotation'],
            'horizon': self.controller.last_event.metadata['agent']['cameraHorizon']
        }
        
        print(f"[VLM ANALYSIS] Analyzing {len(candidates)} candidates individually")
        successful_navigations = 0
        
        for i, obj in enumerate(candidates):
            print(f"[VLM ANALYSIS] Processing candidate {i+1}/{len(candidates)}: {obj['objectType']}_{i+1}")
            
            # Navigate to this specific object
            nav_success = self.navigate_to_object_for_analysis(obj)

            if nav_success:
                successful_navigations += 1

                # Check if this is an openable container (like Cabinet, Drawer, etc.)
                should_open = (
                    obj.get("openable", False) and
                    not obj.get("isOpen", False) and
                    obj.get("receptacle", False)
                )

                container_was_opened = False
                if should_open:
                    try:
                        print(f"  -> Opening {obj['objectType']}_{i+1} for VLM analysis...")
                        open_event = self.action.action_mapping["open"](self.controller, obj['objectId'])
                        if open_event.metadata['lastActionSuccess']:
                            container_was_opened = True
                            self.update_event()
                        else:
                            print(f"  -> Failed to open {obj['objectType']}_{i+1}: {open_event.metadata.get('errorMessage', 'Unknown error')}")
                    except Exception as e:
                        print(f"  -> Error opening container: {e}")

                # Take photo from this position (either with container open or as-is)
                image_path = self.save_frame({
                    "analyzing": f"{obj['objectType']}_{i+1}",
                    "task": task_description,
                    "candidate": i+1,
                    "container_opened": container_was_opened,
                    "total_candidates": len(candidates)
                })

                # Enhanced VLM prompt with GPT-4o context
                gpt4o_reasoning = getattr(self, 'current_gpt4o_reasoning', 'Agent decided to navigate to this object type.')
                
                # Add container state to the prompt if applicable
                container_state = ""
                if container_was_opened:
                    container_state = f"\nContainer Status: This {obj['objectType']} has been OPENED for inspection."
                elif obj.get("openable", False):
                    container_state = f"\nContainer Status: This {obj['objectType']} is closed (could not be opened)."

                prompt = f"""TASK: {task_description}
AGENT'S REASONING: {gpt4o_reasoning}

CURRENT ANALYSIS: {obj['objectType']} #{i+1}
Position: x={obj['position']['x']:.1f}, z={obj['position']['z']:.1f}
Spatial Description: {self.generate_spatial_description(obj, i, candidates)}{container_state}

Analyze this {obj['objectType']} for the given task:
1. What objects/items are visible on or near this {obj['objectType']}?{' (including inside if opened)' if container_was_opened else ''}
2. Based on the task and agent's reasoning, how suitable is this {obj['objectType']}?
3. Provide a confidence score (0-100) for task completion relevance.

Response Format:
Visible Objects: [list what you see]
Task Suitability: [explain relevance to task]
Confidence: [0-100]"""
                
                # Handle VLM call with error recovery
                try:
                    vlm_response = self.vlm_call_with_logging(image_path, prompt)
                    confidence = self.extract_confidence_from_response(vlm_response)
                    analysis_quality = "individual_navigation"
                except Exception as e:
                    print(f"  -> VLM call failed for {obj['objectType']}_{i+1}: {e}")
                    print(f"  -> Continuing with next object...")
                    vlm_response = f"VLM analysis failed: {str(e)}. Object was successfully reached for navigation."
                    confidence = 5  # Very low confidence for VLM failure
                    analysis_quality = "navigation_success_vlm_failed"

                # Close the container if we opened it (after VLM analysis)
                if container_was_opened:
                    try:
                        print(f"  -> Closing {obj['objectType']}_{i+1} after VLM analysis...")
                        close_event = self.action.action_mapping["close"](self.controller, obj['objectId'])
                        if not close_event.metadata['lastActionSuccess']:
                            print(f"  -> Warning: Failed to close {obj['objectType']}_{i+1}")
                        self.update_event()
                    except Exception as e:
                        print(f"  -> Error closing container: {e}")
                
            else:
                # Navigation failed - use spatial reasoning
                print(f"  -> Navigation failed for {obj['objectType']}_{i+1}")
                confidence = 1  # Very low confidence for navigation failure
                vlm_response = f"Navigation to {obj['objectType']}_{i+1} failed. Using spatial estimation."
                analysis_quality = "navigation_failed"
                image_path = ""
            
            analyses.append({
                'object': obj,
                'index': i + 1,
                'confidence': confidence,
                'analysis': vlm_response,
                'analysis_quality': analysis_quality,
                'spatial_desc': self.generate_spatial_description(obj, i, candidates),
                'target_found': self.extract_target_detection(vlm_response) if nav_success else False,
                'image_path': image_path
            })
        
        # Return to original position
        print(f"[VLM ANALYSIS] Returning to original position after analyzing {len(candidates)} candidates")
        self.return_to_position(original_position)
        
        print(f"[VLM ANALYSIS] Analysis complete: {successful_navigations}/{len(candidates)} successful navigations")
        
        # Fallback check: if all navigations failed
        if successful_navigations == 0:
            print("[FALLBACK] All VLM navigations failed - will use xxx[0] fallback")
            return None  # Signal to use fallback
        
        return sorted(analyses, key=lambda x: x['confidence'], reverse=True)
    
    def get_object_visibility_details(self, obj):
        """Check if object is visible from current position"""
        try:
            current_objects = self.controller.last_event.metadata.get('objects', [])
            for scene_obj in current_objects:
                if scene_obj['objectId'] == obj['objectId']:
                    is_visible = scene_obj.get('visible', False)
                    distance = scene_obj.get('distance', float('inf'))
                    return {
                        'is_visible': is_visible,
                        'distance': distance,
                        'reason': 'clearly_visible' if is_visible else 'not_in_view'
                    }
            
            return {'is_visible': False, 'distance': float('inf'), 'reason': 'object_not_found'}
        except Exception as e:
            print(f"Error checking visibility for {obj['objectId']}: {e}")
            return {'is_visible': False, 'distance': float('inf'), 'reason': 'error_checking'}

    def vlm_call_with_object_focus(self, image_path, task_description, target_obj, visibility_info, candidate_index):
        """VLM analysis focusing on specific object"""
        obj_type = target_obj['objectType']
        obj_position = target_obj['position']
        
        focused_prompt = f"""TASK: {task_description}

FOCUS OBJECT: {obj_type} (Candidate {candidate_index})
- Position: x={obj_position['x']:.1f}, z={obj_position['z']:.1f}
- Distance: {visibility_info['distance']:.1f}m

Analyze this scene for the {obj_type} at the given position. How suitable is this {obj_type} for the task "{task_description}"?

Respond with:
Reasoning: [your analysis]
Confidence: [0-100]"""
        
        return self.vlm_call_with_logging(
            image_path, 
            focused_prompt,
            analysis_type=f"focused_{obj_type}_analysis"
        )

    def extract_confidence_from_response(self, vlm_response):
        """Extract confidence score from VLM response"""
        confidence = 40  # Default for visible objects
        if "Confidence:" in vlm_response:
            try:
                confidence_part = vlm_response.split("Confidence:")[-1].strip()
                confidence_str = confidence_part.split()[0].rstrip('%')
                confidence = int(confidence_str)
            except:
                pass
        return confidence
    
    def navigate_to_observe_candidate(self, obj):
        """Navigate to observe a candidate object"""
        try:
            # Calculate observation position
            target_position, target_rotation = self.compute_position_8(obj, [])
            
            # Teleport to position
            event = self.action.action_mapping["teleport"](
                self.controller, 
                position=target_position, 
                rotation=target_rotation
            )
            self.update_event()
            
            return event.metadata['lastActionSuccess']
        except Exception as e:
            print(f"Error navigating to candidate: {e}")
            return False
    
    def generate_disambiguation_message(self, task, candidates, analyses):
        """Generate clear disambiguation message with recommendation"""
        obj_type = candidates[0]['objectType']
        best = analyses[0]
        
        message = f"""Task: {task}

There're {len(candidates)} {obj_type} objects in the room:

"""
        
        for analysis in analyses:
            message += f"Option {analysis['index']}: {obj_type}_{analysis['index']}\n"
            message += f"  Location: {analysis['spatial_desc']}\n"
            message += f"  Analysis: {analysis['analysis'].split('Objects:')[1].split('Credit card visible:')[0].strip() if 'Objects:' in analysis['analysis'] else 'kitchen items'}\n"
            message += f"  Confidence: {analysis['confidence']}%\n\n"
        
        message += f"""[VLM] I recommend Option {best['index']} ({obj_type}_{best['index']}) - it has the highest confidence ({best['confidence']}%).

Please choose:"""
        
        for analysis in analyses:
            message += f"\n  Type '{analysis['index']}' for {obj_type}_{analysis['index']}"
            
        message += f"\n  Type 'auto' to use my recommendation"
        message += f"\n\n  You have {self.human_selection_timeout} seconds to respond."
        
        return message
    
    def parse_user_response(self, response, candidates, analyses):
        """Parse user response with fallback to recommendation"""
        # Convert response to string if it's an integer (from Web Dashboard)
        if isinstance(response, int):
            response = str(response)
        
        if not response or str(response).lower().strip() in ['auto', 'recommend', 'best', '']:
            print(f"Using recommended option: {analyses[0]['object']['objectType']}_{analyses[0]['index']}")
            return analyses[0]['object']
        
        response_clean = str(response).lower().strip()
        
        # Check for number responses (1, 2, 3, etc.)
        try:
            choice_num = int(response_clean)
            if 1 <= choice_num <= len(candidates):
                selected = candidates[choice_num - 1]
                print(f"You selected option {choice_num}: {selected['objectType']}_{choice_num}")
                return selected
        except ValueError:
            pass
        
        # If cannot parse, use recommendation
        print(f"Could not understand '{response}'. Using recommended option.")
        return analyses[0]['object']
    
    def request_user_disambiguation(self, itemtype, candidates, task_description):
        """Main disambiguation flow"""
        # print(f"\nStarting disambiguation for {len(candidates)} {itemtype} objects...")
        
        # Analyze all candidates with VLM
        analyses = self.analyze_candidates_with_vlm(task_description, candidates)
        
        # Check if confidence gap is large enough to auto-select
        if len(analyses) > 1 and analyses[0]['confidence'] - analyses[1]['confidence'] > self.confidence_gap_threshold:
            print(f"High confidence gap ({analyses[0]['confidence']}% vs {analyses[1]['confidence']}%), auto-selecting {itemtype}_{analyses[0]['index']}")
            return analyses[0]['object']
        
        # Generate and display disambiguation request
        message = self.generate_disambiguation_message(task_description, candidates, analyses)
        
        # Get user response (with timeout)
        response = self.get_user_input_with_image_support(analyses)
        
        # Parse response and return selection
        selected = self.parse_user_response(response, candidates, analyses)
        return selected
    
    def set_task_description(self, description):
        """Set current task description for context"""
        self.current_task_description = description

    # ============== Multi-view Observation for Large Objects ==============
    
    def needs_multi_view_observation(self, item):
        try:
            volume = self.eventobject.get_item_volume(self.controller.last_event, item['name'])
            surface_area = self.eventobject.get_item_surface_area(self.controller.last_event, item['name'])
            
            # Threshold judgment
            needs_multi_view = volume > 1.0 or surface_area > 2.0
            
            if needs_multi_view:
                print(f"Large object detected: {item['objectType']}")
                print(f"   Volume: {volume:.3f}m³, Surface Area: {surface_area:.3f}m²")
                print(f"   → Multi-view observation needed")
            
            return needs_multi_view
        except Exception as e:
            print(f"[ERROR] Error checking object size: {e}")
            return False
    
    def get_verified_observation_positions(self, item):
        try:
            current_pos = self.controller.last_event.metadata['agent']['position']
            current_rot = self.controller.last_event.metadata['agent']['rotation']
            
            original_pos = {
                'x': current_pos['x'],
                'y': current_pos['y'], 
                'z': current_pos['z']
            }
            original_rot = {
                'x': current_rot['x'],
                'y': current_rot['y'],
                'z': current_rot['z']
            }
            
            # print(f"Current agent position as original view: ({original_pos['x']:.2f}, {original_pos['z']:.2f})")
            
            event = self.controller.step(dict(
                action='GetInteractablePoses', 
                objectId=item['objectId']
            ))
            
            if not event.metadata['lastActionSuccess']:
                print(f"[ERROR] Failed to get interactable positions for {item['objectType']}")
                return [(original_pos, original_rot)]
            
            all_positions = event.metadata['actionReturn']
            item_pos = item['position']
            
            # Dynamically filter supplementary observation positions
            all_distances = []
            positions_with_distance = []
            
            for pos in all_positions:
                distance_to_item = ((pos['x'] - item_pos['x'])**2 + (pos['z'] - item_pos['z'])**2)**0.5
                all_distances.append(distance_to_item)
                positions_with_distance.append((pos, distance_to_item))
            
            import numpy as np
            all_distances = np.array(all_distances)
            distance_20th = np.percentile(all_distances, 20)  # 20th percentile
            distance_50th = np.percentile(all_distances, 50)  # 50th percentile
            
            # print(f"[DEBUG]   Distance distribution: 20th={distance_20th:.2f}m, 50th={distance_50th:.2f}m")
            # print(f"[DEBUG]   Selecting positions in {distance_20th:.2f}m - {distance_50th:.2f}m range")
            
            suitable_positions = []
            min_distance_from_original = max(0.5, distance_20th * 0.5)  # Minimum distance from original position
            
            for pos, distance_to_item in positions_with_distance:
                # Check if within ideal distance range
                if not (distance_20th <= distance_to_item <= distance_50th):
                    continue
                    
                # Check distance from original position (avoid duplicate viewpoints)
                distance_to_original = ((pos['x'] - original_pos['x'])**2 + (pos['z'] - original_pos['z'])**2)**0.5
                if distance_to_original < min_distance_from_original:
                    continue
                
                suitable_positions.append(pos)
            
            print(f"[DEBUG]  Found {len(all_positions)} total positions, {len(suitable_positions)} suitable for supplementary views")
            
            # Select 2 positions with best angle distribution from supplementary positions
            if len(suitable_positions) >= 2:
                supplementary_positions = self.select_best_distributed_positions(
                    suitable_positions, item_pos, max_positions=2, exclude_angle=self.calculate_angle(original_pos, item_pos)
                )
            elif len(suitable_positions) == 1:
                # Only one supplementary position
                supplementary_pos = suitable_positions[0]
                supplementary_rot = self.calculate_look_at_rotation(supplementary_pos, item_pos)
                supplementary_positions = [(supplementary_pos, supplementary_rot)]
            else:
                # No suitable supplementary positions
                supplementary_positions = []
            
            # 5. Combine final results: original position + supplementary positions
            final_positions = [(original_pos, original_rot)]  # First is always original interaction position
            final_positions.extend(supplementary_positions)
            
            print(f"   [DEBUG] Final multi-view strategy:")
            print(f"     • View 1: Original interaction position (interaction optimized)")
            for i, (pos, rot) in enumerate(supplementary_positions, 2):
                angle = self.calculate_angle(pos, item_pos)
                distance = ((pos['x'] - item_pos['x'])**2 + (pos['z'] - item_pos['z'])**2)**0.5
                print(f"     • View {i}: Supplementary position at {angle:.1f}°, distance {distance:.2f}m")
            
            return final_positions
            
        except Exception as e:
            print(f"[ERROR] Error getting observation positions: {e}")
            # Return at least original position when error occurs
            try:
                original_pos, original_rot = self.compute_position_8(item, [])
                if original_pos:
                    return [(original_pos, original_rot)]
            except:
                pass
            return []
    
    def calculate_angle(self, observer_pos, target_pos):
        import math
        angle = math.atan2(
            observer_pos['z'] - target_pos['z'], 
            observer_pos['x'] - target_pos['x']
        )
        return (math.degrees(angle) + 360) % 360
    
    def select_best_distributed_positions(self, suitable_positions, item_center, max_positions=3, exclude_angle=None):
        import math
        
        if len(suitable_positions) <= max_positions:
            # If not many candidate positions, use all of them
            return [(pos, self.calculate_look_at_rotation(pos, item_center)) 
                    for pos in suitable_positions]
        
        # 1. Calculate angle of each position relative to object center
        position_angles = []
        for pos in suitable_positions:
            angle = self.calculate_angle(pos, item_center)
            
            distance = math.sqrt(
                (pos['x'] - item_center['x'])**2 + 
                (pos['z'] - item_center['z'])**2
            )
            
            position_angles.append({
                'position': pos,
                'angle': angle,
                'distance': distance
            })
        
        # 2. Use improved greedy algorithm (considering exclusion angles)
        selected = self.greedy_angle_selection_with_exclusion(position_angles, max_positions, exclude_angle)
        
        # 3. Calculate rotation angle for each selected position
        result = []
        for pos_info in selected:
            pos = pos_info['position']
            rotation = self.calculate_look_at_rotation(pos, item_center)
            result.append((pos, rotation))
            
            print(f"   Selected view: pos=({pos['x']:.2f}, {pos['z']:.2f}), "
                  f"angle={pos_info['angle']:.1f}°, distance={pos_info['distance']:.2f}m")
        
        return result
    
    def greedy_angle_selection_with_exclusion(self, position_angles, max_positions, exclude_angle=None):
        """
        Improved greedy algorithm: consider optimal distribution after excluding specific angles
        exclude_angle: Already occupied angle (original interaction position)
        """
        if len(position_angles) <= max_positions:
            return position_angles
        
        if exclude_angle is None:
            # No exclusion angle, use original algorithm
            return self.greedy_angle_selection(position_angles, max_positions)
        
        # Calculate ideal supplementary angles
        if max_positions == 2:
            # Need to select 2 supplementary positions, ideally forming 120° intervals
            ideal_angles = [(exclude_angle + 120) % 360, (exclude_angle + 240) % 360]
        elif max_positions == 1:
            # Only need 1 supplementary position, choose 180° opposite
            ideal_angles = [(exclude_angle + 180) % 360]
        else:
            # General case: uniform distribution
            angle_step = 360 / (max_positions + 1)  # +1 because need to consider exclusion angle
            ideal_angles = []
            for i in range(1, max_positions + 1):
                ideal_angles.append((exclude_angle + i * angle_step) % 360)
        
        print(f"   Excluding original angle {exclude_angle:.1f}°, targeting supplementary angles: {[f'{a:.1f}°' for a in ideal_angles]}")
        
        # Find closest actual position for each ideal angle
        selected = []
        used_positions = set()
        
        for ideal_angle in ideal_angles:
            best_match = None
            min_angle_diff = float('inf')
            
            for i, pos_info in enumerate(position_angles):
                if i in used_positions:
                    continue
                
                # Calculate angle difference (considering 360° cycle)
                angle_diff = min(
                    abs(pos_info['angle'] - ideal_angle),
                    abs(pos_info['angle'] - ideal_angle + 360),
                    abs(pos_info['angle'] - ideal_angle - 360)
                )
                
                if angle_diff < min_angle_diff:
                    min_angle_diff = angle_diff
                    best_match = i
            
            if best_match is not None:
                selected.append(position_angles[best_match])
                used_positions.add(best_match)
        
        return selected
    
    def greedy_angle_selection(self, position_angles, max_positions):
        if len(position_angles) <= max_positions:
            return position_angles
        
        # Uniformly divide 360° and select closest positions
        target_angles = [i * 360 / max_positions for i in range(max_positions)]
        
        # Find closest actual position for each target angle
        selected = []
        used_positions = set()
        
        for target_angle in target_angles:
            best_match = None
            min_angle_diff = float('inf')
            
            for i, pos_info in enumerate(position_angles):
                if i in used_positions:
                    continue
                    
                # Calculate angle difference (considering 360° cycle)
                angle_diff = min(
                    abs(pos_info['angle'] - target_angle),
                    abs(pos_info['angle'] - target_angle + 360),
                    abs(pos_info['angle'] - target_angle - 360)
                )
                
                if angle_diff < min_angle_diff:
                    min_angle_diff = angle_diff
                    best_match = i
            
            if best_match is not None:
                selected.append(position_angles[best_match])
                used_positions.add(best_match)
        
        return selected
    
    def calculate_look_at_rotation(self, observer_pos, target_pos):
        """
        Calculate rotation angle from observation position looking towards target position
        """
        import math
        
        dx = target_pos['x'] - observer_pos['x']
        dz = target_pos['z'] - observer_pos['z']
        
        # Calculate facing angle
        angle_rad = math.atan2(dx, dz)
        angle_deg = math.degrees(angle_rad)
        
        # Convert to AI2Thor rotation format
        rotation = {
            'x': 0,
            'y': (angle_deg + 360) % 360,
            'z': 0
        }
        
        return rotation

    def create_multi_view_composite(self, image_paths, itemtype):
        """Create concatenated multi-view image with labeled views"""
        try:
            import cv2
            import numpy as np

            if len(image_paths) < 2:
                return image_paths[0] if image_paths else None

            # Load images
            loaded_images = []
            labels = []

            for i, img_path in enumerate(image_paths):
                img = cv2.imread(img_path)
                if img is not None:
                    # Generate view labels
                    if i == 0:
                        label = "Interaction View"
                    else:
                        label = f"Supplementary View {i}"

                    # Add text label to image
                    img_with_label = img.copy()
                    cv2.rectangle(img_with_label, (10, 10), (350, 50), (255, 255, 255), -1)  # White background
                    cv2.rectangle(img_with_label, (10, 10), (350, 50), (0, 0, 0), 2)  # Black border
                    cv2.putText(img_with_label, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                    loaded_images.append(img_with_label)
                    labels.append(label)

            if len(loaded_images) < 2:
                return image_paths[0] if image_paths else None

            # Concatenate images horizontally
            composite = np.concatenate(loaded_images, axis=1)

            # Save composite image
            composite_path = image_paths[0].replace(f'_{itemtype}_1_interaction_', f'_{itemtype}_composite_')
            cv2.imwrite(composite_path, composite)

            print(f"   Created composite with {len(loaded_images)} views: {labels}")
            return composite_path

        except Exception as e:
            print(f"[WARNING] Failed to create composite image: {e}")
            return image_paths[0] if image_paths else None

    def navigate_complete_view(self, itemtype):
        # print(f"[DEBUG] Starting multi-view observation for {itemtype}")
        
        # Get target object (using task 1 enhanced logic)
        if itemtype in self.objecttype2object:
            objects = self.objecttype2object[itemtype]
            
            if len(objects) > 1 and self.enable_dialogue_system:
                task_description = getattr(self, 'current_task_description', f"observe {itemtype}")
                item = self.request_user_disambiguation_improved(itemtype, objects)
            else:
                item = objects[0]
        else:
            print(f"[Warning] No objects of type {itemtype} found")
            return None, None, None
        
        # Determine if multi-angle observation is needed
        if self.needs_multi_view_observation(item):
            print(f"[DEBUG] Initiating multi-view observation...")
            
            # Get optimized observation positions
            positions = self.get_verified_observation_positions(item)
            if not positions:
                print(f"[Warning] No suitable observation positions, falling back to standard navigation")
                return self.navigate(itemtype)
            
            # Observe each position sequentially (improved strategy)
            collected_images = []
            successful_views = 0

            for i, (pos, rot) in enumerate(positions):
                view_type = "Interaction" if i == 0 else f"Observation"
                print(f"\nNavigating to view {i+1}/{len(positions)} ({view_type})...")
                
                try:
                    # Navigate to observation position (inherit original vision optimization)
                    event = self.action.action_mapping["teleport"](
                        self.controller,
                        position=pos,
                        rotation=rot,
                        horizon=60
                    )
                    
                    if event.metadata['lastActionSuccess']:
                        # print(f"   [DEBUG] Navigation successful!")
                        
                        # Apply original navigation's vision optimization
                        self.adjust_height(item)
                        self.adjust_view(item)
                        
                        # Save observation image, distinguish viewpoint type
                        view_type_label = "interaction" if i == 0 else "supplementary"
                        image_fp = self.save_frame({
                            "step_count": str(self.step_count),
                            "action": "multi_view_observation",
                            "item": item["objectType"],
                            "view": i + 1,
                            "view_type": view_type_label,
                            "total_views": len(positions)
                        })
                        
                        print(f"   Image saved: {image_fp}")
                        collected_images.append(image_fp)  # Collect all images for concatenation
                        successful_views += 1
                        
                    else:
                        print(f"   [Warning] Navigation failed")
                        
                except Exception as e:
                    print(f"   [Warning] Error during navigation: {e}")
            
            print(f"\nMulti-view observation complete:")
            print(f"   • Total positions attempted: {len(positions)}")
            print(f"   • Successful observations: {successful_views}")
            print(f"   • Success rate: {successful_views/len(positions)*100:.1f}%")
            print(f"   • Strategy: Original interaction + {len(positions)-1} supplementary views")

            # Create concatenated multi-view image for comprehensive analysis
            if len(collected_images) > 1:
                composite_image = self.create_multi_view_composite(collected_images, itemtype)
                print(f"   Multi-view composite created: {composite_image}")
                return composite_image, True, True
            elif len(collected_images) == 1:
                return collected_images[0], True, True
            else:
                print(f"[ERROR] All multi-view attempts failed, falling back to standard navigation")
                return self.navigate(itemtype)
        
        else:
            print(f"Small object detected, using standard single-view navigation")
            return self.navigate(itemtype)
    
    def enable_enhanced_navigation(self, enable_indexing=True, enable_dialogue=True, enable_multi_view=True,
                                 disambiguation_mode="human_first_vlm_fallback"):
        """
        Enable enhanced navigation functionality
        Args:
            disambiguation_mode: One of:
                - "human_first_vlm_fallback": Human first, VLM analysis as fallback
                - "vlm_first_human_choice": VLM analysis first, human choice with confidence scores
                - "human_only_random_fallback": Human only, random selection as fallback
        """
        self.enable_object_indexing = enable_indexing
        self.enable_dialogue_system = enable_dialogue
        self.enable_multi_view = enable_multi_view
        self.disambiguation_mode = disambiguation_mode

        print(f"Enhanced Navigation Configuration:")
        print(f"  • Object Indexing: {'[ENABLED]' if enable_indexing else '[DISABLED]'}")
        print(f"  • Dialogue System: {'[ENABLED]' if enable_dialogue else '[DISABLED]'}")
        print(f"  • Multi-view Observation: {'[ENABLED]' if enable_multi_view else '[DISABLED]'}")
        print(f"  • Disambiguation Mode: {disambiguation_mode}")
        
        # Reinitialize object index
        if enable_indexing:
            self.init_object_indexing()

    def create_vlm_dialogue_visualization(self, obj_type, index, input_prompt, input_image_path, vlm_response, confidence, analysis_info=None):
        """
        Create a comprehensive visualization of VLM dialogue for debugging and review
        Creates a composite image showing: input image + prompt text + VLM response
        """
        import cv2
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        import textwrap
        
        try:
            input_img = cv2.imread(input_image_path)
            if input_img is None:
                print(f"Warning: Could not load image {input_image_path}")
                return None
                
            input_img_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            h_img, w_img = input_img_rgb.shape[:2]
            
            # Create text sections
            sections = {
                'title': f"VLM DIALOGUE RECORD - {obj_type}_{index}",
                'input_prompt': input_prompt,
                'vlm_response': vlm_response,
                'confidence': f"Extracted Confidence: {confidence}%",
                'metadata': f"Timestamp: {self.get_timestamp()} | Object: {obj_type}_{index} | Task: {getattr(self, 'current_task', 'N/A')}"
            }
            
            # Calculate text area dimensions
            text_width = max(w_img, 800)  # At least 800px wide for text
            char_width, char_height = 8, 16
            
            # Calculate heights for each section
            title_lines = 2
            prompt_lines = max(8, len(textwrap.wrap(sections['input_prompt'], width=text_width//char_width)))
            response_lines = max(6, len(textwrap.wrap(sections['vlm_response'], width=text_width//char_width)))
            confidence_lines = 2
            metadata_lines = 2
            
            total_text_height = (title_lines + prompt_lines + response_lines + confidence_lines + metadata_lines + 8) * char_height
            
            # Create composite image
            total_width = max(w_img, text_width)
            total_height = h_img + total_text_height + 20  # 20px padding
            
            composite = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255  # White background
            
            # Place input image (resize if too large)
            if w_img > total_width:
                # Resize image to fit width
                scale = total_width / w_img
                new_w, new_h = int(w_img * scale), int(h_img * scale)
                input_img_rgb = cv2.resize(input_img_rgb, (new_w, new_h))
                h_img, w_img = new_h, new_w
            
            # Center image horizontally
            x_offset = (total_width - w_img) // 2
            composite[10:10+h_img, x_offset:x_offset+w_img] = input_img_rgb
            
            # Convert to PIL for text rendering
            pil_image = Image.fromarray(composite)
            draw = ImageDraw.Draw(pil_image)
            
            # Try to use a monospace font, fallback to default
            try:
                font_large = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf", 16)
                font_normal = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf", 12)
                font_small = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf", 10)
            except:
                font_large = ImageFont.load_default()
                font_normal = ImageFont.load_default()
                font_small = ImageFont.load_default()
            
            # Draw text sections
            y_start = h_img + 30
            current_y = y_start
            
            # Title
            draw.text((10, current_y), sections['title'], fill=(0, 0, 0), font=font_large)
            current_y += 40
            
            # Draw separator
            draw.line((10, current_y, total_width-10, current_y), fill=(200, 200, 200), width=2)
            current_y += 15
            
            # Input Prompt
            draw.text((10, current_y), "INPUT PROMPT:", fill=(0, 100, 0), font=font_normal)
            current_y += 20
            
            wrapped_prompt = textwrap.wrap(sections['input_prompt'], width=90)
            for line in wrapped_prompt:
                draw.text((15, current_y), line, fill=(0, 0, 0), font=font_small)
                current_y += 14
            
            current_y += 10
            
            # VLM Response
            draw.text((10, current_y), "VLM RESPONSE:", fill=(0, 0, 100), font=font_normal)
            current_y += 20
            
            wrapped_response = textwrap.wrap(sections['vlm_response'], width=90)
            for line in wrapped_response:
                draw.text((15, current_y), line, fill=(0, 0, 0), font=font_small)
                current_y += 14
            
            current_y += 10
            
            # Confidence
            draw.text((10, current_y), sections['confidence'], fill=(100, 0, 0), font=font_normal)
            current_y += 20
            
            # Metadata
            draw.text((10, current_y), sections['metadata'], fill=(100, 100, 100), font=font_small)
            
            # Convert back to OpenCV format and save
            final_image = np.array(pil_image)
            final_image_bgr = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
            
            # Save visualization
            viz_path = f"{self.result_dir}/vlm_dialogue_viz_{obj_type}_{index}_{self.step_count}.png"
            cv2.imwrite(viz_path, final_image_bgr)
            
            print(f"VLM dialogue visualization saved: {viz_path}")
            return viz_path
            
        except Exception as e:
            print(f"Error creating VLM visualization: {e}")
            return None
    
    def get_timestamp(self):
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def set_task_context(self, task_description, subtasks=None, task_type=None):
        self.current_task = task_description
        self.current_subtasks = subtasks if subtasks else []
        # Extract target object types from original task data (related_objects)
        target_objects = []
        if hasattr(self, 'related_objects') and self.related_objects:
            for obj_id in self.related_objects:
                obj_type = obj_id.split("|")[0] if "|" in obj_id else obj_id
                target_objects.append(obj_type)

        self.task_context = {
            'task': task_description,
            'subtasks': subtasks,
            'task_type': task_type,  # e.g., 'pick_and_place', 'search', 'navigate'
            'target_objects': list(set(target_objects))  # Remove duplicates
        }
        print(f"Task context set:")
        print(f"   Main task: {task_description}")
        if subtasks:
            print(f"   Subtasks: {', '.join(subtasks)}")
    
    def generate_vlm_prompt_improved(self, obj_type, candidate_index):
        """Improved VLM prompt generation - uses original task data instead of hardcoded keywords"""

        main_task = getattr(self, 'current_task', 'Navigation task')

        # Extract target info from original task data (related_objects contains what we're looking for)
        target_info = ""
        if hasattr(self, 'related_objects') and self.related_objects:
            # Parse object IDs to get object types
            target_types = []
            for obj_id in self.related_objects:
                obj_type_from_id = obj_id.split("|")[0] if "|" in obj_id else obj_id
                target_types.append(obj_type_from_id)
            target_info = f"Looking for: {', '.join(set(target_types))}"
        else:
            target_info = "Looking for task-related items"

        prompt = f"""You are a household agent carrying out specific tasks. Your task: "{main_task}"
{target_info}

You need to choose between multiple {obj_type} objects. Currently you are looking at {obj_type}_{candidate_index}.

Question: How likely is it that this {obj_type} contains or is related to what you're looking for?

You may consider:
- What you can see in/on/around this {obj_type}
- Whether this type of {obj_type} typically stores the target item
- Any visual clues that suggest the target might be here

Give a confidence score (0-100) and explain your reasoning.

Format:
Reasoning: [why this location is promising or not for finding the target]
Confidence: [0-100]"""

        return prompt
    
    def generate_vlm_prompt(self, obj_type, candidate_index):
        """Dynamic VLM prompt generation based on task context - DEPRECATED, use improved version"""
        # Keep old version for compatibility but mark as deprecated
        return self.generate_vlm_prompt_improved(obj_type, candidate_index)
    

    
    
    
    
    def create_candidate_comparison_image(self, analyses):
        """Create side-by-side comparison of all candidate objects"""
        import cv2
        import numpy as np
        
        if not analyses:
            return None
        
        images = []
        max_height = 0
        
        for analysis in analyses:
            try:
                # Load image with safety checks
                img_path = analysis.get('image_path', '')
                if not img_path:
                    print(f"Warning: Missing image_path for analysis {analysis.get('index', 'unknown')}")
                    continue
                if not os.path.exists(img_path):
                    print(f"Warning: Image file does not exist: {img_path}")
                    continue
                    
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Could not load image from {img_path}")
                    continue
                
                height, width = img.shape[:2]
                
                # Create title bar
                title_bar_height = 60
                title_bar = np.zeros((title_bar_height, width, 3), dtype=np.uint8)
                title_bar[:] = (50, 50, 50)  # Dark gray background
                
                # Add title text
                title = f"Option {analysis['index']} - Conf: {analysis['confidence']}%"
                if analysis.get('target_found', False):
                    title += " [TARGET FOUND]"
                
                cv2.putText(
                    title_bar,
                    title,
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
                
                # Add spatial description
                spatial_desc = analysis.get('spatial_desc', '')
                if spatial_desc:
                    cv2.putText(
                        title_bar,
                        spatial_desc[:40],  # Truncate if too long
                        (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (200, 200, 200),
                        1
                    )
                
                # Combine title bar and image
                combined = np.vstack([title_bar, img])
                images.append(combined)
                max_height = max(max_height, combined.shape[0])
                
            except Exception as e:
                print(f"Error processing image for analysis {analysis['index']}: {e}")
        
        if not images:
            return None
        
        # Normalize heights
        for i in range(len(images)):
            h, w = images[i].shape[:2]
            if h < max_height:
                padding = np.zeros((max_height - h, w, 3), dtype=np.uint8)
                images[i] = np.vstack([images[i], padding])
        
        # Horizontal concatenation
        comparison = np.hstack(images)
        
        # Save comparison image
        comparison_path = f"{self.result_dir}/candidate_comparison_{self.step_count}.png"
        cv2.imwrite(comparison_path, comparison)
        
        print(f"Candidate comparison saved: {comparison_path}")
        return comparison_path
    
    def extract_target_detection(self, vlm_response):
        """Extract whether target objects were found from VLM response"""
        response_lower = vlm_response.lower()
        
        # Check for old format first
        if 'target objects found: yes' in response_lower:
            return True
        
        # For new format, check confidence level
        # High confidence (>70) suggests task relevance
        confidence = 25
        if "confidence:" in response_lower:
            try:
                confidence_line = response_lower.split("confidence:")[-1].strip()
                confidence_str = confidence_line.split()[0].rstrip('%')
                confidence = int(confidence_str)
            except:
                pass
        elif "confidence score:" in response_lower:  # Fallback
            try:
                confidence_line = response_lower.split("confidence score:")[-1].strip()
                confidence_str = confidence_line.split()[0].rstrip('%')
                confidence = int(confidence_str)
            except:
                pass
        
        # Consider high confidence as target found
        return confidence > 70
    
    def extract_visible_objects(self, vlm_response):
        """Extract reasoning from VLM response"""
        try:
            # Try new format first
            if 'Reasoning:' in vlm_response:
                reasoning_line = vlm_response.split('Reasoning:')[1].split('Confidence:')[0]
                return reasoning_line.strip()
            # Fallback to old format
            elif 'Visible objects:' in vlm_response:
                visible_line = vlm_response.split('Visible objects:')[1].split('\n')[0]
                return visible_line.strip()
        except:
            pass
        return "analysis pending"
    
    def analyze_candidates_with_vlm_improved(self, candidates):
        """Enhanced VLM analysis with dynamic prompts and visualization"""
        analyses = []
        
        # First generate images for all candidates (simplified - no bounding boxes)
        candidate_images = []
        for i, obj in enumerate(candidates):
            obj_type = obj['objectType']
            print(f"Capturing image for {obj_type}_{i+1}")

            # Navigate to observe this candidate
            success = self.navigate_to_observe_candidate(obj)
            if success:
                # Save simple image without bounding box
                image_path = self.save_frame({
                    "step_count": str(self.step_count),
                    "candidate": f"{obj_type}_{i+1}",
                    "action": "candidate_capture"
                })

                candidate_images.append({
                    'index': i+1,
                    'object': obj,
                    'image_path': image_path,
                    'has_box': False,  # No bounding box
                    'bbox': None
                })
                print(f"   Image saved: {image_path}")
            else:
                print(f"   [Warning] Failed to navigate to {obj_type}_{i+1}")

        print(f"Generated {len(candidate_images)} candidate images")
        
        for i, img_info in enumerate(candidate_images):
            obj = img_info['object']
            obj_type = obj['objectType']
            
            print(f"Analyzing {obj_type}_{i+1} with task context...")
            
            # Generate dynamic prompt based on current task
            prompt = self.generate_vlm_prompt(obj_type, i+1)
            
            # Call VLM with the generated prompt using new architecture
            vlm_response = self.vlm_call_with_logging(
                img_info['image_path'], 
                prompt, 
                analysis_type=f"enhanced_analysis_{obj_type}_{i+1}"
            )
            
            # Extract confidence from new simplified format
            confidence = 25  # Default
            if "Confidence:" in vlm_response:
                try:
                    confidence_line = vlm_response.split("Confidence:")[-1].strip()
                    confidence_str = confidence_line.split()[0].rstrip('%')
                    confidence = int(confidence_str)
                except:
                    pass
            elif "Confidence score:" in vlm_response:  # Fallback for old format
                try:
                    confidence_line = vlm_response.split("Confidence score:")[-1].strip()
                    confidence_str = confidence_line.split()[0].rstrip('%')
                    confidence = int(confidence_str)
                except:
                    pass
            
            target_found = self.extract_target_detection(vlm_response)
            
            # Create VLM dialogue visualization for debugging
            viz_path = self.create_vlm_dialogue_visualization(
                obj_type, i+1, prompt, img_info['image_path'], 
                vlm_response, confidence
            )
            
            analyses.append({
                'object': obj,
                'index': i+1,
                'analysis': vlm_response,
                'confidence': confidence,
                'target_found': target_found,
                'image_path': img_info['image_path'],
                'vlm_viz_path': viz_path,
                'has_box': img_info['has_box'],
                'spatial_desc': self.generate_spatial_description(obj, i, candidates)
            })
            
            print(f"  Confidence: {confidence}%")
            print(f"  Target found: {'YES' if target_found else 'NO'}")
            if viz_path:
                print(f"  VLM dialogue record: {viz_path}")
        
        return sorted(analyses, key=lambda x: x['confidence'], reverse=True)
    
    def generate_visual_disambiguation_message(self, candidates, analyses):
        """Generate enhanced disambiguation message with image references"""
        obj_type = candidates[0]['objectType']
        best = analyses[0]
        
        # Create comparison image
        comparison_path = self.create_candidate_comparison_image(analyses)
        
        message = f"""
{'='*80}
MULTI-OBJECT DISAMBIGUATION - ENHANCED INTERFACE
{'='*80}

Current Task: {getattr(self, 'current_task', 'Navigation task')}
Target Objects: {', '.join(getattr(self, 'task_context', {}).get('target_objects', ['task items']))}
Current Step: Navigate to {obj_type}

I found {len(candidates)} {obj_type} objects in the room.

{'='*80}
DETAILED ANALYSIS
{'='*80}"""
        
        for analysis in analyses:
            target_status = "[TARGET DETECTED]" if analysis.get('target_found', False) else "[No target found]"
            
            message += f"""

Option {analysis['index']}: {obj_type}_{analysis['index']}
  Location: {analysis['spatial_desc']}
  Reasoning: {self.extract_visible_objects(analysis['analysis'])}
  Status: {target_status}
  VLM Confidence: {analysis['confidence']}%
  Image: {analysis['image_path']}
  VLM Record: {analysis.get('vlm_viz_path', 'N/A')}
{'─'*60}"""
        
        # Recommendation logic
        target_candidates = [a for a in analyses if a.get('target_found', False)]
        if target_candidates:
            recommended = target_candidates[0]
            reason = f"Target object detected with {recommended['confidence']}% confidence"
        else:
            recommended = best
            reason = f"Highest task relevance confidence ({recommended['confidence']}%)"
        
        message += f"""

{'='*33}RECOMMENDATION{'='*33}
I recommend Option {recommended['index']} ({obj_type}_{recommended['index']})
Reason: {reason}

YOUR CHOICE OPTIONS
{'='*80}
Please enter:
  • Number (1-{len(candidates)}) to select specific {obj_type}
  • 'auto' to use my intelligent recommendation  
  • 'show' to display the comparison image again
  • 'vlm X' to review VLM analysis for option X

You have {self.human_selection_timeout} seconds to respond.
{'='*80}
"""
        
        return message, comparison_path
    
    def create_web_selection_interface(self, analyses, comparison_path):
        """Create web interface for visual candidate selection"""
        try:
            import threading
            import webbrowser
            from http.server import HTTPServer, SimpleHTTPRequestHandler
            import os
            import json
            import urllib.parse
            import time
            
            class CandidateHTTPHandler(SimpleHTTPRequestHandler):
                def __init__(self, *args, analyses=None, comparison_path=None, result_container=None, current_task=None, **kwargs):
                    self.analyses = analyses
                    self.comparison_path = comparison_path  
                    self.result_container = result_container
                    self.current_task = current_task or "Navigation task"
                    super().__init__(*args, **kwargs)
                
                def do_GET(self):
                    if self.path == '/':
                        self.serve_selection_page()
                    elif self.path.startswith('/select/'):
                        choice = int(self.path.split('/')[-1])
                        self.result_container['selection'] = choice
                        self.serve_result_page(choice)
                    elif self.path.startswith('/image/'):
                        # Serve images
                        image_path = urllib.parse.unquote(self.path[7:])  # Remove '/image/'
                        self.serve_image(image_path)
                    else:
                        self.send_error(404)
                
                def serve_selection_page(self):
                    html = self.generate_selection_html()
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html; charset=utf-8')
                    self.end_headers()
                    self.wfile.write(html.encode('utf-8'))
                
                def serve_result_page(self, choice):
                    html = f"""
                    <html><head><title>Selection Made</title></head>
                    <body style="font-family: Arial; text-align: center; padding: 50px;">
                    <h2>Selection Confirmed</h2>
                    <p>You selected <strong>Option {choice}</strong></p>
                    <p>You can close this browser window now.</p>
                    <script>setTimeout(() => window.close(), 3000);</script>
                    </body></html>
                    """
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html; charset=utf-8')
                    self.end_headers()
                    self.wfile.write(html.encode('utf-8'))
                
                def serve_image(self, image_path):
                    try:
                        if os.path.exists(image_path):
                            with open(image_path, 'rb') as f:
                                content = f.read()
                            self.send_response(200)
                            self.send_header('Content-type', 'image/png')
                            self.end_headers()
                            self.wfile.write(content)
                        else:
                            self.send_error(404)
                    except:
                        self.send_error(500)
                
                def generate_selection_html(self):
                    task_name = self.current_task
                    
                    # Generate candidate cards
                    cards_html = ""
                    for i, analysis in enumerate(self.analyses, 1):
                        confidence = analysis.get('confidence', 0)
                        reasoning = analysis.get('reasoning', 'No reasoning provided')
                        image_path = analysis.get('image_path', '')
                        
                        # Color based on confidence
                        if confidence >= 70:
                            color = "#4CAF50"  # Green
                        elif confidence >= 60:
                            color = "#FF9800"  # Orange  
                        else:
                            color = "#E91E63"  # Pink
                        
                        cards_html += f"""
                        <div style="border: 2px solid {color}; margin: 20px; padding: 15px; border-radius: 10px; display: inline-block; width: 300px; vertical-align: top;">
                            <h3>Option {i} - Confidence: {confidence}%</h3>
                            <img src="/image/{urllib.parse.quote(image_path)}" style="width: 280px; height: auto; border: 1px solid #ccc;">
                            <p style="font-size: 14px; margin: 10px 0;"><strong>Reasoning:</strong> {reasoning}</p>
                            <button onclick="selectOption({i})" style="background: {color}; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; font-size: 16px;">
                                Select Option {i}
                            </button>
                        </div>
                        """
                    
                    # Show comparison image if available
                    comparison_html = ""
                    if self.comparison_path and os.path.exists(self.comparison_path):
                        comparison_html = f"""
                        <div style="margin: 30px 0;">
                            <h3>Visual Comparison</h3>
                            <img src="/image/{urllib.parse.quote(self.comparison_path)}" style="max-width: 800px; border: 2px solid #333;">
                        </div>
                        """
                    
                    return f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>Candidate Selection</title>
                        <meta charset="utf-8">
                        <style>
                            body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
                            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }}
                            .header {{ text-align: center; margin-bottom: 30px; }}
                            .task-info {{ background: #e3f2fd; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                        </style>
                    </head>
                    <body>
                        <div class="container">
                            <div class="header">
                                <h1>Multi-Object Disambiguation</h1>
                            </div>
                            
                            <div class="task-info">
                                <h3>Current Task: {task_name}</h3>
                            </div>
                            
                            {comparison_html}
                            
                            <div style="text-align: center;">
                                <h2>Please select the best candidate:</h2>
                                {cards_html}
                            </div>
                        </div>
                        
                        <script>
                            function selectOption(choice) {{
                                fetch('/select/' + choice)
                                    .then(() => {{
                                        document.body.innerHTML = '<div style="text-align: center; padding: 50px; font-family: Arial;"><h2>Selection Confirmed</h2><p>Option ' + choice + ' selected</p></div>';
                                    }});
                            }}
                        </script>
                    </body>
                    </html>
                    """
                
                def log_message(self, format, *args):
                    # Suppress server logs
                    pass
            
            # Setup web server
            result_container = {'selection': None}
            port = 8765
            
            def create_handler(*args, **kwargs):
                return CandidateHTTPHandler(*args, analyses=analyses, comparison_path=comparison_path, result_container=result_container, current_task=getattr(self, 'current_task', 'Navigation task'), **kwargs)
            
            server = HTTPServer(('localhost', port), create_handler)
            
            # Start server in background thread
            server_thread = threading.Thread(target=server.serve_forever, daemon=True)
            server_thread.start()
            
            # Open browser
            url = f"http://localhost:{port}"
            print(f"Web interface available at: {url}")
            
            try:
                webbrowser.open(url)
                print("Browser should open automatically...")
            except:
                print("[ERROR] Could not auto-open browser, please visit the URL manually")
            
            # Wait for selection with timeout
            timeout = getattr(self, 'human_selection_timeout', 60)
            start_time = time.time()
            
            while result_container['selection'] is None:
                time.sleep(0.5)
                if time.time() - start_time > timeout:
                    print(f"Timeout reached ({timeout}s). Using intelligent recommendation.")
                    # Use highest confidence as fallback
                    best_idx = max(range(len(analyses)), key=lambda i: analyses[i].get('confidence', 0))
                    result_container['selection'] = best_idx + 1
                    break
            
            server.shutdown()
            
            return result_container['selection']
            
        except Exception as e:
            print(f"[Warning] Web interface failed: {e}")
            return None

    def create_gui_selection_window(self, analyses, comparison_path):
        """Create GUI window for visual candidate selection"""
        try:
            import tkinter as tk
            from tkinter import ttk
            from PIL import Image, ImageTk
            import os
            
            class CandidateSelector:
                def __init__(self, analyses, comparison_path, timeout, current_task):
                    self.selection = None
                    self.analyses = analyses
                    self.timeout = timeout
                    self.current_task = current_task
                    
                    # Create main window
                    self.root = tk.Tk()
                    self.root.title("Multi-Object Disambiguation - Select Candidate")
                    self.root.geometry("1200x800")
                    self.root.configure(bg='#f0f0f0')
                    
                    # Create main frame
                    main_frame = ttk.Frame(self.root, padding="10")
                    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
                    
                    # Title
                    title_label = ttk.Label(main_frame, text="Choose the Best Candidate Object", 
                                          font=("Arial", 16, "bold"))
                    title_label.grid(row=0, column=0, columnspan=3, pady=(0, 15))
                    
                    # Task info
                    task_info = f"Task: {self.current_task}"
                    task_label = ttk.Label(main_frame, text=task_info, font=("Arial", 12))
                    task_label.grid(row=1, column=0, columnspan=3, pady=(0, 10))
                    
                    # Load and display comparison image if available
                    if comparison_path and os.path.exists(comparison_path):
                        try:
                            comparison_img = Image.open(comparison_path)
                            # Resize if too large
                            if comparison_img.width > 1000:
                                ratio = 1000 / comparison_img.width
                                new_size = (1000, int(comparison_img.height * ratio))
                                comparison_img = comparison_img.resize(new_size, Image.Resampling.LANCZOS)
                            
                            comparison_photo = ImageTk.PhotoImage(comparison_img)
                            comparison_label = ttk.Label(main_frame, image=comparison_photo)
                            comparison_label.image = comparison_photo  # Keep reference
                            comparison_label.grid(row=2, column=0, columnspan=3, pady=(0, 15))
                        except Exception as e:
                            print(f"Warning: Could not load comparison image: {e}")
                    
                    # Create buttons for each candidate
                    button_frame = ttk.Frame(main_frame)
                    button_frame.grid(row=3, column=0, columnspan=3, pady=(10, 0))
                    
                    # Sort analyses by confidence for display
                    sorted_analyses = sorted(analyses, key=lambda x: x['confidence'], reverse=True)
                    
                    for i, analysis in enumerate(sorted_analyses):
                        btn_text = f"Option {analysis['index']}\nConfidence: {analysis['confidence']}%\n{analysis['spatial_desc']}"
                        
                        # Color code by confidence
                        if analysis['confidence'] >= 70:
                            btn_color = '#90EE90'  # Light green
                        elif analysis['confidence'] >= 60:
                            btn_color = '#FFE4B5'  # Light yellow
                        else:
                            btn_color = '#FFB6C1'  # Light pink
                        
                        btn = tk.Button(button_frame, text=btn_text, width=20, height=4,
                                      bg=btn_color, font=("Arial", 10),
                                      command=lambda idx=analysis['index']: self.select_option(idx))
                        btn.grid(row=0, column=i, padx=10, pady=5)
                    
                    # Control buttons
                    control_frame = ttk.Frame(main_frame)
                    control_frame.grid(row=4, column=0, columnspan=3, pady=(15, 0))
                    
                    auto_btn = tk.Button(control_frame, text="Use AI Recommendation", 
                                       bg='#87CEEB', font=("Arial", 11),
                                       command=lambda: self.select_option('auto'))
                    auto_btn.grid(row=0, column=0, padx=10)
                    
                    # Timeout display
                    self.timeout_label = ttk.Label(control_frame, text=f"Auto-select in: {timeout}s", 
                                                 font=("Arial", 10))
                    self.timeout_label.grid(row=0, column=1, padx=20)
                    
                    # Start timeout countdown
                    self.remaining_time = timeout
                    self.update_timeout()
                    
                    # Center window
                    self.root.update_idletasks()
                    x = (self.root.winfo_screenwidth() // 2) - (self.root.winfo_width() // 2)
                    y = (self.root.winfo_screenheight() // 2) - (self.root.winfo_height() // 2)
                    self.root.geometry(f"+{x}+{y}")
                
                def select_option(self, choice):
                    self.selection = choice
                    self.root.quit()
                    self.root.destroy()
                
                def update_timeout(self):
                    if self.remaining_time > 0 and self.selection is None:
                        self.timeout_label.config(text=f"Auto-select in: {self.remaining_time}s")
                        self.remaining_time -= 1
                        self.root.after(1000, self.update_timeout)
                    elif self.selection is None:
                        # Timeout reached
                        self.select_option('auto')
                
                def run(self):
                    self.root.mainloop()
                    return self.selection
            
            # Create and run selector
            selector = CandidateSelector(analyses, comparison_path, self.human_selection_timeout)
            selection = selector.run()
            
            return selection if selection else 'auto'
            
        except ImportError:
            print("[ERROR] GUI libraries not available, falling back to text interface")
            return self.get_user_input_with_text_fallback(analyses)
        except Exception as e:
            print(f"[ERROR] GUI error: {e}, falling back to text interface")
            return self.get_user_input_with_text_fallback(analyses)
    
    def get_user_input_with_text_fallback(self, analyses):
        """Text fallback for when GUI is not available"""
        print(f"\n>>> Your choice: ", end='', flush=True)
        
        try:
            import select
            import sys
            
            # Check if input is available within timeout
            ready, _, _ = select.select([sys.stdin], [], [], self.human_selection_timeout)
            if ready:
                response = sys.stdin.readline().strip()
                
                # Handle special commands
                if response.lower().startswith('vlm '):
                    try:
                        option_num = int(response.split()[1])
                        for analysis in analyses:
                            if analysis['index'] == option_num:
                                vlm_path = analysis.get('vlm_viz_path')
                                if vlm_path:
                                    print(f"\nVLM Analysis for Option {option_num}: {vlm_path}")
                                else:
                                    print(f"\n[ERROR] No VLM record available for Option {option_num}")
                                break
                        return self.get_user_input_with_text_fallback(analyses)  # Ask again
                    except:
                        print("\n❓ Invalid VLM command. Use format: 'vlm 1' or 'vlm 2'")
                        return self.get_user_input_with_text_fallback(analyses)
                
                elif response.lower() == 'show':
                    # Re-display comparison if available
                    if analyses and 'comparison_path' in dir(self):
                        print(f"\nComparison image: {self.comparison_path}")
                    return self.get_user_input_with_text_fallback(analyses)  # Ask again
                
                return response
            else:
                print(f"\nTimeout reached ({self.human_selection_timeout}s). Using intelligent recommendation.")
                return "auto"
                
        except Exception as e:
            print(f"\n[ERROR] Input error: {e}. Using intelligent recommendation.")
            return "auto"
    
    def get_user_input_with_image_support(self, analyses):
        """Enhanced user input with GUI support"""
        # Try GUI first, fallback to text if needed
        comparison_path = getattr(self, 'comparison_path', None)

        # Extract object type from analyses
        object_type = 'Object'
        if analyses and len(analyses) > 0:
            first_analysis = analyses[0]
            if 'object' in first_analysis and 'objectType' in first_analysis['object']:
                object_type = first_analysis['object']['objectType']

        # Prioritize trying Web Dashboard's disambiguation functionality
        try:
            from web_ui import start_disambiguation_web
            web_disambiguation_data = {
                'task_name': getattr(self, 'current_task', 'Navigation task'),
                'object_type': object_type,
                'candidates': [
                    {
                        'image_path': analysis.get('image_path', ''),
                        'confidence': analysis.get('confidence', 0),
                        'reasoning': analysis.get('analysis', analysis.get('spatial_desc', 'No analysis available'))
                    }
                    for analysis in analyses
                ]
            }
            
            web_result = start_disambiguation_web(web_disambiguation_data, timeout=self.human_selection_timeout)
            if web_result is not None:
                if web_result == -1:
                    # Special value: timeout, need VLM analysis
                    print(f"[INFO] Web Dashboard timeout: Triggering VLM analysis")
                    return 'timeout'  # Signal to trigger VLM analysis
                else:
                    # Normal human selection
                    print(f"[INFO] Web Dashboard selection: Option {web_result}")
                    return web_result
        except ImportError:
            # Web dashboard not available, continue using other methods
            pass
        except Exception as e:
            print(f"[ERROR] Web Dashboard disambiguation failed: {e}")
        
        # Fallback to GUI if DISPLAY available
        try:
            import os
            if os.environ.get('DISPLAY') or os.name == 'nt':  # Has display or Windows
                return self.create_gui_selection_window(analyses, comparison_path)
        except Exception as e:
            print(f"[ERROR] GUI interface failed: {e}")
        
        # Final fallback to text interface
        print("Using text interface")
        return self.get_user_input_with_text_fallback(analyses)
    
    def request_user_disambiguation_improved(self, itemtype, candidates):
        """Main enhanced disambiguation flow with configurable modes"""
        print(f"\nStarting ENHANCED disambiguation for {len(candidates)} {itemtype} objects...")
        print(f"Task Context: {getattr(self, 'current_task', 'Not set')}")
        print(f"Disambiguation Mode: {self.disambiguation_mode}")
        
        task_description = getattr(self, 'current_task', f'Navigate to {itemtype}')
        
        if self.disambiguation_mode == "human_first_vlm_fallback":
            # Mode 1: Human-first with VLM fallback
            print("[HUMAN FIRST] Taking photos for human selection...")
            analyses = self.take_candidate_photos_only(task_description, candidates)
            
            if analyses is None or all(a['analysis_quality'] == 'navigation_failed' for a in analyses):
                print(f"[FALLBACK] Photo capture failed, using first object: {itemtype}")
                return candidates[0]
            
            # Generate visual message for human selection
            message, comparison_path = self.generate_visual_disambiguation_message(candidates, analyses)
            self.comparison_path = comparison_path
            
            print(f"[HUMAN SELECTION] Waiting up to {self.human_selection_timeout} seconds for human choice...")
            
            # Get human input with timeout
            try:
                choice = self.get_user_input_with_image_support(analyses)
                
                if choice and choice != 'auto' and choice != 'timeout':
                    print(f"[HUMAN SELECTED] Using human choice: {choice}")
                    return self.get_object_by_choice(choice, analyses)
                else:
                    print("[TIMEOUT/AUTO] Human selection timed out, falling back to VLM analysis...")
                    # Fallback: Run VLM analysis on the photos we already took
                    vlm_analyses = self.run_vlm_analysis_on_photos(analyses, task_description)

                    # Update Web UI history with VLM analysis results
                    self.update_web_ui_disambiguation_history(vlm_analyses, task_description, itemtype)

                    # If VLM analysis produces valid results, use them
                    if vlm_analyses and any(analysis.get('confidence', 0) > 25 for analysis in vlm_analyses):
                        return self.select_best_candidate_from_vlm(vlm_analyses, itemtype)
                    else:
                        # VLM failed, randomly select a candidate
                        import random
                        selected = random.choice(candidates)
                        selected_index = candidates.index(selected) + 1
                        print(f"[FALLBACK] VLM analysis failed, randomly selecting {itemtype}_{selected_index}")
                        return selected

            except Exception as e:
                print(f"[ERROR] Human selection failed: {e}, randomly selecting candidate")
                import random
                selected = random.choice(candidates)
                selected_index = candidates.index(selected) + 1
                print(f"[FALLBACK] Randomly selecting {itemtype}_{selected_index}")
                return selected

        elif self.disambiguation_mode == "vlm_first_human_choice":
            # Mode 2: VLM analysis first, human choice with confidence scores
            print("[VLM FIRST] Running VLM analysis to provide confidence scores...")
            analyses = self.analyze_candidates_with_vlm(task_description, candidates)

            # Check if VLM analysis failed completely
            if analyses is None:
                import random
                selected = random.choice(candidates)
                selected_index = candidates.index(selected) + 1
                print(f"[FALLBACK] VLM analysis failed, randomly selecting {itemtype}_{selected_index}")
                return selected

            # Update Web UI with VLM confidence scores immediately
            self.update_web_ui_disambiguation_history(analyses, task_description, itemtype)

            # Generate visual disambiguation message with VLM confidence scores
            message, comparison_path = self.generate_visual_disambiguation_message(candidates, analyses)
            self.comparison_path = comparison_path

            print(f"[VLM ANALYSIS] VLM has analyzed all candidates. Confidence scores:")
            for analysis in analyses:
                print(f"   • {itemtype}_{analysis['index']}: {analysis['confidence']}% - {analysis.get('analysis', 'No reasoning')[:80]}...")

            print(f"[HUMAN CHOICE] Please make your selection based on VLM analysis...")

            # Get human input with VLM-provided confidence scores
            choice = self.get_user_input_with_image_support(analyses)

            if choice and choice != 'auto' and choice != 'timeout':
                print(f"[HUMAN SELECTED] Using human choice: {choice}")
                return self.get_object_by_choice(choice, analyses)
            else:
                # Human didn't select, use VLM recommendation
                print("[AUTO] Human didn't select, using VLM recommendation...")
                return self.select_best_candidate_from_vlm(analyses, itemtype)

        elif self.disambiguation_mode == "human_only_random_fallback":
            # Mode 3: Human only, random selection as fallback
            print("[HUMAN ONLY] Taking photos for human selection (no VLM analysis)...")
            analyses = self.take_candidate_photos_only(task_description, candidates)

            if analyses is None or all(a['analysis_quality'] == 'navigation_failed' for a in analyses):
                print(f"[FALLBACK] Photo capture failed, using first object: {itemtype}")
                return candidates[0]

            # Generate visual message for human selection (no VLM confidence scores)
            message, comparison_path = self.generate_visual_disambiguation_message(candidates, analyses)
            self.comparison_path = comparison_path

            print(f"[HUMAN SELECTION] Waiting up to {self.human_selection_timeout} seconds for human choice...")

            # Get human input with timeout
            choice = self.get_user_input_with_image_support(analyses)

            if choice and choice != 'auto' and choice != 'timeout':
                print(f"[HUMAN SELECTED] Using human choice: {choice}")
                return self.get_object_by_choice(choice, analyses)
            else:
                # Human didn't select, random fallback (no VLM)
                print("[RANDOM FALLBACK] Human selection timed out, randomly selecting...")
                import random
                selected = random.choice(candidates)
                selected_index = candidates.index(selected) + 1
                print(f"[RANDOM] Randomly selected {itemtype}_{selected_index}")
                return selected

        else:
            # Invalid mode, fallback to random selection
            print(f"[ERROR] Unknown disambiguation mode: {self.disambiguation_mode}")
            print(f"[FALLBACK] Using random selection...")
            import random
            selected = random.choice(candidates)
            selected_index = candidates.index(selected) + 1
            print(f"[RANDOM] Randomly selected {itemtype}_{selected_index}")
            return selected

    def update_web_ui_disambiguation_history(self, vlm_analyses, task_description, itemtype):
        """Update Web UI disambiguation history with VLM analysis results"""
        try:
            if not vlm_analyses:
                return

            # Try to update the Web UI monitor directly
            try:
                from web_ui.server import monitor
                print(f"[DEBUG] Monitor import successful: {monitor is not None}")

                if monitor and hasattr(monitor, 'disambiguation_history'):
                    print(f"[DEBUG] Monitor has disambiguation_history: {len(monitor.disambiguation_history)} entries")

                    # Find the most recent disambiguation entry and update it
                    if monitor.disambiguation_history:
                        latest_entry = monitor.disambiguation_history[-1]
                        print(f"[DEBUG] Latest entry candidates: {len(latest_entry.get('candidates', []))}")
                        print(f"[DEBUG] VLM analyses: {len(vlm_analyses)}")

                        # Update the candidates with VLM analysis results
                        if 'candidates' in latest_entry:
                            for i, analysis in enumerate(vlm_analyses):
                                if i < len(latest_entry['candidates']):
                                    old_confidence = latest_entry['candidates'][i]['confidence']
                                    new_confidence = analysis.get('confidence', 25)
                                    # Update confidence and reasoning with VLM results
                                    latest_entry['candidates'][i]['confidence'] = new_confidence
                                    latest_entry['candidates'][i]['reasoning'] = analysis.get('analysis', 'VLM analysis completed')

                                    print(f"[DEBUG] Updated candidate {i+1}: {old_confidence}% → {new_confidence}%")

                            # Add a note about VLM analysis completion
                            latest_entry['vlm_analysis_completed'] = True
                            latest_entry['final_selection_method'] = 'vlm_analysis'

                            print(f"[WEB UI] Updated disambiguation history with VLM analysis results")
                        else:
                            print(f"[DEBUG] No 'candidates' key in latest entry: {latest_entry.keys()}")
                    else:
                        print(f"[DEBUG] No disambiguation history entries found")
                else:
                    print(f"[DEBUG] Monitor check failed: monitor={monitor is not None}, has_history={hasattr(monitor, 'disambiguation_history') if monitor else False}")

            except ImportError:
                # Web UI not available
                pass
            except Exception as e:
                print(f"[WARNING] Failed to update Web UI history: {e}")

        except Exception as e:
            print(f"[WARNING] Error updating Web UI disambiguation history: {e}")

    # ==================== ENHANCED FEATURES INITIALIZATION ====================
    
    def enable_enhanced_features(self, enable_indexing=True, enable_dialogue=True, confidence_threshold=30, timeout=30):
        """
        Enable enhanced multi-object disambiguation features
        """
        print(f"Configuring Enhanced Features (9.4)...")
        
        self.enable_object_indexing = enable_indexing
        self.enable_dialogue_system = enable_dialogue
        self.confidence_gap_threshold = confidence_threshold
        self.human_selection_timeout = timeout
        
        # Initialize object indexing if enabled
        if enable_indexing:
            self.init_object_indexing()
        
        # Initialize task context storage
        if not hasattr(self, 'current_task'):
            self.current_task = None
        if not hasattr(self, 'current_subtasks'):
            self.current_subtasks = []
        if not hasattr(self, 'task_context'):
            self.task_context = {}
        if not hasattr(self, 'candidate_images'):
            self.candidate_images = []
        
        print(f"Enhanced Features Configuration:")
        print(f"   • Object Indexing: {'[ENABLED]' if enable_indexing else '[DISABLED]'}")
        print(f"   • VLM Dialogue System: {'[ENABLED]' if enable_dialogue else '[DISABLED]'}")
        print(f"   • Confidence Threshold: {confidence_threshold}%")
        print(f"   • User Input Timeout: {timeout}s")
        
        return self


if __name__ == "__main__":
    autogn = RocAgent("", visibilityDistance=10, fieldOfView=90)
    autogn.get_all_item_image()
    autogn.example()
    autogn.init_agent_corner()
    autogn.test_visibility()
    autogn.get_navigate_path()
    autogn.controller.stop()
    