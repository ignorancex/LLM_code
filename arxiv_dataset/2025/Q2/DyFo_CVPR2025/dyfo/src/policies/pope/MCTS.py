from .policy import QuestionSample as BaseQuestionSample
import shortuuid
import base64
import io
from PIL import Image
import numpy as np
import random
import math
import aiohttp
import traceback
import re

class MCTSNode:
    """MCTS Tree Node Class"""
    def __init__(self, state, parent=None, available_actions=None):
        self.state = state  # Node state
        self.parent = parent  # Parent node
        self.children = {}  # Child nodes
        self.visits = 0  # Visit count
        self.value = 0  # Cumulative reward
        self.leaf_reward = 0  # Reward as leaf node
        # Initialize untried actions
        self.untried_actions = available_actions.copy() if available_actions else []
        # Store expert information
        self.expert_info = None
        # Store valid area ratio of current image (initial 1.0 means full area is valid)
        self.valid_area_ratio = 1.0
        # Store region coordinates of current image relative to original
        self.region_coords = state.get('region_coords', (0, 0, state['image_width'], state['image_height']))
        # Additional information storage dictionary
        self.extra_info = {}

class MCTSQuestionSample(BaseQuestionSample):
    def __init__(self, row, args, round_idx=0):
        super().__init__(row, args, round_idx)
        # Get image dimensions
        image_bytes = base64.b64decode(self.image)
        img = Image.open(io.BytesIO(image_bytes))
        self.image_width, self.image_height = img.size
        
        # Create 32x32 blank image
        blank_image = Image.new('RGB', (32, 32), color='white')
        buffered = io.BytesIO()
        blank_image.save(buffered, format="PNG")
        self.blank_image = base64.b64encode(buffered.getvalue()).decode()
        
        # MCTS parameters
        self.max_depth = 3  # Maximum search depth
        self.c_puct = 1.0  # PUCT constant
        self.n_simulations = 2  # Simulation count
        self.use_ensemble = True # Whether to use ensemble
        
        # Define action space
        self.actions = [
            "repeat_question",
            "zoom_out"  # New zoom action
        ]
        
        # Define action prompts
        self.action_prompts = {
            "repeat_question": "Repeat the question.",
            "zoom_out": "Zoom out the region by 1.5x"  # New zoom action prompt
        }
        
        # Define action executor mapping
        self.action_executors = {
            "repeat_question": self.execute_repeat_question_action,
            "zoom_out": self.execute_zoom_out_action  # New zoom action executor
        }
        
        # MCTS root node
        self.root = None
        
        # Visual expert API
        self.expert_ports = [1]  # Multiple expert ports, corresponding to port+8000
        self.expert_ports = [port + 8000 for port in self.expert_ports]
        self.expert_base_url = "http://localhost:{}/predict"

    async def extract_key_objects(self):
        """Extract key objects from question"""
        if 'llava' in self.args.model_path:
            # Use improved extraction method
            # Preprocess question text
            question = self.row['question'].replace('?', '')
            stop_words = ['is', 'in the image', 'IS', 'THE', 'IMAGE','what','color of','there', 'a', 'an', 'How']
            for word in stop_words:
                question = re.sub(r'\b' + word + r'\b', '', question, flags=re.IGNORECASE)
            question = ' '.join(question.split())
            
            # Extract objects
            prompt = f"Task: List objects mentioned in text in List format.\nInput text: {question}\nAction: What objects are mentioned in original text? List separated by commas. For example, from \"person with white trousers on the left or right side of the person in blue\", output \"[\"person with white trousers\", \"person in blue\"]\"."
            response = await self.generate(prompt, self.blank_image, max_tokens=50)
            
            # Try to parse as list format
            try:
                objects = eval(response)
            except:
                response = response.replace('[', '').replace(']', '').replace('"', '')
                objects = response.split(',')
            
            # Print objects
            # print("objects",objects)

            # Exclude non-list types
            if not isinstance(objects,list):
                objects = [question]

            # Filter objects
            filtered_objects = []
            for obj in objects:
                obj = obj.strip().lower()
                if obj in question.lower():
                    filtered_objects.append(obj)
                    
            # If filtered result is empty, use original question
            if not filtered_objects:
                filtered_objects = [question]
                    
            return filtered_objects
            
        else:
            # Use original extraction method
            prompt = f"Task: Extract all objects (including people) with their complete descriptions from the question. For example, from 'Is the person with white trousers on the left or right side of the person in blue?', extract 'person with white trousers' and 'person in blue'.\nQuestion: {self.row['question']}\nAction: Only list the objects separated by commas."
            response = await self.generate(prompt, self.blank_image, max_tokens=50)
            
            if "object" in response.lower() or "description" in response.lower():
                objects = response.split()[-1].lower()

            # Split response into list and remove whitespace
            objects = [obj.strip() for obj in response.split(',')]
            return objects
        
    async def get_expert_boxes(self, image, text):
        """Call visual expert to get boxes"""
        try:
            # Randomly select expert
            port = random.choice(self.expert_ports)
            
            expert_url = self.expert_base_url.format(port)
            timeout = aiohttp.ClientTimeout(total=10000)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    expert_url,
                    json={
                        "image": image,  # image is already base64 string
                        "text": text
                    }
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        print(f"Visual expert API returned error status: {response.status}")
                        print(f"Error message: {error_text}")
                        print(f"Request URL: {expert_url}")
                        print(f"Request text: {text}")
                        return None
        except Exception as e:
            print(f"Error calling visual expert: {str(e)}")
            print(f"Request URL: {expert_url}")
            print(f"Request text: {text}")
            print(f"Exception stack: {traceback.format_exc()}")
            return None

    def selection(self, node):
        """Selection phase: Use UCB algorithm to select best child node"""
        # If node has untried actions, return current node for expansion
        if node.untried_actions:
            return node
            
        if not node.children:
            return node
            
        total_visits = sum(child.visits for child in node.children.values())
        
        def ucb_score(child):
            exploit = child.value / child.visits if child.visits > 0 else 0
            explore = math.sqrt(2 * math.log(total_visits) / (child.visits + 1e-8))
            return exploit + self.c_puct * explore
            
        best_child = max(node.children.values(), key=ucb_score)
        return self.selection(best_child)

    async def execute_repeat_question_action(self, node):
        """Execute repeat question action"""
        node_text = self.row['question']
        expert_result = await self.get_expert_boxes(node.state['image'], node_text)
        
        # If expert result obtained and contains boxes
        if expert_result and expert_result.get('boxes'):
            # Convert all boxes to numpy array
            boxes = np.array(expert_result['boxes'])
            
            # Calculate union of all boxes
            x1 = np.min(boxes[:, 0])
            y1 = np.min(boxes[:, 1]) 
            x2 = np.max(boxes[:, 2])
            y2 = np.max(boxes[:, 3])
            
            # Add padding
            padding = 20
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(node.state['image_width'], x2 + padding)
            y2 = min(node.state['image_height'], y2 + padding)
            
            # Calculate new valid area ratio
            new_area = (x2 - x1) * (y2 - y1)
            total_area = node.state['image_width'] * node.state['image_height']
            valid_area_ratio = new_area / total_area
            
            # Crop image
            image_bytes = base64.b64decode(node.state['image'])
            img = Image.open(io.BytesIO(image_bytes))
            cropped_img = img.crop((x1, y1, x2, y2))
            
            # Convert cropped image back to base64
            buffered = io.BytesIO()
            cropped_img.save(buffered, format="PNG")
            cropped_image_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Update region_coords, considering parent node's coordinate offset
            parent_x1, parent_y1, _, _ = node.state['region_coords']
            new_region_coords = (
                parent_x1 + x1,
                parent_y1 + y1,
                parent_x1 + x2,
                parent_y1 + y2
            )
        else:
            # If no boxes obtained, use original image and region
            cropped_image_base64 = node.state['image']
            valid_area_ratio = node.valid_area_ratio
            new_region_coords = node.state['region_coords']
        
        # Create new state
        new_state = {
            'depth': node.state['depth'] + 1,
            'image': cropped_image_base64,
            'action_history': node.state['action_history'] + [self.action_prompts["repeat_question"]],
            'text': node_text,
            'image_width': node.state['image_width'],
            'image_height': node.state['image_height'],
            'region_coords': new_region_coords
        }
        
        # Create new node
        child = MCTSNode(new_state, parent=node, available_actions=self.actions)
        child.expert_info = expert_result
        child.valid_area_ratio = valid_area_ratio
        
        return child

    async def execute_zoom_out_action(self, node):
        """Execute zoom out region action"""
        # Get current region coordinates
        x1, y1, x2, y2 = node.state['region_coords']
        
        # Calculate region center
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Calculate current region dimensions
        width = x2 - x1
        height = y2 - y1
        
        # Zoom out by 1.5x
        new_width = width * 1.5
        new_height = height * 1.5
        
        # Calculate new region coordinates
        new_x1 = max(0, center_x - new_width/2)
        new_y1 = max(0, center_y - new_height/2)
        new_x2 = min(node.state['image_width'], center_x + new_width/2)
        new_y2 = min(node.state['image_height'], center_y + new_height/2)
        
        # Crop original image
        image_bytes = base64.b64decode(self.image)
        img = Image.open(io.BytesIO(image_bytes))
        cropped_img = img.crop((new_x1, new_y1, new_x2, new_y2))
        
        # Convert cropped image to base64
        buffered = io.BytesIO()
        cropped_img.save(buffered, format="PNG")
        cropped_image_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        final_x1, final_y1, final_x2, final_y2 = new_x1, new_y1, new_x2, new_y2
        
        # If parent node has missing_objects, try to find them in zoomed region
        if 'missing_objects' in node.state and node.state['missing_objects']:
            missing_objects_text = ', '.join(node.state['missing_objects'])
            expert_result = await self.get_expert_boxes(cropped_image_base64, missing_objects_text)
            
            # If boxes found, calculate union
            if expert_result and expert_result.get('boxes'):
                boxes = np.array(expert_result['boxes'])
                # Calculate union of expert boxes
                expert_x1 = np.min(boxes[:, 0]) + new_x1
                expert_y1 = np.min(boxes[:, 1]) + new_y1
                expert_x2 = np.max(boxes[:, 2]) + new_x1
                expert_y2 = np.max(boxes[:, 3]) + new_y1
                
                # Calculate union with parent node region
                final_x1 = min(x1, expert_x1)
                final_y1 = min(y1, expert_y1)
                final_x2 = max(x2, expert_x2)
                final_y2 = max(y2, expert_y2)
                
                # Re-crop image
                cropped_img = img.crop((final_x1, final_y1, final_x2, final_y2))
                buffered = io.BytesIO()
                cropped_img.save(buffered, format="PNG")
                cropped_image_base64 = base64.b64encode(buffered.getvalue()).decode()
        else:
            expert_result = await self.get_expert_boxes(cropped_image_base64, ", ".join(self.key_objects))
        
        # Create new state
        new_state = {
            'depth': node.state['depth'] + 1,
            'image': cropped_image_base64,
            'action_history': node.state['action_history'] + [self.action_prompts["zoom_out"]],
            'text': node.state['text'],
            'image_width': node.state['image_width'],
            'image_height': node.state['image_height'],
            'region_coords': (final_x1, final_y1, final_x2, final_y2)
        }
        
        # Create new node
        child = MCTSNode(new_state, parent=node, available_actions=self.actions)
        child.expert_info = expert_result
            
        # Calculate new valid area ratio
        new_area = (final_x2 - final_x1) * (final_y2 - final_y1)
        total_area = node.state['image_width'] * node.state['image_height']
        child.valid_area_ratio = new_area / total_area
        
        return child

    async def expansion(self, node):
        """Expansion phase: Add new child node"""
        if node.state['depth'] >= self.max_depth or not node.untried_actions:
            return node
            
        action = random.choice(node.untried_actions)
        node.untried_actions.remove(action)
        
        # Call corresponding executor based on action
        child = await self.action_executors[action](node)
        node.children[action] = child
        
        return child

    async def simulation(self, node):
        """Simulation phase: execute actions and obtain rewards"""    
        # Get key objects
        key_objects = self.key_objects
        
        # Ask separately for each key object if it's in image
        all_objects_present = True
        confirmed_objects = []
        missing_objects = []

        try:
            for obj in key_objects:
                # Generate question asking if object is in image
                prompt = f"Task: Only answer yes or no.\nQuestion: Is there a {obj} in this image?"
                response = await self.generate(prompt, node.state['image'], max_tokens=10)
                
                # If object is in image, add to confirmed list
                if 'yes' in response.lower():
                    confirmed_objects.append(obj)
                else:
                    missing_objects.append(obj)
                    all_objects_present = False
                    break

        except Exception as e:
            print(f"Error during simulation: {e}")
            print(f"Key objects being processed: {key_objects}")
            raise  # Continue propagating exception
                
        # Record confirmed and missing objects
        node.state['caption'] = ', '.join(confirmed_objects)
        node.state['missing_objects'] = missing_objects
        
        # Only give reward if all key objects are in image
        if all_objects_present:
            # Reward is inversely proportional to valid area ratio
            reward = 1 - node.valid_area_ratio
        else:
            reward = 0
            
        return reward

    def backpropagation(self, node, reward):
        """Backpropagation phase: Update node values"""
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent
        
    async def single_run(self, root_state):
        """Single MCTS process"""
        if not self.root:
            # Create temporary root node
            temp_root = MCTSNode(root_state, available_actions=self.actions)
            # Execute repeat_question_action to get real root node
            self.root = await self.execute_repeat_question_action(temp_root)
            self.root.parent = None
            
        # 1. Selection
        node = self.selection(self.root)
        
        if node.state['depth'] >= self.max_depth:
            return 0
            
        # 2. Expansion
        node = await self.expansion(node)
            
        # 3. Simulation
        reward = await self.simulation(node)
        
        # Update leaf node reward
        node.leaf_reward = reward

        # 4. Backpropagation
        self.backpropagation(node, reward)
        
        return reward

    async def get_final_answer(self):
        """Run MCTS search for best answer"""
        initial_state = {
            'depth': 0,
            'image': self.image,
            'action_history': [],
            'text': self.row['question'],  # Root node uses original question as text
            'image_width': self.image_width,
            'image_height': self.image_height,
            'region_coords': (0, 0, self.image_width, self.image_height)
        }
        # Run multiple simulations
        for _ in range(self.n_simulations):
            await self.single_run(initial_state)
            
        # Collect all nodes
        all_nodes = []
        nodes_to_visit = [self.root]
        while nodes_to_visit:
            node = nodes_to_visit.pop()
            all_nodes.append(node)
            nodes_to_visit.extend(node.children.values())
            
        # Generate final question
        final_qs = self.row['question']
            
        # Generate answers for each node
        answers = []
        for node in all_nodes:
            answer = await self.generate(final_qs, node.state['image'])
            # Use leaf_reward as weight
            weight = node.leaf_reward if node.leaf_reward is not None else 0
            answers.append(('yes' if 'yes' in answer.lower() else 'no', weight))
        
        # Find node with highest value/visits
        best_node = max(all_nodes, key=lambda x: (x.leaf_reward, all_nodes.index(x)))
        
        if self.use_ensemble:
            # Weighted voting to determine final answer
            from collections import defaultdict
            vote_result = defaultdict(float)
            for answer, weight in answers:
                vote_result[answer] += weight
                
            # Check if all weights are zero
            if all(weight == 0 for weight in vote_result.values()):
                # Regenerate answer using original image
                final_answer = await self.generate(final_qs, self.image)
                final_answer = 'yes' if 'yes' in final_answer.lower() else 'no'
            else:
                final_answer = max(vote_result, key=vote_result.get)
        else:
            # Use best_node's answer
            final_answer = max(answers, key=lambda x: x[1])[0]
        
        return final_answer, final_qs, answers[-1][0], best_node.state['image'], best_node, self.root

    def serialize_tree(self, node):
        """Serialize tree structure for saving to jsonl"""
        node_info = {
            "state": node.state,
            "visits": node.visits, 
            "value": node.value,
            "leaf_reward": node.leaf_reward,
            "expert_info": node.expert_info,
            "valid_area_ratio": node.valid_area_ratio,
            "region_coords": node.region_coords,
            "extra_info": node.extra_info,
            "children": {action: self.serialize_tree(child) for action, child in node.children.items()}
        }
        return node_info

    async def _process(self):
        # Extract key objects from question
        self.key_objects = await self.extract_key_objects()
        
        final_answer, prompt, full_answer, final_image, best_node, root_node = await self.get_final_answer()
        
        # Serialize tree structure for saving
        # tree_info = self.serialize_tree(root_node)
        
        return {
            "question_id": self.row['index'],
            "round_id": self.round_idx,
            "prompt": prompt,
            "text": final_answer,
            "answer_id": shortuuid.uuid(),
            "model_id": self.args.model_path,
            "answer": self.row['answer'],
        }
