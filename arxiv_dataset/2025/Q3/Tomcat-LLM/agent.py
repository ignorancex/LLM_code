import numpy as np
from llm import get_response
from gemma import get_response_from_gemini
from deepseek import get_response_R1
from env_setup import GridEnvironment
class Agent:
    def __init__(self, start_pos):
        self.pos = start_pos  # Agent's initial position (row, col)
        self.keys_collected = []  # List of keys collected (color-coded)
        self.clarification_count = 0  # Track number of clarifications
        self.conversation_history = []  # Track conversation for inference
        self.max_clarifications = 3  # Maximum number of clarifications allowed
        self.previous_symbol = '.'  # Track what was at the agent's previous position
            

    def move(self, direction, grid):
        # Clear the agent's previous position by restoring the previous symbol
        grid[self.pos] = self.previous_symbol

        # Determine the new position based on direction
        new_pos = list(self.pos)
        if direction == 'up':
            new_pos[0] -= 1
        elif direction == 'down':
            new_pos[0] += 1
        elif direction == 'left':
            new_pos[1] -= 1
        elif direction == 'right':
            new_pos[1] += 1

        # Ensure the agent doesn't move through walls (W)
        if grid[new_pos[0], new_pos[1]] != 'W':
            # Update the agent's position
            self.pos = tuple(new_pos)
            # Save the symbol that was on the new position (before placing the agent there)
            self.previous_symbol = grid[self.pos]
            # Mark the agent's new position with 'm'
            grid[self.pos] = 'm'

    def pick_up_key(self, grid):
        # Use the `previous_symbol` to check if the agent was on a key
        if self.previous_symbol in ['r', 'y', 'b']:
            key_color = self.previous_symbol
            print(f"{key_color.capitalize()} key picked up!")
            self.keys_collected.append(key_color)
            self.previous_symbol = '.'  # The key is removed, so the cell becomes empty

    def unlock_door(self, grid):
        # Use the `previous_symbol` to check if the agent was next to a door (R, Y, B)
        if self.previous_symbol == 'R' and 'r' in self.keys_collected:
            print("Red door unlocked!")
            self.previous_symbol = '.'  # The door is removed, so the cell becomes empty
            self.keys_collected.remove('r')

        elif self.previous_symbol == 'Y' and 'y' in self.keys_collected:
            print("Yellow door unlocked!")
            self.previous_symbol = '.'  # The door is removed, so the cell becomes empty
            self.keys_collected.remove('y')

        elif self.previous_symbol == 'B' and 'b' in self.keys_collected:
            print("Blue door unlocked!")
            self.previous_symbol = '.'  # The door is removed, so the cell becomes empty
            self.keys_collected.remove('b')

    def collect_gem(self, grid):
        # Use the `previous_symbol` to check if the agent was on a gem
        if self.previous_symbol == 'g':
            print("Gem collected!")
            self.previous_symbol = '.'  # The gem is removed, so the cell becomes empty

    def chain_of_thought_prompt(self, instruction, grid, human, object_pos, exp, llm):
        # Generate the prompt with the current grid state and human position
        print("Generating chain-of-thought prompt...")
        prompt = self.create_prompt_ToM(instruction, grid, human, object_pos, exp)
        
        # Query the LLM with the prompt
        if llm == "gpt-4o":
            response = get_response(prompt)
        elif llm == "gemma3":
            response = get_response_from_gemini(prompt)
        elif llm == "R1":
            response = get_response_R1(prompt)
        
        # Log the instruction and response for later reference
        self.conversation_history.append(f"Instruction: {instruction}, Response: {response}")
        
        return response  # Returns the LLM's response, either an action or clarification
    
    def create_prompt_ToM(self, instruction, grid, human_move, object_pos, exp):
        intro = "You assist a human in a cooperative planning domain called Doors, Keys, and Gems, set in a gridworld. The human attempts to retrieve a specific gem, and you assist the human. You help by unlocking doors using the correct keys or passing the keys to the human. Think of it as a team game where a human gives callouts to their teammate (you), expecting collaboration to achieve their goal. The challenge for you is correctly understanding human instructions, which may require some inferencing due to a lack of specificity in the instructions.  \n" + \
        "Background: This collaborative game is played in a grid containing the following objects: keys, doors, gems, walls, empty spaces, a human, and you. The colors of keys and doors can be red, blue, or yellow.\n" 

        fig1 = self.get_grid_in_str("3.txt", 1)
        
        human_obj = "\nHuman's objective: The human's goal is to retrieve a specific gem by providing only a single instruction. However, the human's instruction could be ambiguous or lack specificity, adding a challenge for you to correctly infer the gem desired by the human. For instance, consider the configuration in Figure 1. While some gems, like the gem at (9, 11), may be easily accessible (i.e., don't require a key), the human may instead be aiming for a more challenging gem, such as the red gem placed behind two doors, and can provide instruction as \"Get the red key.\" Here, the human is directing you to collect a red key to unlock the red door and get the gem. But the gem is blocked by two red doors. Thus, you must collect and pass two red keys to the human. You need to infer this even when not explicitly stated based on the context (i.e., the movement of the human).\n"
        
        ai_obj = "Your objective: Your primary goal is to assist the human in retrieving the desired gem by fetching the necessary keys or unlocking doors while prioritizing minimal movement. Upon receiving the instruction, you infer which gem the human intends to collect and take only the essential actions—either collecting and passing keys or unlocking doors—to enable the human to access the gem as efficiently and quickly as possible. As a result, your movements are optimized to ensure that the number of steps taken is minimal.\n"
        
                
        key_points = "\nRules of the Game:\n" + \
                "- For each grid configuration (problem), the human provides a single instruction called a callout.\n" + \
                "- The instructions (callouts) can be of two types: (1) either directing you to collect a key (or keys) and passing them to the human or (2) instructing you to unlock a door (or doors).\n" + \
                "- Only the key of the same color as the door can unlock the door. However, once a key is used to unlock a door, it cannot be reused for any other door of the same color. For instance, one red key can only unlock one red door.\n" + \
                "- If you collect one or more keys, they are immediately passed to the human, depending on the grid configuration. However, if you are blocked by a door (or doors) and cannot pass the key directly, you will either collect the necessary key(s) to unlock the door or instruct the human to retrieve the key if it is beyond your reach. This ensures you can overcome obstacles and continue assisting the human efficiently.\n" + \
                "- If the callout instructs to unlock a door or doors, you must first collect the corresponding key or keys before proceeding to unlock the door (or doors). In these instances, you do not pass the key to the human.\n" + \
                "- Gems are randomly placed within the grid.\n" + \
                "- You prioritize efficient, optimal, and obstacle-free routes to achieve the human's objective as quickly and effectively as possible. When instructed to collect key(s) or unlock the door(s), you don't gather all the keys or unlock all the doors on the grid. Instead, you perform only the actions necessary to assist the human in retrieving the gem with minimal steps.\n" + \
                "- The objects, such as gems, keys, and doors, have no associated cost (i.e., they are all the same). Your primary goal is to acquire the necessary keys or unlock the appropriate doors, ensuring the human can obtain the desired gem most efficiently, i.e., requiring minimal movements to execute the instruction.\n" + \
                "- The human's instruction may not always be explicit, requiring you to infer the intended action based on the context and the grid configuration. The instructions may be ambiguous or lacking specificity.\n"
                        
        grid_context = "\nThe Current Grid Layout for the Keys, Doors, and, Gems: \n" + str(grid) + "\n\nBelow are the locations of all objects on the grid, displayed as (row, column) coordinates, along with their total count:\n" + str(object_pos) + "\n\n"
    
        pred = "Based on your understanding of the game rules, current grid layout, human actions, instruction and the labeled examples provided below, generate the instruction type (clear or unclear), response, and optimal actions for the following human actions and instruction, delimited by triple backticks:\n" 
      
        # Use self.cleaning to clean the instruction
        test_sample = "\nHuman Action: " + self.clean_text(str(human_move)) + \
                      "\nInstruction: " + self.clean_text(str(instruction)) + \
                      "\nType: <Identify the instruction type and provide chain-of-thought reasoning for the type based on the current grid configuration>" + \
                      "\nResponse: <Generate chain-of-thought reasoning for the response based on the current grid configuration>" + \
                      "\nActions: <Provide the optimal actions from the response based on the instruction and the current grid configuration>\n"


        text = '```' + test_sample + '```\n'
       
        few_shot = "Use the following problems and examples, delimited by triple quotes, to understand how to generate the appropriate type, response, and actions for each instruction and human actions.\n"
        if exp:
            examples = self.get_ToM_examples()
        else:
            examples = self.get_ToM_FS()
            key_points +="- Always generate your response by cross-checking and verifying it against the current grid configuration to ensure accuracy.\n" + \
                         "- When traversing the grid, minimize and optimize your path, avoiding collisions with walls (W) or other obstacles.\n" + \
                         "- Infer the human's desired gem based on their movement, instruction, and grid configuration. Despite unclear instructions, generate a response that captures the human's intent and ensures successful retrieval of their desired gem.\n" + \
                         "- Apply theory of mind principles to infer the human's desired gem based on their actions and instructions.\n"
        
        few_shot +="'''\n" + examples + "'''\n"
        
       
        prompt = intro + "\n" + fig1 + "\n" + human_obj + "\n" + ai_obj + key_points + grid_context + pred + text + few_shot
        
        #print(prompt)

        return prompt
    
    def get_ToM_FS(self):
        return "Human Action: The human moves downward from their current position at (6,4) to (8,4), where they provide the instruction.\n" + \
                "Instruction: Can you pass me the blue key?\n" + \
                "Type: Unclear. Based on the human's movement, instruction and the current grid configuration, I infer that the human moved to (8,4) to collect the gem at (8,9) located beyond the blue door and red door at (8,6) and (8,8) respectively since they asked for a blue key. To retrieve the gem the human needs to unlock both these doors. Since the human does not mention passing the red key, the instruction is unclear.\n" + \
                "Response: The instruction asks to collect a blue key and based on the human movements and the grid configuration the desired gem they want to collect is placed at (8,9). But this gem is beyond the blue door and red door at (8,6) and (8,8) respectively. So I will collect one blue key and one red key located at (3,8) and (5,8). To minimize movement, I will first pick up the blue key at (3,8) followed by the red key at (5,8), as this strategy reduces the number of steps from my current position at (1,8) to the human at (8,4). Once both keys are collected, I will pass them to the human, allowing them to unlock blue door and red door at (8,6) and (8,8) respectively and retrieve the gem at (8,9).\n" + \
                "Actions:\n1) Collect: blue_key at (3,8).\n2) Collect: red_key at (5,8).\n3) Pass: blue_key and red_key to the human at (8,4).\n4) Unlock: human unlocks the Blue_door at (8,6) and the Red_door at (8,8).\n5) Retrieve: human retrieves gem at (8,9).\n\n" + \
                "Human Action: The human moves left from their current position at (0,5) to (0,3), where they provide the instruction before continuing their movement.\n" + \
                "Instruction: On my way to pick up the blue key, can you find a yellow key?\n" + \
                "Type: Clear. Based on the human's movement, instruction and the current grid configuration, I infer that the human moved to (0,3) to pick up the blue key at (1,0) and requested the yellow key to retrieve the gem at (2,10) located beyond both the blue and yellow doors at (4,7) and (3,10) respectively. Since both the keys are required to retrieve the gem at (2,10), the instruction is clear.\n" + \
                "Response: The instruction asks to collect a yellow key while the human is picking up a blue key, so the desired gem the human wants to collect is placed beyond a blue and a yellow door. To get the gem at (2,10) the human needs to go through a blue door at (4,7) and a yellow door at (3,10), thus the gem at (2,10) must be the desired gem. Since the human is picking up the blue key at (1,0) based on their movement towards the left direction, I will collect the yellow key. The human after collecting the blue key will move to the blue door to unlock the blue door at (4,7), thus their future position can be (4,10) as they will keep moving towards the gem at (2,10). To minimize movement, I will pick up the yellow key at (8,10), as this strategy reduces the number of steps from my current position at (10,9). Once the key is collected, I will pass it to the human at (4,10) or their future position, after they unlock the blue door at (4,7). The human can then unlock the yellow door at (3,10) to retrieve the gem at (2,10).\n" + \
                "Actions:\n1) Collect: yellow_key at (8,10).\n2) Unlock: human unlocks Blue_door at (4,7).\n3) Pass: yellow_key to the human at (4,10) or their future position.\n4) Unlock: human unlocks the Yellow_door at (3,10).\n5) Retrieve: human retrieves gem at (2,10).\n\n" + \
                "Human Action: The human moves right from their current position at (3,3) to (3,5), where they provide the instruction before continuing their movement.\n" + \
                "Instruction: I'll get the blue key. Can you pick up a red key?\n" + \
                "Type: Clear. Based on the human's movement, instruction and the current grid configuration, I infer that the human moved to (3,5) to pick up the blue key at (3,8) and requested the red key to collect either the gem at (11,8) or the gem at (11,12) located behind the red and blue doors at (9,10) and (10,10) respectively. Since the red and blue keys are required for the human to reach either one of their desired gems, the instruction is clear.\n" + \
                "Response: The instruction asks to collect a red key while the human is picking up a blue key, so the desired gem the human wants to collect is placed beyond a blue and a red door. To get the gem at either (11,8) or (11,12) the human needs to go through a red door at (9,10) and a blue door at (10,10), thus the gems at either (11,8) or (11,12) must be one of the desired gems. Since the human is picking up the blue key at (3,8) based on their movement towards the right direction, I will collect the red key. The human after collecting the blue key at (3,8) will move towards the red and blue doors at (9,10) and (10,10), thus their future position can be at (8,10), located just before both the doors. To minimize movement, I will pick up the red key located at (8,6), as this strategy reduces the number of steps from my current position at (6,3). Once the key is collected, I will pass it to the human at (8,10) or their future position, allowing them to unlock the both the doors and retrieve either one of the gems.\n" + \
                "Actions:\n1) Collect: red_key at (8,6).\n2) Pass: red_key to the human at (8,10) or their future position.\n3) Unlock: human unlocks Red_door at (9,9) and Blue_door at (10,10).\n4) Retrieve: human retrieves a gem at either (11,8) or (11,12).\n\n" + \
                "Human Action: The human moves right from their current position at (5,3) to (5,6) and then moves downward to (6,6), where they provide the instruction before continuing their movement.\n" + \
                "Instruction: Can you go get the red key?\n" + \
                "Type: Unclear. Based on the human's movement, instruction and the current grid configuration, I infer that the human moved to (6,6) to pick up the blue key at (8,6) even though it was not mentioned in the instruction, and requested the red key to collect either the gem at (8,0) or (8,4) located behind the red and blue doors at (6,3) and (7,3) respectively. Since the human does not mention anything about a blue key, the instruction is unclear.\n" + \
                "Response: The instruction asks to collect a red key while the human is likely picking up a blue key, so the desired gem the human wants to collect is placed beyond a blue and a red door. To get the gem at either (8,0) or (8,4) the human needs to go through a red door at (6,3) and a blue door at (7,3), thus the gem at either (8,0) or (8,4) must be the desired gem. Since the human is likely picking up the blue key at (8,6) based on their movement towards the downward direction, I will collect the red key. The human after collecting the blue key at (8,6) will move towards the red and blue doors at (6,3) and (7,3), thus their future position can be at (5,3), located just before both the doors. There are two red keys on the grid at (0,7) and at (3,7). Since the red key at (3,7) is blocked by a blue door, I will collect the red key at (0,7) such that it takes minimal steps. Once the red key is collected, I will pass it to the human at (5,3) or their future position, allowing them to unlock the doors and collect either one of the gems.\n" + \
                "Actions:\n1) Collect: red_key at (0,7).\n2) Pass: red_key to the human at (5,3) or their future position.\n3) Unlock: human unlocks Red_door at (6,3) and Blue_door at (7,3).\n4) Retrieve: human retrieves a gem at either (8,0) or (8,4).\n\n" + \
                "Human Action: The human moves upward from their current position at (3,5) to (0,5), where they provide the instruction before continuing their movement.\n" + \
                "Instruction: I will pick up this red key. Can you find a yellow one?\n" + \
                "Type: Unclear. Based on the human's movement, instruction and the current grid configuration, I infer that the human moved to (0,5) to pick up the red key at (0,0) and requested for a yellow key to collect either one of the gems at (7,2), (7,4), or (7,6), respectively. Because to reach these gems it requires to unlock a yellow door first and then a red door. Since there are many possibilities to reach gems which will require both yellow and red keys, it is impossible to infer which gem the human wants, making this an unclear instruction.\n" + \
                "Response: The instruction asks for a yellow key while the human is picking up a red key, so the desired gem the human wants to collect is placed beyond a yellow and a red door. To get the gems at (7,2), (7,4), or (7,6), the human needs to go through a yellow and a red door at. Therefore, based on the human's future position their desired gem can be any one of them. A red door at (6,2) is placed after a yellow door at (5,2). Two red doors at (6,4) and (6,6) are placed after a yellow door at (4,5). Therefore, the human's future position will be before the two yellow doors at (4,2) or (3,5), or it can be someplace else. To minimize movement, I will pick up the yellow key at (9,0) to reduce the number of steps from my current position at (9,5) to the human at (3,5) or (4,2) or their future position. Once I collect the yellow key, I will pass it to the human at (3,5) or (4,2), allowing them to unlock the doors and receive the gem.\n" + \
                "Actions:\n1) Collect: yellow_key at (9,0).\n2) Pass: yellow_key to the human at (3,5) or (4,2) or their future position.\n3) Unlock: human unlocks the Yellow_door at (5,2) or (4,5) and the Red_door at (6,2) or (6,4), (6,6).\n4) Retrieve: human retrieves the gem at either (7,2), (7,4) or (7,6).\n\n" + \
                "Human Action: The human moves down from their current position at (4,4) to (5,4), and then proceeds to move left to collect the red key at (5,0). After collecting the key, they continue moving right until they reach (5,2), where they provide the instruction before continuing their movement.\n" + \
                "Instruction: Can you help me unlock the blue door there?\n" + \
                "Type: Clear. Based on the human's movement, instruction and the current grid configuration, I infer that the human picked up the red key at (5,0), moved to (5,2), and requested me to unlock blue door to retrieve the gem at (0,0), which is located beyond the blue and red doors at (2,4) and (2,0) respectively. To retrieve the gem the human needs to pass both these doors. Since the human picked up the red key and I just have to collect the blue key to unlock the blue door blocking the human's path to the gem, the instruction is clear.\n" + \
                "Response: The instruction asks to unlock the blue door while the human has already picked up the red key at (5,0). The human's desired gem based on their actions and the grid configuration must be beyond a blue and a red door. The gem at (0,0) is the desired gem as it is placed beyond a blue door at (2,4) and a red door at (2,0). To minimize movement, I will collect the only accessible blue key at (0,8) and then move to unlock the blue door at (2,4) which is blocking the path to the gem at (0,0). The human can then unlock the red door at (2,0) after the blue door at (2,4) is unlocked to retrieve the gem at (0,0).\n" + \
                "Actions:\n1) Collect: blue_key at (0,8).\n2) Unlock: Blue_door at (2,4).\n3) Unlock: human unlocks Red_door at (2,0).\n4) Retrieve: human retrieves gem at (0,0).\n\n" + \
                "Human Action: The human moves left from their current position at (6,2) to (6,1) and then upward to (5,1). Upon reaching (5,1), adjacent to the red door at (4,1), they provide the instruction.\n" + \
                "Instruction: Can you unlock this red door?\n" + \
                "Type: Unclear. Based on the human's movement, instruction and the current grid configuration, I infer that the human moved to (5,1) and asked to unlock the red door at (4,1) to retrieve the gem at (1,1) located behind a blue and red door at (2,1) and (4,1) respectively. To retrieve the gem the human needs to have both doors unlocked. However, since there is no mention about unlocking the blue door, the instruction is unclear.\n" + \
                "Response: The instruction specifies unlocking a red door, indicating that the desired gem must be behind one. Based on the grid configuration, there are two gems on the grid, both blocked by a red and a blue door. Given that the human has moved to (5,1), adjacent to the red door at (4,1), they are likely targeting the gem at (1,1), which lies beyond the adjacent red door. However, the gem at (1,1) is also blocked by a blue door at (2,1), in addition to the red door at (4,1). To assist the human, I will collect both the red and blue keys. To minimize movement, I will first collect the red key at (0,5), followed by the blue key at (2,7), as this approach reduces the total steps from my current position at (2,5). After collecting both keys, I will proceed to unlock the blue door at (2,1) and then the red door at (4,1), enabling the human to access the gem at (1,1).\n" + \
                "Actions:\n1) Collect: red_key at (0,5).\n2) Collect: blue_key at (2,7).\n3) Unlock: Blue_door at (2,1) and Red_door at (4,1).\n4) Retrieve: human retrieves gem at (1,1).\n\n"
                
    def get_ToM_examples(self):
        e11 = "e1_1.txt"
        e12 = "e1_2.txt"
        e13 = "e1_3.txt"
        e11_str = self.get_grid_in_str(e11, 2)
        e11_context = e11_str + "\n\n--Initial Grid: This grid in Figure 2 displays the initial positions of you, human, keys (red, yellow), doors (red, yellow), and gems.\n"
        e12_str = self.get_grid_in_str(e12, 3)
        e12_context = e12_str + "\n\n--Observed Grid: The human moves left toward the red door in Figure 3 from position (3,6) to (3,2) and gives the instruction, \"Can you pass me the red key?\" From this, you infer the human's intentions based on their movement and location—specifically, which gem they are aiming to collect.\n"
        e13_str = self.get_grid_in_str(e13, 4)
        e13_context = e13_str + "\n\n--Completed Grid: You infer that the human intends to collect the gem beyond the red door. However, you notice that after the red door, there is also a yellow door blocking the path to the gem. In response, in Figure 4, you collect both the red and yellow keys and passes them to the human. The human then uses the corresponding keys to unlock both doors and successfully retrieves the gem.\n"
        e1 = "\nExample from this grid configuration.\n" + \
            "Human Action: The human moves left from their current position at (3,6) to (3,2), which is adjacent to the red door at (3,1). Upon arriving at (3,2), they provide the instruction.\n" + \
            "Instruction: Can you pass me the red key?\n" + \
            "Type: Unclear. Based on the instruction and the grid, I infer that the human wants to collect the gem at (7,0), beyond a red door. However, before reaching the gem, a yellow door is blocking the path. Therefore, I must collect both the red and yellow keys, and since the instruction doesn't mention picking up the yellow key, this instruction lacks specificity and is thus unclear.\n" + \
            "Response: There are two yellow doors, one red door, one red key, and one yellow key on the grid. The instruction asks to collect a red key, so the desired gem the human wants to collect is placed beyond the red door at (3,1). However, a yellow door at (6,0) blocks the path to the gem at (7,0). Thus, I will collect both the red and yellow keys. To minimize movement, I will first pick up the red key at (0,0) followed by the yellow key at (1,0), as this strategy reduces the number of steps from my current position at (0,4). I will then pass them to the human at (3,2) to unlock the doors and collect the gem.\n" + \
            "Actions:\n1) Collect: red_key at (0,0).\n2) Collect: yellow_key at (1,0).\n3) Pass: red_key and yellow_key to the human at (3,2).\n4) Unlock: human unlocks the Red_door at (3,1) and the Yellow_door at (6,0).\n5) Retrieve: human retrieves gem at (7,0).\n"
            
        e21 = "e2_1.txt"
        e22 = "e2_2.txt"
        e23 = "e2_3.txt"
        e21_str = self.get_grid_in_str(e21, 5)
        e21_context = e21_str + "\n\n--Initial Grid: This grid in Figure 5 displays the initial positions of you, human, keys (red, blue), doors (red, blue), and gems.\n"
        e22_str = self.get_grid_in_str(e22, 6)
        e22_context = e22_str + "\n\n--Observed Grid: The human moves to the right from their current position in Figure 6 from (5,0) to (5,4), and provides an instruction, \"Can you pass me the red keys?\" From this, you infer the human's intentions based on their movement and location—specifically, which gem they are aiming to collect.\n"
        e23_str = self.get_grid_in_str(e23, 7)
        e23_context = e23_str + "\n\n--Completed Grid: You infer that the human intends to retrieve the gem located beyond the two red doors on the grid. Observing that there are three red keys available, you strategically choose to collect the two keys positioned on the upper left of the grid, as they require the fewest steps from your current location. After gathering the keys, you pass them to the human, who then uses them to unlock both doors and successfully retrieve the gem.\n"
        e2 = "\nExample from this grid configuration.\n" + \
             "Human Action: The human moves to the right from their current position at (5,0) to (5,4), where they provide the instruction before continuing their movement.\n" + \
             "Instruction: Can you pass me the red keys?\n" + \
             "Type: Clear. Based on the instruction and the grid, I infer that the human wants to collect the gem at (10,0) or at (10,8), which is beyond two red doors. Since the human needs two red keys to unlock the doors and reach the gem, the instruction is clear.\n" + \
             "Response: There are three red keys, three blue keys, two red doors, and two blue doors on the grid. The instruction asks to collect red keys to retrieve the gem at (10,0) or (10,8). I will collect the two closest red keys at (1,1) and (1,3), as this minimizes the steps required to retrieve them. Since the human continues moving after the instruction, they are likely heading downward from (5,4) to (7,4), adjacent to the red door at (8,4). I will pass the keys to the human at either (5,4) or (7,4), depending on their movements, enabling them to unlock the red doors at (8,4) and (9,4) to access the desired gem.\n" + \
             "Actions:\n1) Collect: red_key at (1,1).\n2) Collect: red_key at (1,3).\n3) Pass: red_keys to the human's future position either at (5,4) or (7,4).\n4) Unlock: human unlocks the Red_doors at (8,4) or (9,4).\n5) Retrieve: human retrieves the gem at either (10,0) or (10,8).\n"
        com_str = e11_context + "\n" + e12_context + "\n" + e13_context + e1 + "\n" + e21_context + "\n" + e22_context + "\n" + e23_context + e2
        return com_str
           
    def get_grid_in_str(self, name, number):
        env = GridEnvironment()
        grid_file_path = env.get_grid_file_path(name)
        if grid_file_path:
            env.load_grid_from_file(grid_file_path)
            object_pos = env.get_all_object_positions()
            grid_str = str(env.grid)
            if number == 1:
                combined_str = f"Figure {number}: Grid Configuration\n{grid_str}\n"
            elif number == 2:
                combined_str = f"---Problem (1) demonstrating initial, observed, completed grids with figures and the example.\nFigure {number}: Initial Grid Configuration\n{grid_str}\n\nObject Positions for Figure {number}:\n{object_pos}"
            elif number == 5:
                combined_str = f"---Problem (2) demonstrating initial, observed, completed grids with figures and the example.\nFigure {number}: Initial Grid Configuration\n{grid_str}\n\nObject Positions for Figure {number}:\n{object_pos}"
            elif number == 3 or number == 6:
                combined_str = f"Figure {number}: Observed Grid Configuration\n{grid_str}\n\nObject Positions for Figure {number}:\n{object_pos}"
            else:
                combined_str = f"Figure {number}: Completed Grid Configuration\n{grid_str}\n\nObject Positions for Figure {number}:\n{object_pos}"
            return combined_str
        else:
            print("Grid file not found.")
            return None
    
  
    def clean_text(self, text):
        '''
        Remove new liens / "" in the students response
        remove extra space
        '''
        # print("Before cleaning: ", text)
        text = text.strip()
        text = text.strip('\"')
        text = text.strip('\n')

        return text


    def is_asking_for_clarification(self, response):
        # Check for common phrases that indicate clarification queries
        clarification_phrases = ["Do you mean", "Which", "Could you specify", "I'm not sure"]
        return any(phrase in response for phrase in clarification_phrases)

        
    def infer_action_from_history(self): #Have to edit this
        # Infer the action based on conversation history if clarification limit is reached
        for entry in reversed(self.conversation_history):
            if "Response" in entry and "action" in entry.lower():
                return entry.split("Response: ")[1]
        return "Perform best-guess action based on environment."
    
        

if __name__ == "__main__":
    # Example grid setup
    grid = np.array([
        ['r', '.', '.', '.', 'm', 'W', 'W', 'g'],
        ['y', '.', 'W', 'W', '.', 'W', 'W', '.'],
        ['W', 'W', 'W', 'W', '.', 'W', 'W', '.'],
        ['.', 'R', '.', '.', '.', '.', 'h', '.'],
        ['.', 'W', '.', 'W', 'W', 'W', 'W', '.'],
        ['.', 'W', '.', 'W', 'W', 'W', 'W', 'Y'],
        ['Y', 'W', '.', 'W', 'W', 'W', 'W', '.'],
        ['g', 'W', 'g', 'W', 'W', 'W', 'W', 'g']
    ])

    agent_start_pos = (0, 4)  # Starting position of the agent (m)
    human_pos = (3, 6)  # Position of the human (h)

    # Initialize the agent
    agent = Agent(agent_start_pos)

    # Define the instruction
    instruction = "Can you pass me the blue key?."
    human_move = "The human moves to the right from their current position at (5,0) until they reach at (5,4) and gives the instruction."
    

    env = GridEnvironment()
    name = "3.txt"
    # str_grid = agent.get_ToM_examples()
    # print(str_grid)     
    grid_file_path = env.get_grid_file_path(name)
    env.load_grid_from_file(grid_file_path)

    pos = env.get_all_object_positions()
    # Get agent and object positions
    agent_pos = env.get_agent_position()
    human_pos = env.get_human_position()
    exp = False
    object_pos = env.get_all_object_positions()

    #print(pos)

    # Generate the prompt
    # print("********")
    prompt = agent.create_prompt_ToM(instruction, env.grid, human_move, object_pos, exp)


    # Print the generated prompt
    print(prompt)

