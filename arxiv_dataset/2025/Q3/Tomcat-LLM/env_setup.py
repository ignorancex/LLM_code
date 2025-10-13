import numpy as np
import os

class GridEnvironment:
    def __init__(self):
        self.grid = None
        self.doors = {}  # Store positions and door colors
        self.keys = {}   # Store positions and key colors
        self.gems = {}
        self.agents = {}
        self.humans = {}
        self.walls = []
        self.empty_positions = []
        self.all_objects = {}  # General data structure to hold all objects and their positions

    def load_grid_from_file(self, file_path):
        """Load the grid from the given file path (txt file) and set it up."""
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Initialize the grid as a numpy array
        self.grid = np.array([list(line.strip()) for line in lines])

        # Scan through the grid to identify doors, keys, gems, walls, empty positions, agents, and humans
        for i, row in enumerate(self.grid):
            for j, cell in enumerate(row):
                if cell == 'R':  # Red door
                    self.doors[(i, j)] = 'red'
                    self.all_objects[(i, j)] = 'door-red'
                elif cell == 'Y':  # Yellow door
                    self.doors[(i, j)] = 'yellow'
                    self.all_objects[(i, j)] = 'door-yellow'
                elif cell == 'B':  # Blue door
                    self.doors[(i, j)] = 'blue'
                    self.all_objects[(i, j)] = 'door-blue'
                elif cell == 'r':  # Red key
                    self.keys[(i, j)] = 'red'
                    self.all_objects[(i, j)] = 'key-red'
                elif cell == 'y':  # Yellow key
                    self.keys[(i, j)] = 'yellow'
                    self.all_objects[(i, j)] = 'key-yellow'
                elif cell == 'b':  # Blue key
                    self.keys[(i, j)] = 'blue'
                    self.all_objects[(i, j)] = 'key-blue'
                elif cell == 'g':  # Gems
                    self.gems[(i, j)] = 'gem'
                    self.all_objects[(i, j)] = 'gem'
                elif cell == 'm':  # Agent (robot)
                    self.agents[(i, j)] = 'agent'
                    self.all_objects[(i, j)] = 'agent'
                elif cell == 'h':  # Human
                    self.humans[(i, j)] = 'human'
                    self.all_objects[(i, j)] = 'human'
                elif cell == 'W':  # Wall
                    self.walls.append((i, j))
                    self.all_objects[(i, j)] = 'wall'
                elif cell == '.':  # Empty space
                    self.empty_positions.append((i, j))
                    self.all_objects[(i, j)] = 'empty'

    def get_agent_position(self):
        """Find and return the current position of the agent (m)."""
        for position, obj in self.all_objects.items():
            if obj == 'agent':
                return position
        return None  # If agent is not found

    def get_human_position(self):
        """Find and return the current position of the human (h)."""
        for position, obj in self.all_objects.items():
            if obj == 'human':
                return position
        return None  # If human is not found

    def display_grid(self):
        """Print the grid in a human-readable format."""
        for row in self.grid:
            print(' '.join(row))

    def get_all_object_positions(self):
        """Display the position of all objects on the grid in a grouped, human-readable format."""
        result = []
        object_positions = {
            'agent': [],
            'human': [],
            'key-red': [],
            'key-yellow': [],
            'key-blue': [],
            'door-red': [],
            'door-yellow': [],
            'door-blue': [],
            'gem': [],
            'wall': [],
            'empty': []
        }

        # Populate the dictionary with positions of each object
        for position, obj in self.all_objects.items():
            object_positions[obj].append(position)

        # Helper function to format positions without square brackets
        def format_positions(positions):
            return ', '.join([f"({x}, {y})" for x, y in positions])

        # Store information for each object in the result string
        if object_positions['agent']:
            result.append(f"My position (Labeled as 'm'): {format_positions(object_positions['agent'])}")
        if object_positions['human']:
            result.append(f"Human (Labeled as 'h'): {format_positions(object_positions['human'])}")
        
        if object_positions['key-red']:
            if len(object_positions['key-red']) > 1:
                result.append(f"Red keys (Labeled as 'r'): {format_positions(object_positions['key-red'])} --> Total Red keys: {len(object_positions['key-red'])}")
            else:
                result.append(f"Red key (Labeled as 'r'): {format_positions(object_positions['key-red'])} --> Total Red key: {len(object_positions['key-red'])}")
        
        if object_positions['key-yellow']:
            if len(object_positions['key-yellow']) > 1:
                result.append(f"Yellow keys (Labeled as 'y'): {format_positions(object_positions['key-yellow'])} --> Total Yellow keys: {len(object_positions['key-yellow'])}")
            else:
                result.append(f"Yellow key (Labeled as 'y'): {format_positions(object_positions['key-yellow'])} --> Total Yellow key: {len(object_positions['key-yellow'])}")
        
        if object_positions['key-blue']:
            if len(object_positions['key-blue']) > 1:
                result.append(f"Blue keys (Labeled as 'b'): {format_positions(object_positions['key-blue'])} --> Total Blue keys: {len(object_positions['key-blue'])}")
            else:
                result.append(f"Blue key (Labeled as 'b'): {format_positions(object_positions['key-blue'])} --> Total Blue key: {len(object_positions['key-blue'])}")
        
        if object_positions['door-red']:
            if len(object_positions['door-red']) > 1:
                result.append(f"Red doors (Labeled as 'R'): {format_positions(object_positions['door-red'])} --> Total Red doors: {len(object_positions['door-red'])}")
            else:
                result.append(f"Red door (Labeled as 'R'): {format_positions(object_positions['door-red'])} --> Total Red door: {len(object_positions['door-red'])}")
        
        if object_positions['door-yellow']:
            if len(object_positions['door-yellow']) > 1:
                result.append(f"Yellow doors (Labeled as 'Y'): {format_positions(object_positions['door-yellow'])} --> Total Yellow doors: {len(object_positions['door-yellow'])}")
            else:
                result.append(f"Yellow door (Labeled as 'Y'): {format_positions(object_positions['door-yellow'])} --> Total Yellow door: {len(object_positions['door-yellow'])}")
        
        if object_positions['door-blue']:
            if len(object_positions['door-blue']) > 1:
                result.append(f"Blue doors (Labeled as 'B'): {format_positions(object_positions['door-blue'])} --> Total Blue doors: {len(object_positions['door-blue'])}")
            else:
                result.append(f"Blue door (Labeled as 'B'): {format_positions(object_positions['door-blue'])} --> Total Blue door: {len(object_positions['door-blue'])}")
        
        if object_positions['gem']:
            result.append(f"Gems (Labeled as 'g'): {format_positions(object_positions['gem'])} --> Total Gems: {len(object_positions['gem'])}")
        
        if object_positions['wall']:
            result.append(f"Walls (Labeled as 'W'): {format_positions(object_positions['wall'])} --> Total Walls: {len(object_positions['wall'])}")
        
        if object_positions['empty']:
            result.append(f"Empty spaces (Labeled as '.'): {format_positions(object_positions['empty'])} --> Total Empty spaces: {len(object_positions['empty'])}")

        # Join all the result strings into a single formatted string
        result_str = "\n".join(result)
        
        # Print the result
        #print(result_str)

        # Optionally return the result string if you want to store or process it further
        return result_str



    # Function to get the path to the .txt file inside the dataset/problem folder
    def get_grid_file_path(self, txt):
        dataset_dir = 'dataset/problems'
        for file_name in os.listdir(dataset_dir):
            if file_name == txt:
                return os.path.join(dataset_dir, file_name)
        return None

# Example usage
if __name__ == "__main__":
    env = GridEnvironment()

    # Dynamically get the path to the .txt file in the dataset/problem folder
    txt = "1.txt"
    grid_file_path = env.get_grid_file_path(txt)
    if grid_file_path:
        env.load_grid_from_file(grid_file_path)
        env.display_grid()
        object_pos = env.get_all_object_positions()  
        # Display positions of all objects
        print(object_pos)
    else:
        print("No grid file found in the dataset/problem folder.")
