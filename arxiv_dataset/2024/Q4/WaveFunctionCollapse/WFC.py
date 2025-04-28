import os
from collections import defaultdict
from enum import Enum
from copy import deepcopy
import random

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from functions import slopes_to_height, augment_images
from utils import default_dict_of_sets, save_state, load_state

class observe_mode(Enum):
    MB = 0
    RANDOM = 1
    
class NoWFCSolution(Exception):
    def __init__(self, message="No WFC solutions Found!"):
        super().__init__(message)
        
        
class WaveFunctionCollapse:
    def __init__(self, input_images: list, pattern_size, wrap_input=False, random_seed=0, nan_value=-32607.0, remove_low_freq=False, low_freq=1, augment_patterns=False, max_runs=25):
        self.random_seed = random_seed
        self.n_images = len(input_images)
        self.input_images = np.dstack(input_images)
        if len(np.array(input_images[0]).shape) == 2:
            self.channels_per_image = 1
        else:
            self.channels_per_image = np.array(input_images[0]).shape[2]

        self.pattern_size = pattern_size
        # self.patterns = []
        self.pattern_frequencies = defaultdict(int)
        self.adjacency_rules = defaultdict(default_dict_of_sets)
        
        self.empty_pattern = tuple(np.full(pattern_size, -32767).flatten())
        self.output_grid = None
        self.possible_patterns = None
        self.observations = []
        self.wrap = wrap_input
        
        if nan_value is None:
            self.nan_value = np.min(input_image) - 1
        else:
            self.nan_value = nan_value
            
        self.remove_low_freq = remove_low_freq
        self.low_freq =low_freq
        self.augment_patterns = augment_patterns
        self.n_runs = 0
        self.n_runs_on_block = 0

        self.max_runs = max_runs


    def generate_pattern_mappings(self):
        # Generate a dictionary mapping each pattern to a unique number
        pattern_to_number = {pattern: idx for idx, pattern in enumerate(self.patterns)}
        
        # Generate the reverse mapping from number to pattern
        number_to_pattern = {idx: pattern for pattern, idx in pattern_to_number.items()}
        
        return pattern_to_number, number_to_pattern
    
    def add_augmentations(self, pattern: np.ndarray):
        """Adds rotations and mirrors of patterns to pattern_frequencies"""
        
        # rotated90 = np.rot90(pattern)
        # rotated180 = np.rot90(rotated90)
        # rotated270 = np.rot90(rotated180)
        # mirror_h = pattern[::-1, ...]
        # mirror_v = pattern[:, ::-1, ...]
        # for pat in [rotated90, rotated180, rotated270, mirror_h, mirror_v]:
        for pat in [-pattern]:
            self.pattern_frequencies[tuple(pat.flatten())] += 1
        
    def extract_patterns(self):
        """Extract patterns and compute their frequencies and adjacency rules."""
        
        for i in range(self.n_images):
            _image = self.input_images[..., self.channels_per_image*i:self.channels_per_image*i+self.channels_per_image]
            height, width = _image.shape[0:2]
            for y in tqdm(range(height), desc="Extracting Patterns"):
                for x in range(width):
                    pattern = self.get_pattern(x, y, height, width, _image)
                    # self.patterns.append(pattern)
                    if self.nan_value not in pattern:
                        self.pattern_frequencies[tuple(pattern.flatten())] += 1
                        if self.augment_patterns:
                            self.add_augmentations(pattern)

        if self.remove_low_freq:
            self.patterns = [k for k in self.pattern_frequencies.keys() if self.pattern_frequencies[k] > self.low_freq] 
        else:
            self.patterns = list(self.pattern_frequencies.keys())
        self.pattern_to_number, self.number_to_pattern = self.generate_pattern_mappings()
        

        # # Set up adjacency rules
        # for y in range(height):
        #     for x in range(width):
        #         pattern = self.get_pattern(x, y)
        #         above_pattern = self.get_pattern(x, (y - 1) % height)
        #         below_pattern = self.get_pattern(x, (y + 1) % height)
        #         left_pattern = self.get_pattern((x - 1) % width, y)
        #         right_pattern = self.get_pattern((x + 1) % width, y)

        #         self.adjacency_rules[tuple(pattern.flatten())]['above'].add(tuple(above_pattern.flatten()))
        #         self.adjacency_rules[tuple(pattern.flatten())]['below'].add(tuple(below_pattern.flatten()))
        #         self.adjacency_rules[tuple(pattern.flatten())]['left'].add(tuple(left_pattern.flatten()))
        #         self.adjacency_rules[tuple(pattern.flatten())]['right'].add(tuple(right_pattern.flatten()))

    def match_patterns(self):
    # Set up adjacency rules based on compatibility
        for i in tqdm(range(len(self.patterns)), "Matching Patterns"):
            for j in range(i, len(self.patterns)):
                pattern1 = self.patterns[i]
                pattern2 = self.patterns[j]
                
                pattern1_array = np.array(pattern1).reshape(*self.pattern_size)
                pattern2_array = np.array(pattern2).reshape(*self.pattern_size)
                # Check if pattern1 can be above pattern2
                if np.array_equal(
                    pattern1_array[-1, ...], 
                    pattern2_array[0, ...]
                ):
                    self.adjacency_rules[pattern1]['below'].add(pattern2)
                    self.adjacency_rules[pattern2]['above'].add(pattern1)

                # Check if pattern1 can be below pattern2
                if np.array_equal(
                    pattern1_array[0, ...], 
                    pattern2_array[-1, ...]
                ):
                    self.adjacency_rules[pattern1]['above'].add(pattern2)
                    self.adjacency_rules[pattern2]['below'].add(pattern1)

                # Check if pattern1 can be to the left of pattern2
                if np.array_equal(
                    pattern1_array[:, -1, ...], 
                    pattern2_array[:, 0, ...]
                ):
                    self.adjacency_rules[pattern1]['right'].add(pattern2)
                    self.adjacency_rules[pattern2]['left'].add(pattern1)

                # Check if pattern1 can be to the right of pattern2
                if np.array_equal(
                    pattern1_array[:, 0, ...], 
                    pattern2_array[:, -1, ...]
                ):
                    self.adjacency_rules[pattern1]['left'].add(pattern2)
                    self.adjacency_rules[pattern2]['right'].add(pattern1)


    def get_pattern(self, x, y, height, width, image):
        """Get a pattern from the input image, considering wrapping."""
        pattern = np.full(self.pattern_size, fill_value=self.nan_value, dtype=float)
        for dy in range(self.pattern_size[0]):
            if not self.wrap and (y + dy >= height):
                break
            for dx in range(self.pattern_size[1]):
                if not self.wrap and (x + dx >= width):
                    break

                pattern[dy, dx,...] = image[(y + dy) % height, (x + dx) % width, ...]
                # pattern[dy, dx] = self.input_image[(y + dy), (x + dx)]

        return pattern
    
    def get_entropy(self, possibilities, method='min_possibility'):
        if method == "min_possibility":
            if possibilities is None:
                return float('inf')
            return len(possibilities)

    def initialize_output_grid(self):
        """Initialize the output grid with all possible patterns."""
        self.output_grid = np.full((self.grid_size[0], self.grid_size[1]), None)
        self.possible_patterns = [[None for _ in range(self.grid_size[1])]
                                  for _ in tqdm(range(self.grid_size[0]), desc="Initializing Grid")]

    def observe(self, mode=observe_mode.RANDOM, chosen_cell=None):
        """Select the most constrained cell and collapse its wave function.
        parameters:
            mode (int): 0 for Modifying Blocks, 1 for Random,
                observe_mode.MB for modifying blocks, observe_mode.RANDOM for random
            chosen_cell: only applicable when using observe_mode.MB or when one wants to observe a specific cell
        
        """
        if mode == observe_mode.RANDOM:
            min_entropy = float('inf')
            chosen_cell = None
            chosen_cells = []
            
            self.observations.append(deepcopy(self.possible_patterns))
            attempt = len(self.observations)
            
            # Find the cell with the least possibilities (highest constraint)
            # for y in tqdm(range(self.grid_size[0]), desc=f"Observation number {attempt}", miniters=20):
            for y in range(self.grid_size[0]):
                for x in range(self.grid_size[1]):
                    if self.output_grid[y, x] is None:
                        entropy = self.get_entropy(self.possible_patterns[y][x])
                        if entropy < min_entropy:
                            min_entropy = entropy
                            chosen_cell = (y, x)
                            chosen_cells.append(chosen_cell)
                        elif entropy == min_entropy:
                            chosen_cell = (y, x)
                            chosen_cells.append(chosen_cell)
                        
            if chosen_cell is None or len(chosen_cells) == 0:
                return False  # All cells are filled

            if min_entropy == 0:
                raise NoWFCSolution
            
            chosen_cell = random.choice(chosen_cells)
        
        y, x = chosen_cell
        
        # Randomly select a pattern weighted by its frequency
        if self.possible_patterns[y][x] is not None:
            if len(self.possible_patterns[y][x]) == 0:
                pattern = self.empty_pattern
                #raise NoWFCSolution
            else:
                pattern = random.choices(
                    list(self.possible_patterns[y][x]),
                    #weights=[self.pattern_frequencies[pat] for pat in self.possible_patterns[y][x]],
                    k=1
                )[0]
        else:
            pattern = random.choices(
                self.patterns,
                #weights=[self.pattern_frequencies[pat] for pat in self.patterns],
                k=1
            )[0]
        self.output_grid[y, x] = pattern
        self.possible_patterns[y][x] = {pattern}
        return True

    def propagate(self):
        """Propagate constraints from the observed cell throughout the grid."""
        changes = True
        while changes:
            changes = False
            for y in range(self.grid_size[0]):
                for x in range(self.grid_size[1]):
                    if self.possible_patterns[y][x] is None:
                        continue
                    if len(self.possible_patterns[y][x]) == 1:
                        pattern = next(iter(self.possible_patterns[y][x]))
                        if pattern != self.empty_pattern:
                            for direction, dy, dx in [('above', -1, 0), ('below', 1, 0), ('left', 0, -1), ('right', 0, 1)]:
                                if not self.wrap and (y + dy >= self.grid_size[0] or x + dx >= self.grid_size[1] or y + dy < 0 or x + dx < 0):
                                    continue
                                
                                ny, nx = (y + dy) % self.grid_size[0], (x + dx) % self.grid_size[1]
                                # ny, nx = (y + dy), (x + dx)
                                valid_patterns = set()
                                if self.possible_patterns[ny][nx] is not None:
                                    for neighbor_pattern in self.possible_patterns[ny][nx]:
                                        if neighbor_pattern in self.adjacency_rules[pattern][direction]:
                                            valid_patterns.add(neighbor_pattern)
                                    if valid_patterns != self.possible_patterns[ny][nx]:
                                        self.possible_patterns[ny][nx] = valid_patterns
                                        changes = True
                                else:
                                    for neighbor_pattern in self.patterns:
                                        if neighbor_pattern in self.adjacency_rules[pattern][direction]:
                                            valid_patterns.add(neighbor_pattern)
                                    if valid_patterns != self.possible_patterns[ny][nx]:
                                        self.possible_patterns[ny][nx] = valid_patterns
                                        changes = True

    def run(self, grid_size, mode=observe_mode.RANDOM, block_size=10, stride=8):
        self.grid_size = grid_size
        self.n_runs += 1
        
        if self.empty_pattern is None:
            self.empty_pattern = len(self.patterns)
        
        if mode==observe_mode.MB:
            if type(block_size) is int:
                block_size = (block_size, block_size)
            if type(stride) is int:
                stride = (stride, stride)
        
        self.initialize_output_grid()
        
        if mode==observe_mode.RANDOM:
            try:
                """Run the WFC algorithm until the grid is fully populated."""
                while True:
                    if not self.observe(mode=mode):
                        break
                    self.propagate()
                            
            except NoWFCSolution:
    
                if self.n_runs >= self.max_runs:
                    print("No Solution Found, Giving Up!")
                    self.n_runs = 0
                    return np.zeros(self.grid_size)
                print("No Solution Found, Retrying...")
                self.observations = []
                return self.run(grid_size)            
                
        elif mode==observe_mode.MB:
            for i in range(0, grid_size[0], stride[0]):
                for j in range(0, grid_size[1], stride[1]):
                    try:
                        self.observe_block(i, j, block_size, grid_size, mode)
                    except NoWFCSolution:
                        if self.n_runs >= self.max_runs:
                            print("No Solution Found, Giving Up!")
                            self.n_runs = 0
                            return np.zeros(self.grid_size)
                        print("No Solution Found, Retrying...")
                        self.observations = []
                        return self.run(grid_size, mode, block_size, stride)

        # Construct the final output image
        final_image = np.zeros(self.grid_size)
        self.gosal_image = np.zeros(self.grid_size[0:2])
        for y in range(self.grid_size[0]):
            for x in range(self.grid_size[1]):
                pattern = self.output_grid[y, x]
                if pattern != self.empty_pattern:
                    final_image[y, x, ...] = np.array(pattern).reshape(*self.pattern_size)[0, 0]
                else:
                    _ = False
                    pat_y = []
                    if y > 0:
                        pat_y.append(np.array(self.output_grid[y-1, x]).reshape(*self.pattern_size)[1, 0, 1]) # Top
                        if x > 0:
                            pat_y.append(np.array(self.output_grid[y-1, x-1]).reshape(*self.pattern_size)[1, 1, 1]) # Top Left
                            pat_y.append(np.array(self.output_grid[y, x-1]).reshape(*self.pattern_size)[0, 1, 1]) # Left
                    pat_y.append(0)
                    
                    for py in pat_y:
                        if py != -32767:
                            pattern_y = py
                            break
                    
                    pat_x = []
                    if x > 0:
                        pat_x.append(np.array(self.output_grid[y, x-1]).reshape(*self.pattern_size)[0, 1, 0]) # Left
                        if y > 0:
                            pat_x.append(np.array(self.output_grid[y-1, x-1]).reshape(*self.pattern_size)[1, 1, 0]) # Top Left
                            pat_x.append(np.array(self.output_grid[y-1, x]).reshape(*self.pattern_size)[1, 0, 0]) # Top
                    pat_x.append(0)
                    
                    for px in pat_x:
                        if px != -32767:
                            pattern_x = px
                            break
                    self.gosal_image[y, x] = 1
                    final_image[y, x, ...] = np.array([pattern_y, pattern_x])

        return final_image
  
    def observe_block(self, i, j, block_size, grid_size, mode):
        try:
            for y in range(i, i+block_size[0]):
                for x in range(j, j+block_size[1]):
                    if y >= grid_size[0] or x >= grid_size[1]:
                        continue
                    cell = (y, x)
                    self.observe(mode, cell)
                    if self.output_grid[y, x] is None:
                        raise
                    self.propagate()
        except NoWFCSolution:
            if self.n_runs_on_block >= self.max_runs:
                print("No Solution Found, Giving Up on this Block!")
                self.n_runs_on_block = 0
                raise NoWFCSolution                
            
            print("No Solution Found, Retrying this Block...")
            self.n_runs_on_block += 1
            self.observe_block(i, j, block_size, grid_size, mode)
            
        self.n_runs_on_block = 0
        return True
 
    def save(self, save_path, overwrite=True):
        pattern_freqs = {str(pat): f for pat, f in self.pattern_frequencies.items()}
        save_state(pattern_freqs, os.path.join(save_path, 'patterns.json'), overwrite=overwrite)
        
        adj = {str(pat): {neigbor_loc: [list(tup) for tup in set_of_pats] for neigbor_loc, set_of_pats in defdict.items()} for pat, defdict in self.adjacency_rules.items()}
        save_state(adj, os.path.join(save_path, 'adjacency.json'), overwrite=overwrite)
        
        pattern_to_number = {str(pat): i for pat, i in self.pattern_to_number.items()}
        save_state(pattern_to_number, os.path.join(save_path, 'pattern_to_number.json'), overwrite=overwrite)

    def load(self, save_path):
       
        pattern_freqs = load_state(os.path.join(save_path, 'patterns.json'))
        self.pattern_frequencies = defaultdict(int, {eval(pat): f for pat, f in pattern_freqs.items()})
        
        adj = load_state(os.path.join(save_path, 'adjacency.json'))
        self.adjacency_rules = defaultdict(default_dict_of_sets)
        # Now fill the self.adjacency_rules with the data from `adj`
        for pat, defdict in adj.items():
            for neigbor_loc, set_of_pats in defdict.items():
                # The key 'pat' is the outer key, and 'neigbor_loc' is the inner key
                # We add the set of tuples to the appropriate set in the defaultdict structure
                self.adjacency_rules[eval(pat)][neigbor_loc] = {tuple(tup) for tup in set_of_pats}

        
        pat2num = load_state(os.path.join(save_path, 'pattern_to_number.json'))
        self.pattern_to_number = {eval(pat): i for pat, i in pat2num.items()}
        self.number_to_pattern = {v:k for k,v in self.pattern_to_number.items()}
        
        if self.remove_low_freq:
            self.patterns = [k for k in self.pattern_frequencies.keys() if self.pattern_frequencies[k] > self.low_freq] 
        else:
            self.patterns = list(self.pattern_frequencies.keys())

class WaveFunctionCollapseVisualizer:
    def __init__(self, wfc: WaveFunctionCollapse, plot_args={}, cmap=None):
        self.wfc =wfc
        self.grid_size = grid_size
        if cmap is not None:
            self.custom_colormap = cmap
        else:
            self.custom_colormap = 'terrain'
        self.plot_args = plot_args
        
    def plot_observations(self):
        n = len(self.wfc.observations)
        fig, axes = plt.subplots(n,1, figsize=(20, n* 4))
        if n == 1:
            axes = [axes]  # Handle the case where n=1

        for idx, observation in enumerate(self.wfc.observations):
            ax = axes[idx]
            grid = np.zeros(self.grid_size, dtype=object)

            # Fill grid with pattern numbers
            for i in range(self.grid_size[0]):
                for j in range(self.grid_size[1]):
                    patterns = observation[i][j]
                    grid[i, j] = ','.join(str(self.wfc.pattern_to_number[p]) for p in patterns)

            # Plot the grid
            ax.set_xticks(np.arange(0, self.grid_size[1] + 1) - 0.5, minor=True)
            ax.set_yticks(np.arange(0, self.grid_size[0] + 1) - 0.5, minor=True)
            ax.grid(which="minor", color="black", linestyle='-', linewidth=1.5)
            ax.tick_params(which="minor", size=0)
            ax.axis('off')  # Turn off the axis

            # Add pattern numbers in each cell
            for (i, j), label in np.ndenumerate(grid):
                ax.text(j, i, label, ha='center', va='center', fontsize=8)

            # ax.set_title(f"Observation {idx + 1}")

        plt.tight_layout()
        plt.show()
    
    def visualize_patterns(self, n_max=None):
        unique_patterns = list(self.wfc.pattern_to_number.keys())
        if n_max is None:
            n = len(unique_patterns)
        else:
            n = min(n_max, len(unique_patterns))
        
        fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
        if n == 1:
            axes = [axes]  # Handle the case where n=1

        for idx in range(n):
            pattern = unique_patterns[idx]
            ax = axes[idx]
            pattern_array = np.array(pattern).reshape(*self.wfc.pattern_size)  # Assuming patterns can be represented as 2D arrays
            if self.wfc.channels_per_image == 2:
                Z = slopes_to_height(pattern_array[..., 0], pattern_array[..., 1])
            elif self.wfc.channels_per_image == 1:
                Z = pattern_array
            ax.imshow(Z, cmap=self.custom_colormap, interpolation="nearest", **self.plot_args)
               
            ax.axis('off')  # Turn off the axis
            ax.set_title(f"Pattern {self.wfc.pattern_to_number[pattern]}")
            for i in range(self.wfc.pattern_size[0]):
                for j in range(self.wfc.pattern_size[1]):
                    if self.wfc.channels_per_image == 2:
                        ax.text(j, i, int(pattern_array[i, j,0]), ha='left', va='center', fontsize=32, color='black')
                        ax.text(j, i, int(pattern_array[i, j,1]), ha='right', va='center', fontsize=32, color='black')
                    elif self.wfc.channels_per_image == 1:
                        ax.text(j, i, int(pattern_array[i, j]), ha='center', va='center', fontsize=32, color='black')
                        
            # ax.text(0, 0, self.pattern_to_number[pattern], ha='center', va='center', fontsize=32, color='black')
            
        plt.tight_layout()
        plt.show()
        
    def visualize_adjacency(self, pattern):
        pattern_number = self.wfc.pattern_to_number[pattern]

        # Initialize a dynamic grid to accommodate multiple adjacent patterns
        # For now, start with 3x3 and expand as necessary
        grid_size = 3
        
        # Start with empty lists to hold patterns for each side
        above_patterns = []
        below_patterns = []
        left_patterns = []
        right_patterns = []
        
        # Get adjacency rules for this pattern (if they exist)
        if pattern in self.wfc.adjacency_rules:
            rules = self.wfc.adjacency_rules[pattern]
            # Collect all patterns for each direction
            if 'above' in rules:
                above_patterns.extend(rules['above'])
            if 'below' in rules:
                below_patterns.extend(rules['below'])
            if 'left' in rules:
                left_patterns.extend(rules['left'])
            if 'right' in rules:
                right_patterns.extend(rules['right'])

        # Determine the grid dimensions needed to fit all patterns

        grid_rows = 1 + len(above_patterns) + len(below_patterns)  
        grid_cols = 1 + len(right_patterns) + len(left_patterns) 
        
        # Initialize an empty grid of pattern arrays
        grid = np.full((grid_rows, grid_cols), fill_value=None)

        # Place the center pattern
        center_row, center_col = len(above_patterns), len(left_patterns)
        grid[center_row, center_col] = pattern

        # Helper function to place patterns in the grid
        def place_patterns(pattern_list, start_row, start_col, row_step, col_step):
            for idx, pat in enumerate(pattern_list):
                grid[start_row + idx * row_step, start_col + idx * col_step] = pat

        # Place above patterns
        place_patterns(above_patterns, 0, center_col, 1, 0)
        # Place below patterns
        place_patterns(below_patterns, center_row + 1, center_col, 1, 0)
        # Place left patterns
        place_patterns(left_patterns, center_row, 0, 0, 1)
        # Place right patterns
        place_patterns(right_patterns, center_row, center_col + 1, 0, 1)
        
        fig, ax = plt.subplots(grid_rows, grid_cols, figsize=(6, 6))
        if grid_cols == 1 and grid_rows == 1:
            ax = np.array(ax).reshape(grid_rows, grid_cols)
        # Plot all patterns in the grid
        for i in range(grid_rows):
            for j in range(grid_cols):
                if i == center_row or j == center_col:
                    pattern_array = np.array(grid[i, j]).reshape(*self.wfc.pattern_size)
                    ax[i,j].grid(which="minor", color="black", linestyle='-', linewidth=1.5)
                    ax[i,j].tick_params(which="minor", size=0)
                    ax[i,j].axis('off')  # Turn off the axis
                    if self.wfc.channels_per_image == 2:
                        Z = slopes_to_height(pattern_array[..., 0], pattern_array[..., 1])
                    elif self.wfc.channels_per_image == 1:
                        Z = pattern_array
                        
                    ax[i,j].imshow(Z, cmap=self.custom_colormap, **self.plot_args,
                            interpolation="nearest")#, extent=(j, j + 1, i, i + 1))

                    # Add grid lines
                    ax[i,j].set_xticks(np.arange(-0.5, grid_cols), minor=True)
                    ax[i,j].set_yticks(np.arange(-0.5, grid_rows), minor=True)

                    ax[i,j].text(0, 0, self.wfc.pattern_to_number[tuple(grid[i, j])], ha='center', va='center', fontsize=10, color='black')
                else:
                    ax[i,j].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
                    ax[i,j].axis('off')  # Turn off the axis


        # Add the title
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Example Usage
    input_image = [[0,0,0,0],
                [0,1,1,1],
                [0,1,2,1],
                [0,1,1,1]]
    
    # input_image = np.dstack((input_image, input_image))
    input_images = [input_image]
    pattern_size=np.array([2,2,1])
    grid_size = (12,12,1)
    
    # grid_size = (12,12)
    # pattern_size=np.array([2,2])#,2])
    input_images = augment_images(input_images)
    for im in input_images:
        sns.heatmap(im, annot=True)
        plt.show()
    
    wfc = WaveFunctionCollapse(input_images, pattern_size=pattern_size, wrap_input=False)
    wfc.match_patterns()
    output_image = wfc.run(grid_size)
    print(output_image)
    
    # Example usage:
    # Assuming you have a grid size, a list of observations, and a mapping from patterns to numbers
    observations = wfc.observations

    pattern_to_number = wfc.pattern_to_number

    plot_args = dict(
        vmin=0,
        vmax=2
    )
    visualizer = WaveFunctionCollapseVisualizer(wfc, plot_args=plot_args)
    # visualizer.plot_observations()
    visualizer.visualize_patterns(n_max=10)
    # for pat in wfc.patterns:
    visualizer.visualize_adjacency(next(iter(wfc.patterns)))
    try:
        sns.heatmap(output_image[..., 0], annot=True)
        plt.show()
    except:
        pass


