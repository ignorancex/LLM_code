import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import time
from matplotlib.animation import FuncAnimation

# Simulation parameters
num_nanorobots = 15  # Number of competing nanorobots
num_cancer_cells = 2  # Number of cancer cells (configurable)
threshold_distance = 0.5  # Distance threshold for stopping the simulation
learning_distance = 8  # Distance threshold for switching to reinforcement learning
biomarker_decay_rate = 0.1  # Biomarker concentration decay rate
learning_rate = 0.5  # Learning rate for nanorobot movement
discount_factor = 0.9  # Discount factor for future rewards
epsilon = 0.1  # Exploration rate for epsilon-greedy policy
biomarker_radius = 40.0  # Radius of biomarker influence

# Environment size and obstacles
env_size = 50
num_biomarkers = 50
num_obstacles = 10

# Initialize cancer cell positions
cancer_cell_positions = [np.random.uniform(-env_size, env_size, size=3) for _ in range(num_cancer_cells)]

# Metrics initialization
total_steps = 0
distances = [[] for _ in range(num_nanorobots)]
rewards = [[] for _ in range(num_nanorobots)]
all_nanorobot_positions = [[] for _ in range(num_nanorobots)]
all_cancer_cell_positions = []
all_biomarker_concentrations = [[] for _ in range(num_nanorobots)]

# Generate initial positions for nanorobots
nanorobot_positions = [
    np.random.uniform(-env_size, env_size, 3) for _ in range(num_nanorobots)
]

# Generate random obstacle positions
obstacle_positions = [np.random.uniform(-env_size, env_size, 3) for _ in range(num_obstacles)]

# Generate biomarker positions around the cancer cells
biomarker_positions = [pos + np.random.uniform(-5, 5, 3) for pos in cancer_cell_positions for _ in range(num_biomarkers)]

# Function to move cancer cells randomly
def random_move(position):
    move = np.random.normal(0, 0.2, 3)
    return position + move

# Function to calculate biomarker concentration
def calculate_biomarker_concentration(nanorobot_pos, biomarker_positions):
    concentrations = []
    for biomarker in biomarker_positions:
        distance = np.linalg.norm(biomarker - nanorobot_pos)
        concentration = np.exp(-distance**2 / biomarker_radius**2)
        concentrations.append(concentration)
    return np.mean(concentrations)

# Function to avoid obstacles
def avoid_obstacles(nanorobot_pos, target_pos):
    for obstacle in obstacle_positions:
        distance_to_obstacle = np.linalg.norm(obstacle - nanorobot_pos)
        if distance_to_obstacle < 1.0:
            avoid_direction = nanorobot_pos - obstacle
            avoid_direction /= np.linalg.norm(avoid_direction)
            target_pos += avoid_direction
    return target_pos

# Function to update nanorobot position
def update_nanorobot(nanorobot_pos, cancer_cell_positions, biomarker_positions):
    concentration = calculate_biomarker_concentration(nanorobot_pos, biomarker_positions)
    # Find the closest cancer cell
    distances_to_cells = [np.linalg.norm(cancer_cell - nanorobot_pos) for cancer_cell in cancer_cell_positions]
    closest_cell_index = np.argmin(distances_to_cells)
    closest_cancer_cell = cancer_cell_positions[closest_cell_index]
    
    distance_to_cell = distances_to_cells[closest_cell_index]
    
    if distance_to_cell > learning_distance:
        max_concentration_index = np.argmax([np.exp(-np.linalg.norm(biomarker - nanorobot_pos)**2) for biomarker in biomarker_positions])
        target_point = biomarker_positions[max_concentration_index]
        target_point = avoid_obstacles(nanorobot_pos, target_point)
        
        direction = target_point - nanorobot_pos
        norm = np.linalg.norm(direction)
        if norm > 0:
            step = direction / norm * 0.5
            nanorobot_pos += step
        reward = concentration
    else:
        direction = closest_cancer_cell - nanorobot_pos
        direction = avoid_obstacles(nanorobot_pos, direction)
        norm = np.linalg.norm(direction)
        if norm > 0:
            step = (direction / norm * (learning_rate / (distance_to_cell + 0.1)))
            nanorobot_pos += step
        reward = -distance_to_cell
    
    return nanorobot_pos, concentration, reward

# Start simulation
fig = plt.figure(figsize=(16, 10), facecolor="w")
ax = fig.add_subplot(111, projection='3d')

# Customizing plot appearance
ax.grid(False)
ax.set_facecolor('whitesmoke')

# Function for updating the plot
def update(frame):
    global cancer_cell_positions, total_steps, biomarker_positions, obstacle_positions

    cancer_cell_positions = [random_move(pos) for pos in cancer_cell_positions]
    all_cancer_cell_positions.append(cancer_cell_positions.copy())
    
    biomarker_positions = [pos + np.random.uniform(-5, 5, 3) for pos in cancer_cell_positions for _ in range(num_biomarkers)]
    
    for i in range(num_nanorobots):
        nanorobot_positions[i], concentration, reward = update_nanorobot(nanorobot_positions[i], cancer_cell_positions, biomarker_positions)
        
        # Store metrics
        distance = np.linalg.norm(cancer_cell_positions[0] - nanorobot_positions[i])  # Calculate distance to the first cancer cell (for metrics)
        distances[i].append(distance)
        rewards[i].append(reward)
        all_nanorobot_positions[i].append(nanorobot_positions[i].copy())
        all_biomarker_concentrations[i].append(concentration)

    # Plotting
    ax.cla()
    ax.set_xlim([-env_size, env_size])
    ax.set_ylim([-env_size, env_size])
    ax.set_zlim([-env_size, env_size])
    ax.grid(False)
    colors = ['blue'] * num_nanorobots
    for i, pos in enumerate(nanorobot_positions):
        ax.scatter(pos[0], pos[1], pos[2], color=colors[i], s=100, label=f'Nanorobot {i+1}')
    
    for pos in cancer_cell_positions:
        ax.scatter(pos[0], pos[1], pos[2], color='red', s=500, label='Cancer Cell')
    ax.scatter(*zip(*biomarker_positions), color='green', alpha=0.2, s=20, marker='^', label='Biomarkers')
    ax.scatter(*zip(*obstacle_positions), color='black', alpha=0.5, label='Obstacles')
    
    ax.set_title(f'Step: {total_steps}', fontsize=20, fontweight='bold')
    
    # Change the position of the legend
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
    
    total_steps += 1

# Animation
ani = FuncAnimation(fig, update, frames=range(1000), repeat=False)
plt.show()
