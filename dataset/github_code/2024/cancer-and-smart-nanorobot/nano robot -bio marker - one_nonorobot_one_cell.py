import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import time
from matplotlib.animation import FuncAnimation

# Simulation parameters
threshold_distance = 0.5  # Distance threshold for stopping the simulation
learning_distance = 8  # Distance threshold for switching to reinforcement learning
biomarker_decay_rate = 0.1  # Biomarker concentration decay rate
learning_rate = 0.5  # Learning rate for nanorobot movement
discount_factor = 0.9  # Discount factor for future rewards
epsilon = 0.1  # Exploration rate for epsilon-greedy policy
biomarker_radius = 40.0  # شعاع تاثیر بیومارکر

# Environment size and obstacles
env_size = 50  # Environment dimension
num_biomarkers = 50  # Number of biomarker points
num_obstacles = 10  # Number of obstacles

#cancer_cell_pos = np.array([5.0, 5.0, 5.0])  # Initial position of the cancer cell
cancer_cell_pos = np.random.uniform(-env_size, env_size, size=3)


# Metrics initialization
total_steps = 0
distances = []
rewards = []
biomarker_concentrations = []
start_time = time.time()  # Start timer
all_nanorobot_positions = []  # To track nanorobot positions for analysis
all_cancer_cell_positions = []  # To track cancer cell positions for analysis
all_biomarker_concentrations = []  # To track biomarker concentrations

# Q-learning parameters
action_space = ['toward_biomarker', 'toward_cancer_cell']  # Possible actions
q_table = np.zeros((2, 2))  # Q-values for each state-action pair

# Function to randomly move cancer cell
def random_move(position):
    move = np.random.normal(0, 0.2, 3)  # Small random movement
    return position + move

# Function to calculate biomarker concentration using density estimation
def calculate_biomarker_concentration(nanorobot_pos, biomarker_positions):
    concentrations = []
    for biomarker in biomarker_positions:
        distance = np.linalg.norm(biomarker - nanorobot_pos)
        concentration = np.exp(-distance**2 / biomarker_radius**2)  # Gaussian d
        concentrations.append(concentration)
    return np.mean(concentrations)

# Generate random obstacle positions
obstacle_positions = [np.random.uniform(-env_size, env_size, 3) for _ in range(num_obstacles)]

# Function to move obstacles randomly
def move_obstacles(obstacle_positions):
    for i in range(len(obstacle_positions)):
        obstacle_positions[i] += np.random.normal(0, 0.0, 3)  # Small random movement
        obstacle_positions[i] = np.clip(obstacle_positions[i], -env_size, env_size)  # Keep within bounds
    return obstacle_positions

# Function to check for obstacles and avoid them
def avoid_obstacles(nanorobot_pos, target_pos):
    for obstacle in obstacle_positions:
        distance_to_obstacle = np.linalg.norm(obstacle - nanorobot_pos)
        if distance_to_obstacle < 1.0:  # If too close to an obstacle
            avoid_direction = nanorobot_pos - obstacle
            avoid_direction /= np.linalg.norm(avoid_direction)
            target_pos += avoid_direction  # Adjust target position away from obstacle
    return target_pos

# Function to choose action using epsilon-greedy strategy
def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.choice(action_space)  # Explore
    else:
        return action_space[np.argmax(q_table[state])]  # Exploit

# Function to update Q-table
def update_q_table(state, action, reward, next_state):
    action_index = 0 if action == 'toward_biomarker' else 1
    q_table[state, action_index] += learning_rate * (reward + discount_factor * np.max(q_table[next_state]) - q_table[state, action_index])

# Function to update nanorobot position based on learning
def update_nanorobot(nanorobot_pos, cancer_cell_pos, biomarker_positions):
    concentration = calculate_biomarker_concentration(nanorobot_pos, biomarker_positions)
    distance_to_cell = np.linalg.norm(cancer_cell_pos - nanorobot_pos)
    
    if distance_to_cell > learning_distance:
        max_concentration_index = np.argmax([np.exp(-np.linalg.norm(biomarker - nanorobot_pos)**2) for biomarker in biomarker_positions])
        target_point = biomarker_positions[max_concentration_index]
        target_point = avoid_obstacles(nanorobot_pos, target_point)
        
        action = choose_action(0)
        if action == 'toward_biomarker':
            direction = target_point - nanorobot_pos
            norm = np.linalg.norm(direction)
            if norm > 0:
                step = direction / norm * 0.5
                nanorobot_pos += step
            
        reward = concentration  # Store reward
        rewards.append(reward)  # Store reward
        next_state = 0
        update_q_table(0, action, reward, next_state)

    else:
        action = choose_action(1)
        direction = cancer_cell_pos - nanorobot_pos
        direction = avoid_obstacles(nanorobot_pos, direction)
        norm = np.linalg.norm(direction)
        
        if norm > 0:
            step = (direction / norm * (learning_rate / (distance_to_cell + 0.1)))
            nanorobot_pos += step
        
        reward = -distance_to_cell
        rewards.append(reward)  # Store penalty (negative reward)
        next_state = 1
        update_q_table(1, action, reward, next_state)
    
    return nanorobot_pos, concentration

# Start simulation
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Random starting position for nanorobot
nanorobot_start_pos = np.array([
    random.uniform(-env_size, -env_size / 2), 
    random.uniform(-env_size, -env_size / 2), 
    random.uniform(-env_size, -env_size / 2)
])  
nanorobot_pos = nanorobot_start_pos

# Initialize biomarker positions around cancer cell
biomarker_positions = [cancer_cell_pos + np.random.uniform(-5, 5, 3) for _ in range(num_biomarkers)]

# Prepare for path tracking
nanorobot_paths = []  # To store nanorobot path
cancer_cell_paths = []  # To store cancer cell path

# Function for updating the plot
def update(frame):
    global nanorobot_pos, cancer_cell_pos, total_steps, biomarker_positions, obstacle_positions

    cancer_cell_pos = random_move(cancer_cell_pos)
    all_cancer_cell_positions.append(cancer_cell_pos.copy())
    
    # Generate biomarker positions around the cancer cell
    biomarker_positions = [cancer_cell_pos + np.random.uniform(-5, 5, 3) for _ in range(num_biomarkers)]
    
    # Calculate distance
    distance = np.linalg.norm(cancer_cell_pos - nanorobot_pos)
    distances.append(distance)

    # Check for stopping condition
    if distance < threshold_distance:
        plt.close(fig)  # Stop the animation by closing the figure
        return

    obstacle_positions = move_obstacles(obstacle_positions)  # Move obstacles randomly
    nanorobot_pos, concentration = update_nanorobot(nanorobot_pos, cancer_cell_pos, biomarker_positions)
    total_steps += 1
    
    all_biomarker_concentrations.append(concentration)
    all_nanorobot_positions.append(nanorobot_pos.copy())
    nanorobot_paths.append(nanorobot_pos.copy())
    cancer_cell_paths.append(cancer_cell_pos.copy())

    ax.cla()
    
    ax.set_xlim([-env_size, env_size])
    ax.set_ylim([-env_size, env_size])
    ax.set_zlim([-env_size, env_size])
    ax.grid(False)
    ax.scatter(nanorobot_pos[0], nanorobot_pos[1], nanorobot_pos[2], c='blue', s=100, label='Nanorobot')
    ax.scatter(cancer_cell_pos[0], cancer_cell_pos[1], cancer_cell_pos[2], c='red', s=500, label='Cancer Cell')
    
    ax.scatter(*zip(*biomarker_positions), c='green', alpha=0.2, s=20, marker='^', label='Biomarkers')
    ax.scatter(*zip(*obstacle_positions), c='black', alpha=0.5, label='Obstacles')

    ax.set_title(f'Step: {total_steps}, Distance to Cancer Cell: {distance:.2f}, Instantaneous Biomarker Concentration: {concentration:.10f}', fontsize=16)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.legend(fontsize=12)
    
    if distance < threshold_distance:
        plt.close(fig)

# Animation
ani = FuncAnimation(fig, update, frames=range(1000), repeat=False)

plt.show()

# End of simulation and evaluation
end_time = time.time()
simulation_duration = end_time - start_time

final_distance = np.linalg.norm(cancer_cell_pos - nanorobot_pos)
average_distance = np.mean(distances)
average_biomarker_concentration = np.mean(all_biomarker_concentrations)

print(f"Total Steps: {total_steps}")
print(f"Final Distance: {final_distance:.2f}")
print(f"Average Distance: {average_distance:.2f}")
print(f"Average Biomarker Concentration: {average_biomarker_concentration:.2f}")
print(f"Simulation Duration: {simulation_duration:.2f} seconds")

# Plot biomarker concentration vs. distance
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Steps')
ax1.set_ylabel('Biomarker Concentration', color=color)
ax1.plot(range(len(all_biomarker_concentrations)), all_biomarker_concentrations, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('Distance to Cancer Cell', color=color)
ax2.plot(range(len(distances)), distances, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title("Biomarker Concentration and Distance to Cancer Cell over Steps")
plt.show()

# 3D path plotting
fig = plt.figure(figsize=(10, 6))
ax2 = fig.add_subplot(111, projection='3d')
ax2.plot(*zip(*all_nanorobot_positions), color='blue', label='Nanorobot Path')
ax2.plot(*zip(*all_cancer_cell_positions), color='red', label='Cancer Cell Path')
ax2.scatter(*zip(*obstacle_positions), c='black', alpha=0.7, s=50, label='Obstacles')
ax2.set_title('3D Path of Nanorobot and Cancer Cell')
ax2.set_xlabel('X Axis')
ax2.set_ylabel('Y Axis')
ax2.set_zlabel('Z Axis')
ax2.legend()
plt.show()




# ایجاد زیرنمودارها
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# X-Y Plane
biomarker_density_xy = np.zeros((30, 30))
x = np.linspace(-env_size, env_size, 30)
y = np.linspace(-env_size, env_size, 30)

# محاسبه غلظت بیومارکرها با شعاع گوسی
for biomarker in biomarker_positions:
    xi = np.digitize(biomarker[0], x) - 1
    yi = np.digitize(biomarker[1], y) - 1
    if 0 <= xi < 30 and 0 <= yi < 30:
        # افزودن غلظت گوسی
        for i in range(30):
            for j in range(30):
                distance = np.sqrt((biomarker[0] - x[i]) ** 2 + (biomarker[1] - y[j]) ** 2)
                biomarker_density_xy[xi, yi] += np.exp(-0.5 * (distance / biomarker_radius) ** 2)

# افزودن سلول سرطانی
cancer_cell_xi = np.digitize(cancer_cell_pos[0], x) - 1
cancer_cell_yi = np.digitize(cancer_cell_pos[1], y) - 1
if 0 <= cancer_cell_xi < 30 and 0 <= cancer_cell_yi < 30:
    for i in range(30):
        for j in range(30):
            distance = np.sqrt((cancer_cell_pos[0] - x[i]) ** 2 + (cancer_cell_pos[1] - y[j]) ** 2)
            biomarker_density_xy[cancer_cell_xi, cancer_cell_yi] += 10 * np.exp(-0.5 * (distance / biomarker_radius) ** 2)

# نمایش X-Y Plane
cax1 = axs[0].imshow(biomarker_density_xy.T, extent=[-env_size, env_size, -env_size, env_size], origin='lower', cmap='magma')
axs[0].set_title('Biomarker Density Heatmap (X-Y Plane)')
axs[0].set_xlabel('X Coordinate')
axs[0].set_ylabel('Y Coordinate')

# X-Z Plane
biomarker_density_xz = np.zeros((30, 30))
z = np.linspace(-env_size, env_size, 30)

for biomarker in biomarker_positions:
    xi = np.digitize(biomarker[0], x) - 1
    zi = np.digitize(biomarker[2], z) - 1
    if 0 <= xi < 30 and 0 <= zi < 30:
        for i in range(30):
            for j in range(30):
                distance = np.sqrt((biomarker[0] - x[i]) ** 2 + (biomarker[2] - z[j]) ** 2)
                biomarker_density_xz[xi, zi] += np.exp(-0.5 * (distance / biomarker_radius) ** 2)

# افزودن سلول سرطانی به X-Z Plane
cancer_cell_xi = np.digitize(cancer_cell_pos[0], x) - 1
cancer_cell_zi = np.digitize(cancer_cell_pos[2], z) - 1
if 0 <= cancer_cell_xi < 30 and 0 <= cancer_cell_zi < 30:
    for i in range(30):
        for j in range(30):
            distance = np.sqrt((cancer_cell_pos[0] - x[i]) ** 2 + (cancer_cell_pos[2] - z[j]) ** 2)
            biomarker_density_xz[cancer_cell_xi, cancer_cell_zi] += 10 * np.exp(-0.5 * (distance / biomarker_radius) ** 2)

# نمایش X-Z Plane
cax2 = axs[1].imshow(biomarker_density_xz.T, extent=[-env_size, env_size, -env_size, env_size], origin='lower', cmap='magma')
axs[1].set_title('Biomarker Density Heatmap (X-Z Plane)')
axs[1].set_xlabel('X Coordinate')
axs[1].set_ylabel('Z Coordinate')

# Y-Z Plane
biomarker_density_yz = np.zeros((30, 30))

for biomarker in biomarker_positions:
    yi = np.digitize(biomarker[1], y) - 1
    zi = np.digitize(biomarker[2], z) - 1
    if 0 <= yi < 30 and 0 <= zi < 30:
        for i in range(30):
            for j in range(30):
                distance = np.sqrt((biomarker[1] - y[i]) ** 2 + (biomarker[2] - z[j]) ** 2)
                biomarker_density_yz[yi, zi] += np.exp(-0.5 * (distance / biomarker_radius) ** 2)

# افزودن سلول سرطانی به Y-Z Plane
cancer_cell_yi = np.digitize(cancer_cell_pos[1], y) - 1
cancer_cell_zi = np.digitize(cancer_cell_pos[2], z) - 1
if 0 <= cancer_cell_yi < 30 and 0 <= cancer_cell_zi < 30:
    for i in range(30):
        for j in range(30):
            distance = np.sqrt((cancer_cell_pos[1] - y[i]) ** 2 + (cancer_cell_pos[2] - z[j]) ** 2)
            biomarker_density_yz[cancer_cell_yi, cancer_cell_zi] += 10 * np.exp(-0.5 * (distance / biomarker_radius) ** 2)

# نمایش Y-Z Plane
cax3 = axs[2].imshow(biomarker_density_yz.T, extent=[-env_size, env_size, -env_size, env_size], origin='lower', cmap='magma')
axs[2].set_title('Biomarker Density Heatmap (Y-Z Plane)')
axs[2].set_xlabel('Y Coordinate')
axs[2].set_ylabel('Z Coordinate')

# افزودن colorbar برای هر زیرنمودار
fig.colorbar(cax1, ax=axs[0], label='Biomarker Density')
fig.colorbar(cax2, ax=axs[1], label='Biomarker Density')
fig.colorbar(cax3, ax=axs[2], label='Biomarker Density')

plt.tight_layout()
plt.show()
