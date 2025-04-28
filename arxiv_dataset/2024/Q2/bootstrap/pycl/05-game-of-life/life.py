import numpy as np
import pyopencl as cl

# Define the size of the grid
WIDTH = 100
HEIGHT = 100

# Define the kernel that will compute the next state of the grid
kernel_code = """
__kernel void game_of_life(__global const int *current_grid, __global int *next_grid, const int width, const int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int idx = y * width + x;

    if (x >= width || y >= height) return;

    int num_neighbors = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;

            int nx = (x + dx + width) % width;
            int ny = (y + dy + height) % height;
            int nidx = ny * width + nx;

            num_neighbors += current_grid[nidx];
        }
    }

    int current_state = current_grid[idx];
    int next_state = 0;

    if (current_state == 1 && (num_neighbors == 2 || num_neighbors == 3)) {
        next_state = 1;
    } else if (current_state == 0 && num_neighbors == 3) {
        next_state = 1;
    }

    next_grid[idx] = next_state;
}
"""

# Initialize OpenCL context and queue
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)

# Compile the kernel
program = cl.Program(context, kernel_code).build()

# Initialize the grid with random values
current_grid = np.random.randint(2, size=(HEIGHT, WIDTH)).astype(np.int32)
next_grid = np.zeros_like(current_grid)

# Create OpenCL buffers
current_grid_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=current_grid)
next_grid_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, next_grid.nbytes)

# Define the global size and local size
global_size = (WIDTH, HEIGHT)

# Run the Game of Life for a number of iterations
num_iterations = 100
for iteration in range(num_iterations):
    # Execute the kernel
    program.game_of_life(queue, global_size, None, current_grid_buf, next_grid_buf, np.int32(WIDTH), np.int32(HEIGHT))

    # Read the result back to host
    cl.enqueue_copy(queue, next_grid, next_grid_buf).wait()

    # Swap buffers
    current_grid, next_grid = next_grid, current_grid
    current_grid_buf, next_grid_buf = next_grid_buf, current_grid_buf

    # Optionally, print or visualize the grid here
    print(f"Iteration {iteration + 1}")
    print(current_grid)

print("Game of Life finished.")
