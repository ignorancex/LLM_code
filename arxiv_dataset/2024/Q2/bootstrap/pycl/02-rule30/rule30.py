import pyopencl as cl
import numpy as np

# Kernel code for applying Rule 30

def main():
    # Read kernel code from external file
    with open("rule30.cl", "r") as file:
        kernel_code = file.read()

    # Initialize OpenCL context and command queue
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context)

    # Load and compile the kernel
    program = cl.Program(context, kernel_code).build()

    # Parameters
    width = 100  # Width of the automaton
    iterations = 50  # Number of iterations to simulate

    # Initialize input and output arrays
    initial_state = np.zeros(width, dtype=np.int32)
    initial_state[width // 2] = 1  # Set the initial state to have a single '1' in the middle
    output_state = np.zeros(width, dtype=np.int32)

    # Create OpenCL buffers
    input_buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=initial_state)
    output_buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE, size=output_state.nbytes)

    # Simulation loop
    for i in range(iterations):
        # Execute the kernel
        program.rule30(queue, (width,), None, input_buffer, output_buffer, np.int32(width))

        # Swap input and output buffers for next iteration
        input_buffer, output_buffer = output_buffer, input_buffer

        # Read the output buffer back to host memory
        cl.enqueue_copy(queue, output_state, input_buffer).wait()

        # Print the current state
        print(''.join(['*' if cell == 1 else ' ' for cell in output_state]))

if __name__ == "__main__":
    main()
