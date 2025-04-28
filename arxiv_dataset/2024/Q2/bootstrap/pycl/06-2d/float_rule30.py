import pyopencl as cl
import numpy as np

# Kernel code for applying Rule 30
def main():
    # Read kernel code from external file
    try:
        with open("float_rule30.cl", "r") as file:
            kernel_code = file.read()
    except FileNotFoundError:
        print("Kernel file not found.")
        return

    # Initialize OpenCL context and command queue
    try:
        context = cl.create_some_context()
        queue = cl.CommandQueue(context)
    except cl.Error as e:
        print(f"Error initializing OpenCL context: {e}")
        return

    # Load and compile the kernel
    try:
        program = cl.Program(context, kernel_code).build()
    except cl.Error as e:
        print(f"Error building the kernel: {e}")
        return

    # Parameters
    width = 100  # Width of the automaton
    iterations = 10  # Number of iterations to simulate

    # Initialize input buffer
    initial_state = np.zeros((2, width), dtype=np.double)
    # Set the initial state to have a single '1' in the middle of both arrays
    initial_state[0][width // 2] = 1.0
    initial_state[1][width // 2] = 1.0

    try:
        input_buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=initial_state)
        output_state = np.zeros((2, width), dtype=np.double)
        output_buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE, size=output_state.nbytes)
    except cl.Error as e:
        print(f"Error creating buffers: {e}")
        return

    # Simulation loop
    for i in range(iterations):
        queue.finish()  # Ensure the kernel execution is finished before swapping buffers
        # Read the output buffer back to the host

        cl.enqueue_copy(queue, output_state, input_buffer).wait()
        print("input")
        print(''.join(['*' if cell > 0.5 else '.' for cell in output_state[0, :]]))
        print(''.join(['*' if cell > 0.5 else '.' for cell in output_state[1, :]]))
        cl.enqueue_copy(queue, output_state, output_buffer).wait()
        print("output")
        print(''.join(['*' if cell > 0.5 else '.' for cell in output_state[0, :]]))
        print(''.join(['*' if cell > 0.5 else '.' for cell in output_state[1, :]]))
        print('')


        try:
            # Execute the kernel
            program.rule30(queue, (2, width), None, input_buffer, output_buffer, np.int32(width))
            queue.finish()  # Ensure the kernel execution is finished before swapping buffers
        except cl.Error as e:
            print(f"Error during kernel execution: {e}")
            return

        # Swap input and output buffers for next iteration
        input_buffer, output_buffer = output_buffer, input_buffer

if __name__ == "__main__":
    main()
