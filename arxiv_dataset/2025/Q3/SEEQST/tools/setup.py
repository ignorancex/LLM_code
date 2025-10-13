from itertools import product
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


def get_selective_blocks(num_qubits, wanted_indexes):
    """
    Given a number of qubits and a list of wanted index pairs,
    this function returns:
    1. List of selective blocks (bitwise XOR of index pairs)
    2. Dictionary mapping each block to contributing index pairs.

    Parameters:
        num_qubits (int): Number of qubits (N)
        wanted_indexes (list of tuples): Pairs of indexes (i, j) to process.

    Returns:
        selective_blocks (list of int)
        block_map (dict): {block_value: [index_pairs]}
    """
    max_index = 2 ** num_qubits
    selective_blocks = []
    block_map = {}

    for index_pair in wanted_indexes:
        i, j = index_pair

        # Assert that indices are within valid range
        assert 0 <= i < max_index and 0 <= j < max_index, \
            f"Index pair {index_pair} exceeds valid range [0, {max_index - 1}] for {num_qubits} qubits."

        # XOR operation
        xor_result = i ^ j  # Faster and simpler than using binary strings

        # Add to list if not already there
        if xor_result not in selective_blocks:
            selective_blocks.append(xor_result)

        # Update block_map
        if xor_result not in block_map:
            block_map[xor_result] = []
        block_map[xor_result].append((i, j))

    return selective_blocks, block_map


def generate_selective_elements(selective_blocks, wanted_indexes, num_qubits):
    """
    Generate selective elements from selective_blocks using
    bit-flip (X) and identity (I) operators on binary tuples.

    Parameters:
        selective_blocks (list): List of integers (e.g., [3,2]).
        wanted_indexes (list): List of wanted index pairs (not used directly here, kept for structure).
        num_qubits (int): Number of qubits (N), determines binary string length.

    Returns:
        List of lists: Each sublist contains 2^N decimal indices for one selective_block.
    """
    selective_elements = []

    for block in selective_blocks:
        # Convert to N-bit binary
        block_bin = format(block, f'0{num_qubits}b')

        # Create pair: (zero_state, block_state)
        zero_state = '0' * num_qubits
        pair = (zero_state, block_bin)

        # Apply (I,X)^⊗N: For each bit, apply I or X (flip bit or keep same)
        group = []
        for i in range(2 ** num_qubits):
            mask = format(i, f'0{num_qubits}b')
            # Apply X where mask has 1, else I
            new_state = ''.join(
                str(int(pair[0][j]) ^ int(mask[j])) for j in range(num_qubits)
            )
            new_block = ''.join(
                str(int(pair[1][j]) ^ int(mask[j])) for j in range(num_qubits)
            )
            # Convert both to decimal and store as a tuple
            index_pair = (int(new_state, 2), int(new_block, 2))
            group.append(index_pair)

        selective_elements.append(group)

    return selective_elements


def generate_observable_sets(selective_blocks, num_qubits):
    """
    For each selective block, generate two sets of Pauli observables [E, O],
    where E = observables with even number of 'Y', O = with odd number of 'Y'.

    Parameters:
        selective_blocks (list): List of integers like [3, 4]
        num_qubits (int): Number of qubits (N)

    Returns:
        dict: {block: [E_set, O_set]} where E_set and O_set are lists of Pauli strings
    """
    observable_sets = {}

    for block in selective_blocks:
        bin_block = format(block, f'0{num_qubits}b')

        # Prepare options for each qubit position
        pauli_options = []
        for bit in bin_block:
            if bit == '1':
                pauli_options.append(('X', 'Y'))  # bit flip positions
            else:
                pauli_options.append(('I', 'Z'))  # identity/control positions

        # Generate all combinations (2^N strings)
        all_observables = [''.join(p) for p in product(*pauli_options)]

        # Split into even-Y and odd-Y sets
        even_set = [obs for obs in all_observables if obs.count('Y') % 2 == 0]
        odd_set  = [obs for obs in all_observables if obs.count('Y') % 2 == 1]

        # Store result
        observable_sets[block] = [even_set, odd_set]

    return observable_sets



def plot_density_matrix_highlight(wanted_indexes, result, selective_blocks, num_qubits):
    """
    Plot N-qubit density matrix with:
      - Wanted indexes in red with legend
      - Selective block indexes in distinct colors with legend
      - Binary axis ticks
      - Grid aligned to cell centers

    Parameters:
        wanted_indexes (list of tuples): List of (i,j) pairs (highlighted in red)
        result (list of lists): List of [(i,j), ...] for each selective block
        selective_blocks (list): List of block integers corresponding to result
        num_qubits (int): Number of qubits (N)
    """
    dim = 2 ** num_qubits
    fig, ax = plt.subplots(figsize=(6, 6))

    # Move X ticks to top
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(dim) + 0.5)
    ax.set_xticklabels([format(i, f'0{num_qubits}b') for i in range(dim)], rotation=90)

    # Set Y ticks
    ax.set_yticks(np.arange(dim) + 0.5)
    ax.set_yticklabels([format(i, f'0{num_qubits}b') for i in range(dim)])
    plt.setp(ax.get_xticklabels(), rotation=45)
    # Set limits to match cell centers
    ax.set_xlim(0, dim)
    ax.set_ylim(0,dim)
    ax.invert_yaxis()


    # Draw grid centered between ticks
    ax.set_xticks(np.arange(dim), minor=True)
    ax.set_yticks(np.arange(dim), minor=True)
    ax.grid(which='minor', color='lightgray', linestyle='-', linewidth=1)
    ax.tick_params(which='major', length=0)

    # Draw outer box
    ax.set_frame_on(True)

    # Color palette
    color_list = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())

    legend_elements = []

    # Plot selective block indexes
    for idx, block in enumerate(selective_blocks):
        color = color_list[idx % len(color_list)]
        for (i, j) in result[idx]:
            ax.add_patch(Rectangle((j,  i ), 1, 1, color=color, alpha=0.5))
        legend_elements.append(Line2D([0], [0], marker='s', color='w', label=f'Selective Block {block}',
                                      markerfacecolor=color, markersize=12, alpha=0.6))

    # Plot wanted indexes (on top of everything)
    for i, j in wanted_indexes:
        ax.add_patch(Rectangle((j, i), 1, 1, color='red', alpha=0.9))
    legend_elements.insert(0, Line2D([0], [0], marker='s', color='w', label='Wanted Index',
                                     markerfacecolor='red', markersize=12, alpha=0.9))

    # Add legend
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)

    # Title and aspect
    ax.set_title(f'{num_qubits}-Qubit Density Matrix', y=1.08)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()

def build_parallel_entangler_blocks(selective_blocks, num_qubits):
    """
    Build parallel GHZ-style entangling gate sequences for each selective block.
    Returns two gate sequences per block: one with RY90 and one with RX90 (in reverse order).
    """
    all_sequences = []

    for block in selective_blocks:
        bin_str = format(block, f'0{num_qubits}b')
        active_qubits = [i for i, bit in enumerate(bin_str[::1]) if bit == '1']  # LSB = qubit 0

        if not active_qubits:
            all_sequences.append([''])  # No gates needed
            continue

        sequence = []

        # Step 1: Initial rotation on first qubit (arbitrary choice)
        sequence.append(f'(RY90:{active_qubits[0]})')

        head = [active_qubits[0]]
        tail = active_qubits[1:]

        # Step 2: GHZ layering: use ALL head qubits as controls
        while tail:
            new_tail = []
            for h in head:
                if not tail:
                    break
                # Assign one tail target to this control
                tgt = tail.pop(0)
                sequence.append(f'(CNOT:{h},{tgt})')
                new_tail.append(tgt)
            head.extend(new_tail)

        # Step 3: Create RX90 version of same circuit
        rx_sequence = [gate.replace('RY90', 'RX90') for gate in sequence]

        # Step 4: Return both sequences in reverse order
        all_sequences.append([''.join(sequence[::-1]), ''.join(rx_sequence[::-1])])

    return all_sequences




def build_non_entangling_circuits(selective_blocks, num_qubits):
    """
    For each selective block, generate all RX/RY tensor product combinations
    over active qubits (where the bit is 1 in the block binary).
    
    Returns list of list of circuits (each inner list is all combinations for a block).
    """
    all_block_circuits = []

    for block in selective_blocks:
        bin_str = format(block, f'0{num_qubits}b')
        active_qubits = [i for i, bit in enumerate(bin_str[::]) if bit == '1']  # LSB = qubit 0

        # Generate all 2^m combinations of RX/RY for m active qubits
        gate_choices = list(product(['RY90', 'RX90'], repeat=len(active_qubits)))

        circuits = []
        for choice in gate_choices:
            circuit_str = ''.join([f'({gate}:{q})' for gate, q in zip(choice, active_qubits)])
            circuits.append(circuit_str)

        all_block_circuits.append(circuits)

    return all_block_circuits


def parse_circuit(text_circuits, n_qubits,initial_text=""):
    """
    Convert a text-based circuit description into Qiskit circuits.
    
    Args:
        text_circuits (list of str): List of circuit descriptions in text format.
        n_qubits (int): Number of qubits and classical bits in each circuit.

    Returns:
        list of QuantumCircuit: List of Qiskit QuantumCircuit objects.
    """
    circuits = []

    for circuit_text in text_circuits:
        circuit_text=initial_text+circuit_text
        # Initialize quantum and classical registers
        qreg = QuantumRegister(n_qubits, 'q')
        creg = ClassicalRegister(n_qubits, 'c')
        qc = QuantumCircuit(qreg, creg)

        # Split operations while handling concatenated gates
        operations = circuit_text.split(')')

        for op in operations:
            if not op.strip():  # Skip empty parts
                continue
            op = op.strip().strip('(')  # Remove leading (
            gate_info = op.split(':')

            if len(gate_info) < 2:
                continue  # Skip invalid formats

            gate_name = gate_info[0]
            qubit_indices = list(map(int, gate_info[1].split(',')))  # Extract qubit indices

            # Map gate names to Qiskit gates
            if gate_name == "RX90":
                qc.rx(3.14159/2, qubit_indices[0])
            elif gate_name == "RY90":
                qc.ry(3.14159/2, qubit_indices[0])
            elif gate_name == "CNOT":
                qc.cx(qubit_indices[0], qubit_indices[1])
            elif gate_name == "H":
                qc.h(qubit_indices[0])
            elif gate_name == "MEAS":
                qc.measure(qubit_indices[0], qubit_indices[0])
            else:
                raise ValueError(f"Unsupported gate: {gate_name}")
            
        qc.measure_all()

        circuits.append(qc)

    return circuits



# Helper function to format complex numbers for display
def format_complex(val):
    real_part = val.real
    imag_part = val.imag
    if abs(real_part) < 1e-9: real_part = 0
    if abs(imag_part) < 1e-9: imag_part = 0
    if real_part == 0 and imag_part == 0: return "0.00"
    if imag_part == 0: return f"{real_part:.3f}"
    if real_part == 0: return f"{imag_part:.3f}j"
    sign = '+' if imag_part >= 0 else '-'
    return f"{real_part:.3f}\n{sign} {abs(imag_part):.3f}j"

# Helper function to pick black or white text for readability
def get_text_color(bg_color):
    rgb = mcolors.to_rgb(bg_color)
    luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return 'white' if luminance < 0.5 else 'black'


def create_plotter(wanted_indexes, result, selective_blocks, num_qubits):
    """
    Creates a plotting function with a pre-defined style.
    
    Returns:
        A function that only needs 'rho' to make a plot.
    """
    
    def plot(rho):
        dim = 2 ** num_qubits
        fig, ax = plt.subplots(figsize=(6, 6))

        # --- Axes and Ticks (matched to original style) ---
        ax.xaxis.tick_top()
        ax.set_xticks(np.arange(dim) + 0.5)
        ax.set_xticklabels([format(i, f'0{num_qubits}b') for i in range(dim)])
        ax.set_yticks(np.arange(dim) + 0.5)
        ax.set_yticklabels([format(i, f'0{num_qubits}b') for i in range(dim)])
        plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
        
        ax.set_xlim(0, dim)
        ax.set_ylim(0, dim)
        ax.invert_yaxis()

        # --- Grid (matched to original style) ---
        ax.set_xticks(np.arange(dim), minor=True)
        ax.set_yticks(np.arange(dim), minor=True)
        ax.grid(which='minor', color='lightgray', linestyle='-', linewidth=1)
        ax.tick_params(which='major', length=0)
        
        ax.set_frame_on(True)
        
        colors = list(mcolors.TABLEAU_COLORS.values())
        legend_items = []

        # --- Plot colored blocks ---
        for i, label in enumerate(selective_blocks):
            color = colors[i % len(colors)]
            text_color = get_text_color(color)
            for (r, c) in result[i]: # r for row, c for column
                ax.add_patch(Rectangle((c, r), 1, 1, color=color, alpha=0.6))
                ax.text(c + 0.5, r + 0.5, format_complex(rho[r, c]), ha='center', va='center', color=text_color, fontsize=10)
            legend_items.append(Line2D([0], [0], marker='s', color='w', label=f'Block {label}',
                                       markerfacecolor=color, markersize=12, alpha=0.6))

        # --- Plot wanted indexes (red cells) ---
        text_color = get_text_color('red')
        for (r, c) in wanted_indexes:
            ax.add_patch(Rectangle((c, r), 1, 1, color='red', alpha=0.9))
            ax.text(c + 0.5, r + 0.5, format_complex(rho[r, c]), ha='center', va='center', color=text_color, fontsize=10)
        legend_items.insert(0, Line2D([0], [0], marker='s', color='w', label='Wanted Index',
                                     markerfacecolor='red', markersize=12, alpha=0.9))

        # --- Final touches (matched to original style) ---
        ax.legend(handles=legend_items, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
        ax.set_title(f'{num_qubits}-Qubit Density Matrix', y=1.08)
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.show()

    return plot



def generate_experiment(wanted_indexes,N,
                        options={"density matrix plot":True, "wanted elements":True ,"selective elements":True,'circuits text':True,'non entangling circuits text':True,'observable':True}
                        ):
    selective_blocks, block_dict = get_selective_blocks(N, wanted_indexes)
    
    print('')
    print(f"Wanted Indexes {wanted_indexes} corresponds to these Selective Blocks:", selective_blocks)
    print('')
    
    
    result = generate_selective_elements(selective_blocks, wanted_indexes, N)
            
    observable_dict = generate_observable_sets(selective_blocks, N)

        
    
    sel_circ_text = build_parallel_entangler_blocks(selective_blocks, N)
            
    sel_circ_text_non_entangle = build_non_entangling_circuits(selective_blocks, N)
          
            
    for i, block in enumerate(selective_blocks):
        print('********************')
        print(f"Selective Block {block}")
        if options["wanted elements"]:
            print('')
            print(f"Wanted Elements {block_dict[block]}")
        if options["selective elements"]:
            print('')
            print(f"Selective elements in block",result[i])
        
        if options['observable']: 
            (E, O) = observable_dict[block]
            print('')
            print(f"Pauli Observables:")
            print("  Even-Y set (E):", E)
            print("  Odd-Y set  (O):", O)
            
        if options['circuits text']: 
            if len(sel_circ_text[i])==2:
                (ry_seq, rx_seq) =sel_circ_text[i]
                print('')
                print(f"SEEQST circuit text:")
                print("  Circuit 1:", ry_seq)
                print("  Circuit 2:", rx_seq) 
            else: 
                print('')
                print(f"SEEQST circuit text:")
                print("  Circuit 1:", sel_circ_text[i])
           
            
        if options['non entangling circuits text']:
            print('')
            print("Non-entangling cirucits:")
            print(sel_circ_text_non_entangle[i])
            

    if options["density matrix plot"]:
        plot_density_matrix_highlight(wanted_indexes, result, selective_blocks, N)  
        
    plot_rho = create_plotter(wanted_indexes, result, selective_blocks, N)      
            
    sel_circ_qiskit=[ parse_circuit(i,N) for i in sel_circ_text] 
    non_e_circ_qiskit=[parse_circuit(i,N) for i in sel_circ_text_non_entangle]       
    
    return selective_blocks, sel_circ_text, sel_circ_text_non_entangle, sel_circ_qiskit, non_e_circ_qiskit, plot_rho
    
    
    
    

