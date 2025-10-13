import pennylane as qml
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from sklearn.metrics import pairwise_distances


class QuantumCircuit(nn.Module):
    def __init__(self, n_qubits, graph_adj, n_layers=1):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.graph_adj = graph_adj
        self.n_nodes = 2**n_qubits
        self.qubit_connections = self.compute_edge_phases(graph_adj, n_qubits)
        print("Qubit connections matrix:\n", self.qubit_connections)
        print("graph adjacency matrix:\n", graph_adj)
        # Initialize device
        self.dev = qml.device("default.qubit", wires=n_qubits)

        # Calculate number of parameters
        # non_zeros = np.count_nonzero(self.qubit_connections)

        self.rot_params_per_layer = (
            n_qubits * (n_qubits + 2)
        ) // 2 + n_qubits  # RY rotations + CRY gates
        self.qft_params_per_layer = (n_qubits * (n_qubits - 1)) // 2  # CRZ gates in QFT

        # Parameter initialization
        self.ry_thetas = nn.Parameter(
            torch.rand(n_layers, self.rot_params_per_layer, dtype=torch.float32)
        )
        if graph_adj is not None:
            # Force minimum connectivity
            if np.any(graph_adj.sum(axis=1) == 0):
                graph_adj = graph_adj + np.eye(graph_adj.shape[0]) * 1e-4

            self.qft_phases = nn.Parameter(
                torch.tensor(
                    self.init_qft_phases_from_connections(), dtype=torch.float32
                )
            )
        else:
            # random initialization
            self.qft_phases = nn.Parameter(
                torch.rand(n_layers, self.qft_params_per_layer) * 0.5 + 0.1
            )

        # Create QNode
        self.qnode = qml.QNode(
            self.circuit, self.dev, interface="torch", diff_method="adjoint"
        )

    # 2. ROBUST PHASE INITIALIZATION
    def compute_edge_phases(self, adj_matrix, n_qubits, base_phase=0.1, noise=0.01):
        """Handle uniform/singular graphs with base phases + noise"""
        N = adj_matrix.shape[0]
        phase_matrix = np.zeros([n_qubits, n_qubits])

        # Uniform graph fallback
        if np.all(adj_matrix == adj_matrix[0, 0]):
            return np.full((n_qubits, n_qubits), base_phase) * (
                1 + noise * np.random.randn(n_qubits, n_qubits)
            )

        for i in range(N):
            for j in range(N):
                if i != j:
                    # Handle disconnected nodes
                    weight = adj_matrix[i, j] if adj_matrix[i, j] != 0 else base_phase
                    i_bin = format(i, f"0{n_qubits}b")
                    j_bin = format(j, f"0{n_qubits}b")
                    for control in range(n_qubits):
                        for target in range(n_qubits):
                            if (
                                control != target
                                and i_bin[control] == "1"
                                and j_bin[target] == "1"
                            ):
                                phase_matrix[control, target] += weight / N

        # Add noise to break symmetry
        phase_matrix += noise * np.random.randn(*phase_matrix.shape)
        return phase_matrix

    def init_qft_phases_from_connections(self):
        """Initialize QFT phases based on qubit connections"""
        n = self.n_qubits
        phases = np.zeros([self.n_layers, self.qft_params_per_layer])

        for layer in range(self.n_layers):
            phase_idx = 0
            for target in range(n):
                for control in range(target + 1, n):
                    if (
                        self.graph_adj is None
                        or self.qubit_connections[control, target] != 0
                    ):
                        phases[layer][phase_idx] = (
                            np.random.rand() * 0.5
                            + 0.5 * self.qubit_connections[control, target]
                        )
                        phase_idx += 1

        return phases

    def rotation_layer(self, thetas):
        """Parameterized rotation layer with entangling gates"""
        param_idx = 0
        n = self.n_qubits
        # First RY layer
        for i in range(n):
            qml.RY(thetas[param_idx], wires=i)
            param_idx += 1

        for target in range(n):
            for control in range(target + 1, n):
                # Apply controlled phase if qubits are connected in graph
                if (
                    self.graph_adj is None
                    or self.qubit_connections[control, target] != 0
                ):
                    qml.CRY(thetas[param_idx], wires=[control, target])
                    param_idx += 1

    def parameterized_qft(self, phases):
        """Connection-adapted QFT circuit"""
        n = self.n_qubits
        phase_idx = 0

        for target in range(n):
            qml.Hadamard(wires=target)
            for control in range(target + 1, n):
                # Apply controlled phase if qubits are connected in graph
                if (
                    self.graph_adj is None
                    or self.qubit_connections[control, target] != 0
                ):
                    qml.CRZ(phases[phase_idx], wires=[control, target])
                    phase_idx += 1

    def circuit(self):
        """Quantum circuit definition"""
        # Apply layered operations
        for layer in range(self.n_layers):
            self.rotation_layer(self.ry_thetas[layer])
            self.parameterized_qft(self.qft_phases[layer])

        return qml.state()


def train_circuit(model, laplacian, epochs=200, lr=0.01, patience=20):
    """Train the quantum circuit to approximate Laplacian eigenvectors"""

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=20, factor=0.1, min_lr=1e-7
    )

    # Convert Laplacian to tensor
    L = torch.tensor(laplacian, dtype=torch.complex128)

    # Store best parameters
    best_loss = float("inf")
    best_params = None
    epochs_no_improve = 0

    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Get circuit unitary
        U = qml.matrix(model.qnode)()

        # Compute U^dagger L U
        transformed_L = U.conj().T @ L @ U  # not efficient for large matrices

        # Calculate loss (off-diagonal minimization)
        diag = torch.diag(transformed_L)
        off_diag = transformed_L - torch.diag(diag)
        loss = torch.norm(off_diag, p="fro") ** 2

        # Backpropagation
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        losses.append(loss.item())

        # Save best parameters
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_params = [p.detach().clone() for p in model.parameters()]
            if best_loss < 1e-6:
                break
        #     epochs_no_improve = 0
        # else:
        #     epochs_no_improve += 1
        #     if epochs_no_improve == patience:
        #         print(f"Early stopping at epoch {epoch}")
        #         break
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.6f}")

    # Restore best parameters
    with torch.no_grad():
        for param, best in zip(model.parameters(), best_params):
            param.copy_(best)

    return losses


def evaluate_performance(model, laplacian):
    """Evaluate circuit performance against true eigenvectors"""
    # Get circuit unitary
    with torch.no_grad():
        U_circuit = qml.matrix(model.qnode)().numpy()

    # Compute true eigenvectors
    eigvals, U_true = eigh(laplacian)

    # Calculate subspace overlaps
    overlaps = []
    for i in range(U_true.shape[1]):
        max_overlap = 0
        for j in range(U_circuit.shape[1]):
            overlap = np.abs(np.vdot(U_true[:, i], U_circuit[:, j]))
            if overlap > max_overlap:
                max_overlap = overlap
        overlaps.append(max_overlap**2)  # Squared overlap (probability)

    # Calculate average and minimum overlap
    avg_overlap = np.mean(overlaps)
    min_overlap = np.min(overlaps)

    print(f"\nEigenvector Approximation Quality:")
    print(f"Average Subspace Overlap: {avg_overlap:.4f}")
    print(f"Minimum Subspace Overlap: {min_overlap:.4f}")

    # Plot overlaps
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(overlaps)), overlaps)
    plt.axhline(
        avg_overlap, color="r", linestyle="--", label=f"Average: {avg_overlap:.4f}"
    )
    plt.xlabel("Eigenvector Index")
    plt.ylabel("Squared Overlap")
    plt.title("Eigenvector Approximation Quality")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()
    print("true eigenvalues", np.sort(eigvals).round(2))
    print(
        "found eigenvalues",
        np.sort(np.diag(U_circuit.conjugate().T @ laplacian @ U_circuit)).round(2),
    )
    return avg_overlap, min_overlap


def visualize_circuit(model):
    """Visualize the quantum circuit"""
    fig, ax = qml.draw_mpl(model.qnode, show_all_wires=True)()
    # plt.title("Connection-Adapted and Rotations-Added Parameterized-QFT Circuit")
    plt.tight_layout()
    plt.show()


# Graph utilities
# GRAPH LAPLACIAN with REGULARIZATION
def build_laplacian_matrix(adj_matrix, normalized=True, reg=1e-5):
    # Add self-loops to prevent isolated nodes
    adj_matrix_reg = adj_matrix + np.eye(adj_matrix.shape[0]) * reg

    degrees = np.sum(adj_matrix_reg, axis=1)
    if normalized:
        D_sqrt_inv = np.diag(1.0 / np.sqrt(degrees))
        return np.eye(adj_matrix.shape[0]) - D_sqrt_inv @ adj_matrix_reg @ D_sqrt_inv
    return np.diag(degrees) - adj_matrix_reg


def generate_random_graph(n_nodes, edge_prob=0.3):
    G = nx.erdos_renyi_graph(n_nodes, edge_prob)
    adj = nx.adjacency_matrix(G).toarray().astype(float)
    return adj - np.diag(np.diag(adj))  # Remove self-loops


def main():
    # Configuration
    n_qubits = 3  # Nodes = 2^n_qubits (for 3, it is 8 nodes)
    n_layers = 5  # 5 layers for 3 qubit, more for larger qubits
    epochs = 5000
    learning_rate = 0.001  # 0.001 seems to work well for any number of qubits

    # Generate graph and Laplacian
    n_nodes = 2**n_qubits

    adj_matrix = generate_random_graph(n_nodes, edge_prob=0.1)
    laplacian = build_laplacian_matrix(adj_matrix, normalized=True)

    print(f"Graph with {n_nodes} nodes, {np.sum(adj_matrix > 0)//2} edges")
    print(f"Laplacian eigenvalues: {np.linalg.eigvalsh(laplacian)[:5]}...")

    # Initialize quantum circuit
    model = QuantumCircuit(n_qubits, adj_matrix, n_layers=n_layers)

    # Train the circuit
    print("\nStarting training...")
    losses = train_circuit(model, laplacian, epochs, learning_rate)

    # Plot training loss
    plt.plot(losses)
    plt.title("Training Loss (Off-diagonal Minimization)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.yscale("log")
    plt.show()

    # Evaluate performance
    avg_overlap, min_overlap = evaluate_performance(model, laplacian)

    # Visualize circuit
    visualize_circuit(model)


def multi_main(
    # Configuration
    n_qubits=3,  # Nodes = 2^n_qubits (for 3, it is 8 nodes)
    n_layers=5,  # 5 layers for 3 qubit, more for larger qubits
    epochs=1000,
    learning_rate=0.001,  # 0.001 seems to work well for  small, 0.01 for larger qubits
    nruns=10,
    fname="multi_main_out",
    rand_graph_edge_prob=0.5,
):

    losses = np.zeros([nruns, epochs])
    for run in range(nruns):
        print(f"\nRun {run+1} of {nruns}")
        # Generate graph and Laplacian
        n_nodes = 2**n_qubits

        adj_matrix = generate_random_graph(n_nodes, edge_prob=rand_graph_edge_prob)
        laplacian = build_laplacian_matrix(adj_matrix, normalized=True)

        print(f"Graph with {n_nodes} nodes, {np.sum(adj_matrix > 0)//2} edges")
        print(f"Laplacian eigenvalues: {np.linalg.eigvalsh(laplacian)[:5]}...")

        # Initialize quantum circuit
        model = QuantumCircuit(n_qubits, adj_matrix, n_layers=n_layers)

        # Train the circuit
        print("\nStarting training...")
        result = train_circuit(model, laplacian, epochs, learning_rate)
        losses[run][: len(result)] = result
    plt.figure(figsize=(12, 8))

    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 24

    plt.rc("font", size=BIGGER_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

    # Plot training loss
    plt.plot(losses.T, alpha=0.9, linewidth=2)
    # Plot training loss
    plt.title(
        "Training loss (off-diagonal minimization)\n"
        f"{n_qubits}-qubit and {n_layers}-layer circuit"
        f" for graphs with edge probability {rand_graph_edge_prob}"
    )
    plt.xlim(0, epochs)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.4)
    plt.yscale("log")
    ax = plt.gca()
    ax.set_xlim([0.0, epochs])
    ax.set_ylim([0.01, 3])
    plt.plot(
        np.mean(losses, 0), color="black", ls="--", linewidth="4", alpha=1, label="mean"
    )
    plt.tight_layout()
    plt.savefig(f"{fname}.pdf", bbox_inches="tight")
    plt.savefig(f"{fname}.png", bbox_inches="tight")
    plt.show()

    return losses


# if __name__ == "__main__":
# Set random seed for reproducibility

losses = {}
for q in range(5, 6):
    torch.manual_seed(42)
    np.random.seed(42)
    losses[q] = multi_main(
        n_qubits=q,
        fname=f"{q}",
        n_layers=30,
        rand_graph_edge_prob=0.3,
        learning_rate=0.01,
        nruns=10,
        epochs=2000,
    )
# import cProfile, pstats, io
# from pstats import SortKey
# pr = cProfile.Profile()
# pr.enable()
# main()
# pr.disable()
# s = io.StringIO()
# sortby = SortKey.CUMULATIVE
# ps = pstats.Stats(pr, stream=s).sort_stats("time")
# ps.print_stats("parameterized_qft.py")
# print(s.getvalue())
