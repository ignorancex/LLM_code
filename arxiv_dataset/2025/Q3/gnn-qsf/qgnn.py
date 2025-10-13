import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch_geometric.datasets import TUDataset, MNISTSuperpixels, MoleculeNet
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_adj
from sklearn.model_selection import StratifiedKFold, train_test_split
import os

os.environ["OMP_NUM_THREADS"] = "4"  # limit OpenMP threads for Pennylane
import pennylane as qml
import warnings


# ========================
# Quantum Circuit Module
# ========================
class QuantumCircuit(nn.Module):
    def __init__(self, n_qubits, n_layers=1):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.state_dim = 2**n_qubits

        # PennyLane device
        self.dev = qml.device("lightning.qubit", wires=n_qubits, batch_obs=True)

        # count of rotation & QFT params per layer
        n_pairs = n_qubits * (n_qubits - 1) // 2
        self.rot_params_per_layer = n_qubits + n_pairs
        self.qft_params_per_layer = n_pairs

        # trainable parameters (float32 by default)
        self.ry_thetas = nn.Parameter(torch.rand(n_layers, self.rot_params_per_layer))
        self.qft_phases = nn.Parameter(torch.rand(n_layers, self.qft_params_per_layer))

        # Torch‐backed QNode
        self.qnode = qml.QNode(
            self.circuit, self.dev, interface="torch", diff_method="adjoint"
        )

    def rotation_layer(self, thetas, qubit_connections):
        idx = 0
        for i in range(self.n_qubits):
            qml.RY(thetas[idx], wires=i)
            idx += 1
        for c in range(self.n_qubits):
            for t in range(c + 1, self.n_qubits):
                if qubit_connections[c, t] != 0:
                    angle = thetas[idx]
                    qml.CRY(angle, wires=[c, t])
                idx += 1

    def parameterized_qft(self, phases, qubit_connections):
        pidx = 0
        for tgt in range(self.n_qubits):
            qml.Hadamard(wires=tgt)
            for ctrl in range(tgt + 1, self.n_qubits):
                if qubit_connections[ctrl, tgt] != 0:
                    angle = phases[pidx] * 0.9 + 0.1 * qubit_connections[ctrl, tgt]
                    qml.CRZ(angle, wires=[ctrl, tgt])
                pidx += 1

    def circuit(self, input_state, qubit_connections):
        # amplitude embedding of a 2**n_qubits vector
        norm = torch.norm(input_state)
        if torch.isclose(norm, torch.tensor(0.0)):
            warnings.warn("⚠️ Zero‐vector embedding: input to quantum circuit is 0!")

            qml.AmplitudeEmbedding(
                input_state + 1e-8, wires=range(self.n_qubits), normalize=True
            )
        else:
            qml.AmplitudeEmbedding(
                input_state, wires=range(self.n_qubits), normalize=True
            )

        for layer in range(self.n_layers):
            self.rotation_layer(self.ry_thetas[layer], qubit_connections)
            self.parameterized_qft(self.qft_phases[layer], qubit_connections)

        # measure PauliZ on each qubit
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, input_state, qubit_connections):
        # print(input_state)
        raw_output = self.qnode(input_state, qubit_connections)
        # Normalize and clip outputs to prevent explosion
        output = torch.stack(raw_output).float()  # Ensure float32
        # print(output)
        # output = torch.clamp(output, -1.0, 1.0)  # Constrain to [-1,1]
        return output  #


# ========================
# Quantum GNN Module
# ========================
class QuantumGNN(nn.Module):
    def __init__(
        self,
        n_qubits,
        n_layers=1,
        hidden_dims=[64, 32],
        output_dim=1,
        dropout_prob=0.25,
    ):

        super().__init__()
        self.n_qubits = n_qubits

        # quantum backbone
        self.quantum_circuit = QuantumCircuit(n_qubits, n_layers)

        # classical head takes exactly n_qubits inputs
        layers = []
        input_dim = n_qubits
        for h in hidden_dims:
            layers += [
                nn.Linear(input_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
            ]
            input_dim = h

        # final classifier
        layers.append(nn.Linear(input_dim, output_dim))
        self.classical_net = nn.Sequential(*layers)

    def forward(self, features):
        padded_features, qubit_connections = (
            features  # shapes: [B, 2**n_qubits], [B, n_qubits, n_qubits]
        )
        batch_size = padded_features.size(0)

        # run each graph through the QNode
        quantum_outputs = []
        for i in range(batch_size):
            oq = self.quantum_circuit(padded_features[i], qubit_connections[i])
            # stack the n_qubit expectation values → [n_qubits]
            # quantum_outputs.append(torch.stack(oq))
            quantum_outputs.append(oq)

        # → [B, n_qubits] but ensure float32
        probs = torch.stack(quantum_outputs, dim=0).float()

        # feed classical head → logits
        return self.classical_net(probs)


# ========================
# Data Preprocessing
# ========================
class GraphPreprocessor:
    def __init__(self, dataset):
        self.dataset = dataset
        self.max_nodes = max(d.num_nodes for d in dataset)
        self.feature_dim = dataset.num_node_features
        if self.feature_dim == 0:
            self.feature_dim = 1
        self.n_qubits = math.ceil(math.log2(self.max_nodes * self.feature_dim))
        self.state_dim = 2**self.n_qubits
        self.bin_repr = self.create_bin_repr()  # [max_nodes, n_qubits] float32

    def create_bin_repr(self):
        nodes = torch.arange(self.max_nodes)
        # bit-shift into n_qubits columns, then float32
        br = (nodes.unsqueeze(1) >> torch.arange(self.n_qubits - 1, -1, -1)) & 1
        return br.float()

    # def compute_qubit_connections(self, adj):
    #     # adj: [max_nodes, max_nodes] float32, no self–loops
    #     a = adj.clone() / adj.shape[0]  # normalize by node count

    #     a.fill_diagonal_(0)
    #     # → [n_qubits, n_qubits]
    #     return torch.einsum("ij,ic,jt->ct", a, self.bin_repr, self.bin_repr)

    # MORE ROBUST PHASE INITIALIZATION
    def compute_qubit_connections(
        self, adj_matrix, n_qubits=None, base_phase=0.1, noise=0.01
    ):
        N = adj_matrix.shape[0]
        if n_qubits is None:
            n_qubits = self.n_qubits

        # Use precomputed binary representation
        node_bin = self.bin_repr[:N].to(adj_matrix.device)  # Only use first N nodes

        # Create off-diagonal mask
        off_diag_mask = ~torch.eye(N, dtype=torch.bool, device=adj_matrix.device)

        # Prepare weight matrix
        W = adj_matrix / N
        zero_off_diag = (W == 0) & off_diag_mask
        W = torch.where(zero_off_diag, base_phase, W)
        W = W * off_diag_mask.float()  # Zero out diagonal

        # Compute phase matrix using vectorized operations
        phase_matrix = node_bin.t() @ (W @ node_bin)

        # Add noise to break symmetry
        phase_matrix += noise * torch.randn_like(phase_matrix)

        return phase_matrix

    def preprocess_graph(self, data):
        # flatten & pad
        # 1) detect missing or zero-width features
        if (
            (not hasattr(data, "x"))
            or data.x == None
            or data.x.numel() == 0
            or data.x.size(1) == 0
        ):

            # ensure you modify the same Data object
            warnings.warn("⚠️ No node features found, using ones.")
            data.x = torch.ones(self.state_dim, 1, dtype=torch.float32)

        # 2) flatten and pad
        flat = data.x.reshape(-1).float()  # length = num_nodes*feat_dim

        # Check for NaN/Inf in features
        if torch.isnan(flat).any() or torch.isinf(flat).any():
            flat = torch.nan_to_num(flat, nan=0.0, posinf=1.0, neginf=-1.0)

        if flat.size(0) != self.state_dim:
            padded = torch.zeros(self.state_dim, dtype=torch.float32)
            L = min(flat.size(0), self.state_dim)
            padded[:L] = flat[:L]
        else:
            padded = flat
        # adjacency → dense, pad, float32..
        # if hasattr(data, "edge_attr") and data.edge_attr is not None:
        #     warnings.warn(
        #         "⚠️ Edge attributes found,  but not used in QNN. Adj can be taken as W,"
        #     )
        # adj = to_dense_adj(data.edge_index,edge_attr= data.edge_attr, max_num_nodes=data.num_nodes)[0].float()

        adj = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0].float()
        padA = torch.zeros(self.max_nodes, self.max_nodes, dtype=torch.float32)
        padA[: data.num_nodes, : data.num_nodes] = adj

        pm = self.compute_qubit_connections(padA)

        # label → long for CrossEntropy
        label = data.y.long().squeeze()
        return padded, pm, label

    def preprocess_dataset(self):
        triplets = [self.preprocess_graph(d) for d in self.dataset]
        feats, phases, labs = zip(*triplets)
        return torch.stack(feats), torch.stack(phases), torch.stack(labs)

    # Add method to process subsets
    def preprocess_subset(self, subset):
        triplets = [self.preprocess_graph(d) for d in subset]
        feats, phases, labs = zip(*triplets)
        return torch.stack(feats), torch.stack(phases), torch.stack(labs)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for feats, phases, labs in loader:
            feats, phases, labs = feats.to(device), phases.to(device), labs.to(device)
            logits = model((feats, phases))
            preds = logits.argmax(dim=1)
            correct += (preds == labs).sum().item()
            total += labs.size(0)
    return correct / total


def train_and_eval_fold(
    train_loader,
    val_loader,
    test_loader,
    n_qubits,
    num_classes,
    dataset_name,
    n_layers,
    hidden_dims,
    lr,
    epochs,
    seed,
    dropout_prob=0.25,
    weight_decay=1e-5,
    patience=15,
):
    # reproducibility & device
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model, loss, optimizer, scheduler
    model = QuantumGNN(
        n_qubits=n_qubits,
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        output_dim=num_classes,
        dropout_prob=dropout_prob,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    print(f"Using {device} for training")
    print(f"Model: {model}")
    print(
        f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )
    print(f"Optimizer: {optimizer}")
    print(f"Loss Function: {criterion}")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=patience, factor=0.1, min_lr=1e-7
    )

    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_val_epoch = 0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        # ——— train ———
        model.train()
        total_loss = 0.0
        total_samples = 0
        for feats, phases, labs in train_loader:
            feats, phases, labs = (
                feats.to(device),
                phases.to(device),
                labs.to(device),
            )
            optimizer.zero_grad()
            logits = model((feats, phases))
            loss = criterion(logits, labs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * labs.size(0)
            total_samples += labs.size(0)

        train_loss = total_loss / total_samples

        # ——— validate ———
        val_acc = evaluate(model, val_loader, device)
        scheduler.step(val_acc)

        # ——— early stopping bookkeeping ———
        if (val_acc > best_val_acc) or (
            val_acc == best_val_acc and train_loss < best_val_loss
        ):
            best_val_loss = train_loss
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stop @ epoch {epoch} | Best val_acc={best_val_acc:.4f}")
                break

        # optional progress print
        # if epoch % 10 == 0 or epoch == 1:
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"[{dataset_name}] Epoch {epoch:03d} | "
            f"Train Loss {train_loss:.4f} | Val Acc {val_acc:.4f} | LR {lr_now:.1e}"
            f" [Best Val Acc: {best_val_acc:.4f} @ Epoch {best_val_epoch}]"
        )

    # ——— test ———
    # restore best weights
    model.load_state_dict(best_state)
    test_acc = evaluate(model, test_loader, device)
    print(f"[{dataset_name}] Fold Test Acc: {test_acc:.4f}\n")
    return test_acc


# ========================
# Cross-Validation Function
# ========================

# assume GraphPreprocessor, QuantumGNN, cross_validate_model,
# train_and_eval_fold, evaluate are already defined/imported above


def cross_validate_model(
    dataset_name="MUTAG",
    n_splits=10,
    val_ratio=0.1,
    n_layers=2,
    hidden_dims=[32, 16],
    batch_size=16,
    lr=1e-3,
    epochs=20,
    seed=42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    # if dataset_name == "MNISTSuperpixels":
    #     dataset = MNISTSuperpixels(root="data/MNISTSuperpixels", transform=None)
    # elif dataset_name in ("ESOL", "FreeSolv", "Lipo", "PCBA", "MUV",
    #                       "HIV", "BACE", "BBBP", "Tox21", "ToxCast",
    #                       "SIDER", "ClinTox"):
    #     dataset = MoleculeNet(root="data/MoleculeNet", name=dataset_name)
    # else:
    #
    dataset = TUDataset(
        root="data/TUDataset",
        name=dataset_name,
        use_node_attr=True,  # use_edge_attr=False,  # edge attributes not used in this implementation
    )
    # get a numpy array of all graph labels
    labels = dataset.y.cpu().numpy()  # → shape: [num_graphs]

    # build an index array to split on
    idx = np.arange(len(dataset))  # → [0,1,2,…,num_graphs-1]

    skf_outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_accs = []

    for fold, (train_val_idx, test_idx) in enumerate(
        skf_outer.split(idx, labels), start=1
    ):
        # carve out small validation set from train_val
        tv_labels = labels[train_val_idx]
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=int(len(train_val_idx) * val_ratio),
            stratify=tv_labels,
            random_state=seed,
        )

        # build subsets
        train_ds = dataset[train_idx.tolist()]
        val_ds = dataset[val_idx.tolist()]
        test_ds = dataset[test_idx.tolist()]

        # preprocess graphs
        pre = GraphPreprocessor(dataset)
        print(f"number of qubits {pre.n_qubits}")
        print(f"number of nodes {pre.max_nodes}")
        print(f"number of features {pre.feature_dim}")
        train_feats, train_phases, train_labs = pre.preprocess_subset(train_ds)
        val_feats, val_phases, val_labs = pre.preprocess_subset(val_ds)
        test_feats, test_phases, test_labs = pre.preprocess_subset(test_ds)
        print(f"train_feats shape: {train_feats.shape}")
        print(f"train_phases shape: {train_phases.shape}")
        print(f"train_labs shape: {train_labs.shape}")
        # data loaders
        train_loader = DataLoader(
            TensorDataset(train_feats, train_phases, train_labs),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(val_feats, val_phases, val_labs),
            batch_size=batch_size,
            shuffle=False,
        )
        test_loader = DataLoader(
            TensorDataset(test_feats, test_phases, test_labs),
            batch_size=batch_size,
            shuffle=False,
        )

        # train & evaluate this fold
        test_acc = train_and_eval_fold(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            n_qubits=pre.n_qubits,
            num_classes=dataset.num_classes,
            dataset_name=dataset_name,
            n_layers=n_layers,
            hidden_dims=hidden_dims,
            lr=lr,
            epochs=epochs,
            seed=seed,
        )
        fold_accs.append(test_acc)

    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    print(f"\n=== Cross-Validation Results for {dataset_name} ===")
    print(f"Per-fold Accuracies: {fold_accs}")
    print(f"Mean Test Acc: {mean_acc:.4f} ± {std_acc:.4f}")
    return fold_accs


if __name__ == "__main__":
    # Configuration
    config = {
        "dataset_name": "Letter-low",  # Choose from MUTAG, COX2, DHFR, PTC_MR, PTC_FM
        "n_layers": 1,  # Number of quantum layers
        "hidden_dims": [32, 16],  # Hidden dimensions for classical head
        # "hidden_dims": [32,64],  # Hidden dimensions for classical head
        # "hidden_dims": [256, 128],  # Hidden dimensions for classical head
        "batch_size": 32,  # Batch size for training
        "lr": 0.01,  # Learning rate for optimizer
        "n_splits": 10,  # Number of cross-validation splits
        "val_ratio": 0.1,  # Validation set ratio
        "epochs": 50,  # Number of training epochs
        "seed": 42,  # Random seed
    }
    print(config)
    # Run CV
    fold_accuracies = cross_validate_model(**config)
