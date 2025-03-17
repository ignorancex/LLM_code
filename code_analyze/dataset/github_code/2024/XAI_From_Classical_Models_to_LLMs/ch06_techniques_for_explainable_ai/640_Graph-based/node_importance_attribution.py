import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

# Load the Cora dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# Define a simple GCN model
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# Train the GCN model
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

# Calculate node importance
model.eval()
data.x.requires_grad = True  # Enable gradient calculation for node features

target_node = 10
output = model(data.x, data.edge_index)

# Perform backward pass for the predicted class of the target node
predicted_class = output[target_node].argmax()
output[target_node, predicted_class].backward()

# Calculate the L2 norm of the gradient for the target node's features as the importance score
node_importance = torch.norm(data.x.grad[target_node], p=2).item()
print(f"Importance score for node {target_node}: {node_importance:.4f}")
