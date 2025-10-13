import spektral
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout, Flatten, Dense
from spektral.layers import GCNConv
from spektral.utils import normalized_laplacian




class GCN(Model):
    def __init__(self, num_classes):
        super(GCN, self).__init__()
        self.graph_conv1 = GCNConv(32, activation='relu')
        self.dropout = Dropout(0.5)
        self.graph_conv2 = GCNConv(32, activation='relu')
        self.fc = Dense(num_classes, activation='softmax')

    def call(self, inputs):
        adjacency_matrix, node_features = inputs
        x = self.dropout(node_features)
        x = self.graph_conv1([node_features, adjacency_matrix])
        x = self.dropout(x)
        x = self.graph_conv2([x, adjacency_matrix])
        x = Flatten()(x)  # Flatten the output
        x = self.fc(x)
        return x


