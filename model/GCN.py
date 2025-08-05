
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F

class GNNModel(nn.Module):
    """
    Graph Neural Network for node-level classification based on repeated message passing.

    Architecture Overview:
        - Input projection: one-hot encoding of node types followed by a linear projection to an embedding space.
        - Message passing: repeated application of a single shared GCNConv layer with ReLU activations.
        - Output decoding: a final GCNConv layer produces per-node class logits.

    This model is suitable for node classification tasks where each node is labeled with a class from a finite alphabet.
    The input node features are assumed to be discrete integers representing node types.
    """

    def __init__(self,size_alphabet,embed_dim, num_hidden):
        """
        Initializes the GNNModel.

        Parameters:
            embed_dim (int): Dimension of the embedding space after input projection.
            num_hidden (int): Number of hidden units in the internal GCN layer.
            size_alphabet (int): Size of the alphabet of the subshift.
            num_iterations (int): Number of times the shared GCN layer is applied for message passing.
        """
        super().__init__()

        self.input_proj = nn.Linear(size_alphabet+1, embed_dim, bias=False)
        self.shared_conv = GCNConv(embed_dim, num_hidden)
        self.decoder = GCNConv(num_hidden, size_alphabet+1)

        self.num_node_types = size_alphabet+1

    # def forward(self, data):
    #     """
    #     Forward pass of the GNNModel.

    #     Parameters:
    #         data (torch_geometric.data.Data): Graph data object containing:
    #             - x: LongTensor of shape (num_nodes, 1) with integer-encoded node types.
    #             - edge_index: LongTensor of shape (2, num_edges) representing the graph connectivity.

    #     Returns:
    #         torch.Tensor: Tensor of shape (num_nodes, num_classes), where each row contains
    #                       the class logits for a node.
    #     """
    #     x, edge_index = data.x, data.edge_index

    #     x = F.one_hot(x.squeeze().long(), num_classes=self.num_node_types).float()
    #     x = self.input_proj(x)  # Shape: (num_nodes, embed_dim)

    #     for _ in range(self.num_iterations):
    #         x = F.relu(self.shared_conv(x, edge_index))

    #     x = self.decoder(x, edge_index)  # Shape: (num_nodes, num_classes)

    #     return x

    def forward(self, data, num_iterations):
        """
        Forward pass of the GNN model.

        Parameters:
            data (torch_geometric.data.Data): Batch of graph data.

        Returns:
            torch.Tensor: Output tensor of shape (num_nodes, 1) with predicted values.
        """
        # TODO: use -100 on ground truth to ignore the correspondinglabels.
        # TODO: ultimately, use random values instead of * or -1 ? 
        x, edge_index = data.x, data.edge_index
        
        x = F.one_hot(x.squeeze().long(), num_classes=self.num_node_types).float()
        x = self.input_proj(x)  # Now shape (num_nodes, embed_dim)

        # Repeated application of same convolutional layer
        for _ in range(num_iterations):
            x = F.relu(self.shared_conv(x, edge_index))

        # Final decoder layer
        x = self.decoder(x, edge_index)  # logits
        return x
