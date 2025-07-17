"""
GNN training pipeline for reconstructing masked symbolic patterns from subshifts of finite type (SFT).

This script loads string-encoded 2D symbolic patterns (e.g. '0', '1', '*'), converts them to numeric arrays, 
constructs a grid graph over the 2D positions, and trains a GCN (Graph Convolutional Network) to reconstruct
the full pattern from the masked version.

The model is trained using PyTorch Geometric, and saved to disk upon completion.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import GCNConv
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split

def convert_string_patterns_to_float(array, mask_symbol="*", mask_value=-1.0):
    """
    Converts a NumPy array of strings ('0', '1', '*') to float values.

    Parameters:
        array (np.ndarray): Input array of dtype string (e.g., '<U1') representing symbolic patterns.
        mask_symbol (str): The symbol representing a masked value (default is '*').
        mask_value (float): The numeric value to substitute for the masked symbol (default is -1.0).

    Returns:
        np.ndarray: Array of dtype float32 with symbolic entries converted to floats.
    """
    vectorized = np.vectorize(lambda x: mask_value if x == mask_symbol else float(x))
    return vectorized(array).astype(np.float32)

def create_grid_edges(H, W):
    """
    Constructs 4-neighbor grid connectivity over a 2D grid of size H x W.

    Parameters:
        H (int): Number of rows in the grid.
        W (int): Number of columns in the grid.

    Returns:
        torch.LongTensor: Edge index tensor of shape (2, num_edges), with bidirectional edges.
    """
    G = nx.grid_2d_graph(H, W)
    mapping = {(i, j): i * W + j for i in range(H) for j in range(W)}
    G = nx.relabel_nodes(G, mapping)
    edge_list = list(G.edges())
    edge_index = torch.tensor(edge_list + [(j, i) for (i, j) in edge_list], dtype=torch.long).t().contiguous()
    return edge_index

def make_dataset(complete, masked):
    """
    Constructs a dataset of PyTorch Geometric Data objects from complete and masked pattern arrays.

    Parameters:
        complete (np.ndarray): Complete pattern array of shape (H, W, N), where N is the number of patterns.
        masked (np.ndarray): Masked pattern array of same shape, with masked values replaced numerically.

    Returns:
        List[torch_geometric.data.Data]: List of graph data objects with fields (x, y, edge_index).
    """
    N,H, W = complete.shape
    edge_index = create_grid_edges(H, W)
    dataset = []
    for i in range(N):
        x = torch.tensor(masked[i,:, :].flatten(), dtype=torch.float).unsqueeze(1)
        y = torch.tensor(complete[i,:, :].flatten(), dtype=torch.float).unsqueeze(1)
        data = Data(x=x, y=y, edge_index=edge_index)
        dataset.append(data)
    return dataset

class GNNModel(nn.Module):
    """
    Simple 2-layer Graph Convolutional Network for node-wise regression.

    Architecture:
        - GCNConv(1, hidden_dim)
        - ReLU
        - GCNConv(hidden_dim, hidden_dim)
        - ReLU
        - Linear(hidden_dim → 1)
    """
    def __init__(self, hidden_dim=64):
        """
        Initializes the GNN model.

        Parameters:
            hidden_dim (int): Number of hidden units in the GCN layers.
        """
        super().__init__()
        self.conv1 = GCNConv(1, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        """
        Forward pass of the GNN model.

        Parameters:
            data (torch_geometric.data.Data): Batch of graph data.

        Returns:
            torch.Tensor: Output tensor of shape (num_nodes, 1) with predicted values.
        """
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        return self.head(x)

def train(model, loader, epochs=20):
    """
    Trains the GNN model on a dataset of masked patterns.

    Parameters:
        model (nn.Module): The GNN model to train.
        loader (DataLoader): PyTorch Geometric DataLoader for batching training data.
        epochs (int): Number of epochs to train for.
    """
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            pred = model(batch)
            loss = loss_fn(pred, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1:02d}: Loss = {total_loss / len(loader):.4f}")

def run_training(complete_patterns, masked_patterns, model_path, batch_size=10, epochs=20,test_ratio=0.1):
    """
    Main training pipeline: prepares dataset, trains the model, and saves it to disk.

    Parameters:
        complete_patterns (np.ndarray): Full pattern data of shape (H, W, N) with float entries.
        masked_patterns (np.ndarray): Masked version of the patterns, same shape.
        model_path (str): Path where the trained model will be saved.
        batch_size (int): Number of patterns per training batch.
        epochs (int): Number of training epochs.

    Returns:
        GNNModel: The trained model instance.
    """
    dataset = make_dataset(complete_patterns, masked_patterns)
    train_set, test_set = train_test_split(dataset, test_size=test_ratio, random_state=42)
    train_loader = GeoDataLoader(train_set, batch_size=batch_size)

    model = GNNModel()
    train(model, train_loader, epochs=epochs)

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    return model, test_set

def evaluate(model, test_set):
    """
    Evaluates the GNN model on a test set using mean squared error.

    Parameters:
        model (nn.Module): The trained GNN model.
        test_set (List[Data]): List of test graphs.

    Returns:
        float: Average MSE loss over the test set.
    """

    model.eval()
    loader = GeoDataLoader(test_set, batch_size=10)
    loss_fn = nn.MSELoss()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for batch in loader:
            pred = model(batch)
            loss = loss_fn(pred, batch.y)
            total_loss += loss.item() * batch.num_graphs
            count += batch.num_graphs

    avg_loss = total_loss / count
    print(f"Test MSE: {avg_loss:.4f}")
    return avg_loss

def inspect_prediction(model, data_point, H, W):
    """
    Runs the model on a single test pattern and displays the prediction.

    Parameters:
        model (GNNModel): Trained model.
        data_point (torch_geometric.data.Data): One pattern graph.
        H (int): Height of the original 2D pattern.
        W (int): Width of the original 2D pattern.
    """
    model.eval()
    print(data_point)
    with torch.no_grad():
        pred = model(data_point)

    # Flattened versions
    predicted = pred.squeeze().cpu().numpy().reshape(H, W)
    ground_truth = data_point.y.squeeze().cpu().numpy().reshape(H, W)
    input_masked = data_point.x.squeeze().cpu().numpy().reshape(H, W)

    # Display
    print("\n🔍 Masked input (x):")
    print(input_masked)

    print("\n✅ Ground truth (y):")
    print(ground_truth)

    print("\n🔮 Model prediction:")
    print(np.round(predicted, decimals=2))

if __name__ == "__main__":
    # Load and preprocess pattern data
    complete_patterns = np.load("patterns/subshift_1/all_patterns.npy")
    complete_patterns = convert_string_patterns_to_float(complete_patterns)

    masked_patterns = np.load("outside_subbox_masked_patterns/subshift_1/all_patterns.npy")
    masked_patterns = convert_string_patterns_to_float(masked_patterns)

    # Train and save the model
    model,test_set = run_training(
        complete_patterns,
        masked_patterns,
        model_path="models/gnn.pt",
        batch_size=10,
        epochs=20
    )
    # test_loss = evaluate(model, test_set)

    # Pick one pattern (e.g. the first)
    sample = test_set[0]

    # Recover the original dimensions
    H, W = 19, 19  # set these to your actual pattern size

    # Run inspection
    inspect_prediction(model, sample, H, W)