"""
GNN training pipeline for reconstructing masked symbolic patterns from subshifts of finite type (SFT).

This script loads string-encoded 2D symbolic patterns (e.g. '0', '1', '*'), converts them to numeric arrays, 
constructs a grid graph over the 2D positions, and trains a GCN (Graph Convolutional Network) to reconstruct
the full pattern from the masked version.

The model is trained using PyTorch Geometric, and saved to disk upon completion.
"""
# TODO: eval is exactly zero: after inspection, it seems that the model tends to complete uniformly (tested on box 19 x 19 with subbox of 7 x 7). Why is that so ? Is it due to positive entropy ? 

# TODO: later, one possibility is to evaluate the model not on finding a correct answer but on minimizing the number of forbidden patterns in the output. 

# TODO: enrich the dataset by creating pattern that break the rules, patterns which do not extend, etc ? 

# In all tests so far (19x19 and 7x7, 5x5 and 3x3), the loss stabilises around 0.2 or 0.3. For small problems, like box 3x3 and subbox 1x1, non-zero precision (approx 1/3 of success). 
# Can this be used to ameliorate training (apply prediction one by one).

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import GCNConv
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
from utils import directional_round
from eval import eval_subbox_to_outside

def convert_string_patterns_to_float(array, mask_symbol="*", mask_value=-1):
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

def make_dataset(complement, masked):
    """
    Constructs a dataset of PyTorch Geometric Data objects from complement and masked pattern arrays.

    Parameters:
        complement (np.ndarray): Complement pattern array of shape (H, W, N), where N is the number of patterns.
        masked (np.ndarray): Masked pattern array of same shape, with masked values replaced numerically.

    Returns:
        List[torch_geometric.data.Data]: List of graph data objects with fields (x, y, edge_index).
    """
    N,H, W = complement.shape
    edge_index = create_grid_edges(H, W)
    dataset = []
    for i in range(N):
        x = torch.tensor(masked[i,:, :].flatten(), dtype=torch.float).unsqueeze(1)
        y = torch.tensor(complement[i,:, :].flatten(), dtype=torch.float).unsqueeze(1)
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
        # TODO: output dimension should be the number of labels
        # TODO: use crossentropy for the loss.
        # TODO: use -100 on ground truth to ignore the correspondinglabels.
        # TODO: ultimately, use random values instead of * or -1. 

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
        # TODO: 1 convo layer and for loop -> better: higher number of layers. 
        # Better generalization
        # Other idea: 1 conv layer to encode input + for loops with same layer (same parameters for each copy of the layer) + conv layer for decoding.
        # For first layer: embedding layer (int -> vector, one hot encoding + LinearLayer, or EmbeddingLayer).
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
    # TODO: here change to crossentropy loss

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

def run_training(complement_patterns, masked_patterns, model_path, batch_size=10, epochs=20,test_ratio=0.1):
    """
    Main training pipeline: prepares dataset, trains the model, and saves it to disk.

    Parameters:
        complement_patterns (np.ndarray): Full pattern data of shape (H, W, N) with float entries.
        masked_patterns (np.ndarray): Masked version of the patterns, same shape.
        model_path (str): Path where the trained model will be saved.
        batch_size (int): Number of patterns per training batch.
        epochs (int): Number of training epochs.

    Returns:
        GNNModel: The trained model instance.
    """
    dataset = make_dataset(complement_patterns, masked_patterns)
    train_set, test_set = train_test_split(dataset, test_size=test_ratio, random_state=42)
    train_loader = GeoDataLoader(train_set, batch_size=batch_size)

    model = GNNModel()
    train(model, train_loader, epochs=epochs)

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    return model, test_set

# def evaluate(model, test_set):
#     """
#     Evaluates the GNN model on a test set using mean squared error.

#     Parameters:
#         model (nn.Module): The trained GNN model.
#         test_set (List[Data]): List of test graphs.

#     Returns:
#         float: Average MSE loss over the test set.
#     """

#     model.eval()
#     loader = GeoDataLoader(test_set, batch_size=10)
#     loss_fn = nn.MSELoss()
#     total_loss = 0.0
#     count = 0

#     with torch.no_grad():
#         for batch in loader:
#             pred = model(batch)
#             loss = loss_fn(pred, batch.y)
#             total_loss += loss.item() * batch.num_graphs
#             count += batch.num_graphs

#     avg_loss = total_loss / count
#     print(f"Test MSE: {avg_loss:.4f}")
#     return avg_loss

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
    print("\n Input (x):")
    print(input_masked)

    print("\n Ground truth (y):")
    print(ground_truth)

    print("\n Model prediction:")
    print(directional_round(predicted,decimals=0))

if __name__ == "__main__":
    # TODO: decide which option to take: predict complete pattern or the complement pattern.

    NAME = "subshift_2"
    H, W = 3,3
    BOX_SIZE = 1
    
    # Load and preprocess pattern data
    complement_patterns = np.load(f"subbox_masked_patterns/{NAME}/all_patterns.npy")
    complement_patterns = convert_string_patterns_to_float(complement_patterns)

    masked_patterns = np.load(f"outside_subbox_masked_patterns/{NAME}/all_patterns.npy")
    masked_patterns = convert_string_patterns_to_float(masked_patterns)

    # Train and save the model
    model,test_set = run_training(
        complement_patterns,
        masked_patterns,
        model_path="models/gnn.pt",
        batch_size=10,
        epochs=20
    )

    # Pick one pattern (e.g. the first)
    sample = test_set[0]

    # Run inspection
    inspect_prediction(model, sample, H, W)

    loader = GeoDataLoader(test_set, batch_size=10)
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            preds = model(batch)
            preds = preds.squeeze().cpu().numpy()
            B = batch.num_graphs
            preds = preds.reshape(B, H, W)
            preds = directional_round(preds, decimals=0).astype(int)
            all_preds.append(preds)

    predicted_stack = np.concatenate(all_preds, axis=0)
    input_stack = np.stack([
    data.x.squeeze().cpu().numpy().reshape(H, W)
    for data in test_set
        ])
    with open("samples.json", "r") as f:
        samples = json.load(f)
    eval = eval_subbox_to_outside(input_stack,predicted_stack,box_size=BOX_SIZE,forbidden_patterns=samples[NAME]["forbidden_pairs"])
    print(eval)