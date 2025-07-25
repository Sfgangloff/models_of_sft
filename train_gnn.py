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
import pickle
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import GCNConv
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
from utils import directional_round
from eval import eval_subbox_to_outside, merge_patterns_stack
import torch.nn.functional as F

def get_alphabet_size(subshift_key):
    """
    Returns the size of the alphabet for a given subshift in a JSON file.

    Parameters:
        filename (str): Path to the JSON file.
        subshift_key (str): Key of the subshift (e.g., "subshift_2").

    Returns:
        int: Size of the alphabet for the specified subshift.
    """
    with open("samples.json", 'r') as f:
        data = json.load(f)

    if subshift_key not in data:
        raise KeyError(f"Subshift key '{subshift_key}' not found in file.")

    alphabet = data[subshift_key]["alphabet"]
    return len(alphabet)

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
    Graph Neural Network for node-level classification based on repeated message passing.

    Architecture Overview:
        - Input projection: one-hot encoding of node types followed by a linear projection to an embedding space.
        - Message passing: repeated application of a single shared GCNConv layer with ReLU activations.
        - Output decoding: a final GCNConv layer produces per-node class logits.

    This model is suitable for node classification tasks where each node is labeled with a class from a finite alphabet.
    The input node features are assumed to be discrete integers representing node types.
    """

    def __init__(self,size_alphabet,embed_dim, num_hidden,num_iterations):
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

        self.num_iterations = num_iterations
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

    def forward(self, data):
        """
        Forward pass of the GNN model.

        Parameters:
            data (torch_geometric.data.Data): Batch of graph data.

        Returns:
            torch.Tensor: Output tensor of shape (num_nodes, 1) with predicted values.
        """
        # TODO: use -100 on ground truth to ignore the correspondinglabels.
        # TODO: ultimately, use random values instead of * or -1. 
        x, edge_index = data.x, data.edge_index
        
        x = F.one_hot(x.squeeze().long(), num_classes=self.num_node_types).float()
        x = self.input_proj(x)  # Now shape (num_nodes, embed_dim)

        # Repeated application of same convolutional layer
        for _ in range(self.num_iterations):
            x = F.relu(self.shared_conv(x, edge_index))

        # Final decoder layer
        x = self.decoder(x, edge_index)  # logits
        return x

def train(model, loader, epochs):
    """
    Trains the GNN model on a dataset of masked patterns.

    Parameters:
        model (nn.Module): The GNN model to train.
        loader (DataLoader): PyTorch Geometric DataLoader for batching training data.
        epochs (int): Number of epochs to train for.
    """
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # Cross entropy loss, for classification task.
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            pred = model(batch)
            loss = loss_fn(pred, batch.y.squeeze(1).long())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1:02d}: Loss = {total_loss / len(loader):.4f}")

def run_training(complement_patterns, masked_patterns, model_path, size_alphabet, epochs, batch_size,test_ratio, embed_dim,num_hidden,
                     num_iterations):
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

    model = GNNModel(size_alphabet=size_alphabet,
                     embed_dim=embed_dim,
                     num_hidden=num_hidden,
                     num_iterations=num_iterations)
    train(model, train_loader, epochs=epochs)

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    test_path = model_path.replace(".pt", "_testset.pkl")
    with open(test_path, "wb") as f:
        pickle.dump(test_set, f)
    print(f"Test set saved to {test_path}")

    return model, test_set

def inspect_prediction(model, data_point, H, W,subbox_size):
    """
    Runs the model on a single test pattern and displays the prediction.

    Parameters:
        model (GNNModel): Trained model.
        data_point (torch_geometric.data.Data): One pattern graph.
        H (int): Height of the original 2D pattern.
        W (int): Width of the original 2D pattern.
        subbox_size: Size of the subbox used to mask.
    """
    model.eval()
    print(data_point)
    with torch.no_grad():
        pred = model(data_point)

    # Flattened versions
    predicted = pred.argmax(dim=1).squeeze().cpu().numpy().reshape(H, W)
    ground_truth = data_point.y.squeeze().cpu().numpy().reshape(H, W)
    input_masked = data_point.x.squeeze().cpu().numpy().reshape(H, W)

    # Display
    print("\n Input (x):")
    print(input_masked)

    print("\n Ground truth (y):")
    print(ground_truth)

    print("\n Model prediction:")
    print(predicted)

    print("\n After merging:")
    print(merge_patterns_stack(np.expand_dims(input_masked, axis=0),np.expand_dims(predicted, axis=0),subbox_size))

if __name__ == "__main__":
    # TODO: decide which option to take: predict complete pattern or the complement pattern.
    import yaml 

    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Parameters from config
    NAME = "subshift_2" 
    BOX_SIZE = config["box_size"]
    SUBBOX_SIZE  = config["subbox_size"]
    ALPHABET_SIZE = get_alphabet_size(NAME)
    EPOCHS = config["epochs"]
    BATCH_SIZE = config["batch_size"]
    TEST_RATIO = config["test_ratio"]
    HIDDEN_DIMENSIONS = config["hidden_dimensions"]
    EMBEDDING_DIMENSIONS = config["embedding_dimensions"]
    NUMBER_ITERATIONS = config["number_iterations"]
    LOAD_MODEL = False
    
    # Load and preprocess pattern data
    complement_patterns = np.load(f"subbox_masked_patterns/{NAME}/all_patterns.npy")
    complement_patterns = convert_string_patterns_to_float(complement_patterns,mask_value=ALPHABET_SIZE)

    masked_patterns = np.load(f"outside_subbox_masked_patterns/{NAME}/all_patterns.npy")
    masked_patterns = convert_string_patterns_to_float(masked_patterns,mask_value=ALPHABET_SIZE)

    if LOAD_MODEL is False:
        # Train and save the model
        model,test_set = run_training(
            complement_patterns,
            masked_patterns,
            model_path="models/gnn.pt",
            size_alphabet=ALPHABET_SIZE,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            test_ratio = TEST_RATIO,
            num_hidden=HIDDEN_DIMENSIONS,
            num_iterations=NUMBER_ITERATIONS,
            embed_dim=EMBEDDING_DIMENSIONS
        )
    else: 
        with open("models/gnn_testset.pkl", "rb") as f:
            test_set = pickle.load(f)
        model = GNNModel(size_alphabet=ALPHABET_SIZE)  # Use the same size_alphabet as before
        # Load saved weights
        model.load_state_dict(torch.load("models/gnn.pt"))

    inspection_list = range(5)
    # inspection_list = [1]
    for k in inspection_list:
        sample = test_set[k]
        # Run inspection
        inspect_prediction(model, sample, BOX_SIZE, BOX_SIZE,SUBBOX_SIZE)

    loader = GeoDataLoader(test_set, batch_size=10)
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            preds = model(batch)
            preds = preds.squeeze().cpu().numpy().astype(int)
            batch = batch.num_graphs
            preds = preds.argmax(axis=1).reshape(batch, BOX_SIZE, BOX_SIZE)
            # preds = directional_round(preds, decimals=0).astype(int)
            preds[preds == 3] = -1
            all_preds.append(preds)

    predicted_stack = np.concatenate(all_preds, axis=0)
    input_stack = np.stack([
    data.x.squeeze().cpu().numpy().reshape(BOX_SIZE, BOX_SIZE)
    for data in test_set
        ])
    with open("samples.json", "r") as f:
        samples = json.load(f)
    eval = eval_subbox_to_outside(input_stack,
                                  predicted_stack,
                                  box_size=SUBBOX_SIZE,
                                  forbidden_patterns=samples[NAME]["forbidden_pairs"])
    print(eval)