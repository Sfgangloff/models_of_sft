
from torch_geometric.loader import DataLoader as GeoDataLoader
import pytorch_lightning as pl
import numpy as np
from sklearn.model_selection import train_test_split
import networkx as nx
from torch_geometric.data import Data
import torch
import os
import json

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

def convert_string_patterns_to_float(array, mask_symbol="*", mask_value=-100):
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



def get_data(cfg):
    # Parameters from config
    NAME = cfg.data.name 
    
    # Get the directory where the script was originally called from
    original_cwd = os.getcwd().split('/outputs')[0] if '/outputs' in os.getcwd() else os.getcwd()
    
    complement_patterns = np.load(f"{original_cwd}/subbox_masked_patterns/{NAME}/all_patterns.npy")
    complement_patterns = convert_string_patterns_to_float(complement_patterns,mask_value=-100)

    masked_patterns = np.load(f"{original_cwd}/outside_subbox_masked_patterns/{NAME}/all_patterns.npy")
    masked_patterns = convert_string_patterns_to_float(masked_patterns,mask_value=len(cfg.data.alphabet))


    dataset = make_dataset(complement_patterns, masked_patterns)
    train_set, test_set = train_test_split(dataset, test_size=cfg.data.test_ratio, random_state=42)
    
    data = {'train': train_set, 'val': test_set, 'test': test_set}

    return data

class SFTDataModule(pl.LightningDataModule):
   def __init__(self, data, batch_size=32, num_workers=0):
       super().__init__()
       self.train_batch_size = batch_size
       self.val_batch_size = batch_size
       self.test_batch_size = batch_size
       self.num_workers = num_workers
       self.train_dataset, self.val_dataset, self.test_dataset = data['train'], data['val'], data['test']

   def train_dataloader(self):
       return GeoDataLoader(
           self.train_dataset, 
           batch_size=self.train_batch_size,
           shuffle=True, 
           num_workers=self.num_workers,
       )

   def val_dataloader(self):
       return GeoDataLoader(
           self.val_dataset, 
           batch_size=self.val_batch_size, 
           num_workers=self.num_workers,
       )

   def test_dataloader(self):
       return GeoDataLoader(
           self.test_dataset, 
           batch_size=self.test_batch_size, 
           num_workers=self.num_workers,

       )