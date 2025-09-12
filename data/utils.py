import json 
import numpy as np
import networkx as nx
import torch

def get_alphabet_size(subshift_key):
    """
    Return the size of the alphabet for a given subshift key from 'samples.json' in the CWD.
    """
    with open("samples.json", 'r') as f:
        data = json.load(f)
    if subshift_key not in data:
        raise KeyError(f"Subshift key '{subshift_key}' not found in file.")
    alphabet = data[subshift_key]["alphabet"]
    return len(alphabet)


def convert_string_patterns_to_float(array, mask_symbol="*", mask_value=-100):
    """
    Convert a numpy array of strings (e.g., '0','1','*') to float32.
    '*' -> mask_value, others via float(x).
    """
    vectorized = np.vectorize(lambda x: mask_value if x == mask_symbol else float(x))
    return vectorized(array).astype(np.float32)

def create_grid_edges(H, W):
    """
    Construct bidirectional 4-neighbour edges on an H×W grid.
    Returns edge_index ∈ ℕ^{2×E} with E = 2 * (H*(W-1) + W*(H-1)).
    """
    G = nx.grid_2d_graph(H, W)
    mapping = {(i, j): i * W + j for i in range(H) for j in range(W)}
    G = nx.relabel_nodes(G, mapping)
    edge_list = list(G.edges())
    edge_index = torch.tensor(edge_list + [(j, i) for (i, j) in edge_list],
                              dtype=torch.long).t().contiguous()
    return edge_index
