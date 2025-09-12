"""
Dataset utilities and a LightningDataModule for SFT patterns, with two backends:
  (A) GCN backend: node = cell; edges = 4-neighbour grid.
  (B) CNF bipartite backend: literals ↔ clauses.

Selection:
----------
If cfg.model.name == "bipartite", we build items reflecting the CNF formulation of the completion problem.
Otherwise, if cfg.model.name == "gnn", we build data items which correspond directly to the input patterns.

Overview of backend (A):
------------------------------
For each sample:
    Data(x=(H*W,1), y=(H*W,1), edge_index=(2,E)) with bidirectional 4-neighbour edges.
Batching is by PyG’s GeoDataLoader.

Overview of backend (B):
------------------------------
For each sample (masked, complement) of shape (H, W):
  • Infer the inside "hole" (cells to fill) as those where masked == |A|.
  • Reduce observed context to the immediate 4-neighbour boundary around the hole (kept for future use).
  • Create boolean variables X_{p,s} for each inside cell p and symbol s ∈ {0,…,|A|-1}.
  • Create CNF clauses for EXACTLY-ONE per cell:
        (∨_s  X_{p,s})  and  (¬X_{p,s} ∨ ¬X_{p,t}) for all s<t
    (O(|A|^2) per cell; replace by sequential/ladder encoding if needed.)
  • Add SFT forbidden domino clauses (horizontal/vertical) from samples.json:
        For each forbidden pair (a,b) at adjacent cells u→v, add  (¬X_{u,a} ∨ ¬X_{v,b}).
        If u or v is observed (known boundary), add unit clause on the unknown neighbor.
  • Build a PyG Data with:
        x_l : (2V, 1)   literals (positives then negatives for each variable)
        x_c : (C, 1)    clauses (dummy features)
        adj_t_lit : SparseTensor (C, 2V) with edges (clause → literal)
        y_l : (2V,)     binary targets per literal (BCE-with-logits)
        y_l_mask : (2V,) bool mask of supervised literals
"""

from torch_geometric.loader import DataLoader as GeoDataLoader
import pytorch_lightning as pl
import numpy as np
from sklearn.model_selection import train_test_split

from torch_geometric.data import Data
import torch
from torch_sparse import SparseTensor
import os
import json

from .utils import (
    convert_string_patterns_to_float,
    create_grid_edges,
)

# ----------------------------- Grid (GCN) backend --------------------------- #

def make_dataset_grid(complement, masked):
    """
    Build a list of PyG `Data` objects for the grid backend.
    """
    N, H, W = complement.shape
    edge_index = create_grid_edges(H, W)
    dataset = []
    for i in range(N):
        x = torch.tensor(masked[i], dtype=torch.float).flatten().unsqueeze(1)
        y = torch.tensor(complement[i], dtype=torch.float).flatten().unsqueeze(1)
        data = Data(x=x, y=y, edge_index=edge_index)
        dataset.append(data)
    return dataset

# -------------------------- CNF (bipartite) backend ------------------------- #

def _boundary_layer(inside: torch.Tensor) -> torch.Tensor:
    """
    Given inside∈{False,True}^{H×W}, return boundary cells (adjacent in 4-neighbourhood to inside, but outside).
    """
    H, W = inside.shape
    pad = torch.zeros((H+2, W+2), dtype=torch.bool, device=inside.device)
    pad[1:-1, 1:-1] = inside
    nb = (pad[:-2, 1:-1] | pad[2:, 1:-1] | pad[1:-1, :-2] | pad[1:-1, 2:])
    boundary = nb & ~inside
    return boundary

def _reduce_observations_to_boundary(masked_i: torch.Tensor, inside: torch.Tensor, A: int) -> torch.Tensor:
    """
    Keep only observations on the immediate boundary of the inside region; set all other outside cells to the mask token A.
    """
    boundary = _boundary_layer(inside)
    reduced = torch.full_like(masked_i, fill_value=float(A))
    reduced[boundary] = masked_i[boundary]
    # Inside remains A by construction (unknowns).
    return reduced

def _cnf_from_sample(masked_i: torch.Tensor,
                     complement_i: torch.Tensor,
                     alphabet_size: int,
                     inside: torch.Tensor,
                     forbidden_pairs_idx: list[tuple[tuple[int, int], str]]) -> Data:
    """
    Build a CNF instance (literals ↔ clauses) for ONE sample.

    forbidden_pairs_idx: list of ((a_idx, b_idx), direction) with direction in {"horizontal","vertical"}.

    Literal ordering (required by GNN_SAT._flip_literals):
      For each variable v, literals are ordered as [pos(v), neg(v)].
      We address literals by indices 2*v (pos) and 2*v+1 (neg).

    Returns a Data with:
      x_l : (2V,1), x_c : (C,1), adj_t_lit : SparseTensor(C, 2V),
      y_l : (2V,), y_l_mask : (2V,)
    """
    H, W = masked_i.shape
    A = alphabet_size

    # ----- Variables: one boolean per (inside cell, symbol)
    var_index = {}  # (r,c,s) -> v
    for r in range(H):
        for c in range(W):
            if inside[r, c]:
                for s in range(A):
                    var_index[(r, c, s)] = len(var_index)
    V = len(var_index)
    L = 2 * V  # two literals per variable

    # Literal features (unused, shape (L,1))
    x_l = torch.zeros((L, 1), dtype=torch.float)

    clauses = []  # list[list[int]] of literal indices (pos = 2*v, neg = 2*v+1)

    # Group variables per cell for exactly-one constraints
    from collections import defaultdict
    per_cell = defaultdict(list)
    for (r, c, s), v in var_index.items():
        per_cell[(r, c)].append(v)

    # EXACTLY-ONE per inside cell
    for _, vs in per_cell.items():
        # At-least-one: (X_{p,0} ∨ ... ∨ X_{p,A-1})
        clauses.append([2 * v for v in vs])
        # At-most-one: pairwise (¬X_{p,s} ∨ ¬X_{p,t})
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                clauses.append([2 * vs[i] + 1, 2 * vs[j] + 1])

    # --- SFT forbidden domino clauses (horizontal/vertical) ---
    # Use ONLY boundary observations (unit clauses) and inside-inside binary clauses.
    observed = _reduce_observations_to_boundary(masked_i, inside, A)  # values in {0..A-1} or A=mask

    def var_id_if_exists(r: int, c: int, s: int):
        """Return (neg_literal_index, pos_literal_index) or None if variable does not exist."""
        v = var_index.get((r, c, s))
        if v is None:
            return None
        return (2 * v + 1, 2 * v)

    for (a_idx, b_idx), direction in forbidden_pairs_idx:
        if direction == "horizontal":
            for r in range(H):
                for c in range(W - 1):
                    # left (r,c) -> right (r,c+1)
                    left_obs = observed[r, c]
                    right_obs = observed[r, c + 1]

                    left_inside = inside[r, c].item()
                    right_inside = inside[r, c + 1].item()

                    if left_inside and right_inside:
                        # (¬X_{left,a} ∨ ¬X_{right,b})
                        Lneg = var_id_if_exists(r, c, a_idx)
                        Rneg = var_id_if_exists(r, c + 1, b_idx)
                        if Lneg and Rneg:
                            clauses.append([Lneg[0], Rneg[0]])
                    elif (not left_inside) and right_inside and int(left_obs) == a_idx:
                        # unit: ¬X_{right,b}
                        Rneg = var_id_if_exists(r, c + 1, b_idx)
                        if Rneg:
                            clauses.append([Rneg[0]])
                    elif left_inside and (not right_inside) and int(right_obs) == b_idx:
                        # unit: ¬X_{left,a}
                        Lneg = var_id_if_exists(r, c, a_idx)
                        if Lneg:
                            clauses.append([Lneg[0]])
                    # else: both observed or at least one unknown observed -> no clause
        elif direction == "vertical":
            for r in range(H - 1):
                for c in range(W):
                    # up (r,c) -> down (r+1,c)
                    up_obs = observed[r, c]
                    down_obs = observed[r + 1, c]

                    up_inside = inside[r, c].item()
                    down_inside = inside[r + 1, c].item()

                    if up_inside and down_inside:
                        Un = var_id_if_exists(r, c, a_idx)
                        Dn = var_id_if_exists(r + 1, c, b_idx)
                        if Un and Dn:
                            clauses.append([Un[0], Dn[0]])
                    elif (not up_inside) and down_inside and int(up_obs) == a_idx:
                        Dn = var_id_if_exists(r + 1, c, b_idx)
                        if Dn:
                            clauses.append([Dn[0]])
                    elif up_inside and (not down_inside) and int(down_obs) == b_idx:
                        Un = var_id_if_exists(r, c, a_idx)
                        if Un:
                            clauses.append([Un[0]])
        else:
            raise ValueError(f"Invalid direction: {direction}. Must be 'horizontal' or 'vertical'.")

    # ----- Supervision per literal (for BCE-with-logits in the wrapper)
    # Map var id -> symbol index
    var_to_sym = torch.empty(V, dtype=torch.long)
    for (r, c, s), v in var_index.items():
        var_to_sym[v] = s

    y_l = torch.zeros(L, dtype=torch.float)        # targets in {0., 1.}
    y_l_mask = torch.zeros(L, dtype=torch.bool)    # True -> include in loss

    targets = complement_i.long()  # ground truth symbols: {0..A-1} or -100 for unknown

    for (r, c), vs in per_cell.items():
        t = int(targets[r, c].item())
        if not (0 <= t < A):
            # Unknown / ignore: leave mask False for all literals at this cell
            continue
        for v in vs:
            s = int(var_to_sym[v].item())
            if s == t:
                # true symbol -> pos(v)=1, neg(v)=0
                y_l[2 * v] = 1.0
                y_l[2 * v + 1] = 0.0
            else:
                # other symbols -> pos(v)=0, neg(v)=1
                y_l[2 * v] = 0.0
                y_l[2 * v + 1] = 1.0
            y_l_mask[2 * v] = True
            y_l_mask[2 * v + 1] = True

    # Build SparseTensor adjacency (clause -> literal)
    C = len(clauses) if len(clauses) > 0 else 1
    if len(clauses) == 0:
        clauses = [[]]

    rows, cols = [], []
    for ci, lits in enumerate(clauses):
        for lit in lits:
            rows.append(ci)
            cols.append(lit)

    if len(rows) == 0:
        adj_t_lit = SparseTensor(
            row=torch.tensor([], dtype=torch.long),
            col=torch.tensor([], dtype=torch.long),
            sparse_sizes=(C, L),
        )
    else:
        row = torch.tensor(rows, dtype=torch.long)
        col = torch.tensor(cols, dtype=torch.long)
        adj_t_lit = SparseTensor(row=row, col=col, sparse_sizes=(C, L))

    x_c = torch.zeros((C, 1), dtype=torch.float)

    return Data(
        x_l=x_l,
        x_c=x_c,
        adj_t_lit=adj_t_lit,
        y_l=y_l,
        y_l_mask=y_l_mask,
    )

def make_dataset_cnf(masked_np: np.ndarray,
                     complement_np: np.ndarray,
                     alphabet_size: int,
                     forbidden_pairs_idx: list[tuple[tuple[int, int], str]]) -> list[Data]:
    """
    Build a list of CNF bipartite `Data` objects for GNN_SAT (graph_type='lit').
    We infer the inside region as cells where masked == |A|, and inject SFT forbidden clauses.
    """
    assert masked_np.shape == complement_np.shape and masked_np.ndim == 3
    N, _, _ = masked_np.shape
    dataset = []
    A = alphabet_size

    for i in range(N):
        masked_i = torch.from_numpy(masked_np[i]).float()
        complement_i = torch.from_numpy(complement_np[i]).float()

        # Inside mask: positions to fill are marked by the mask token A in the inputs
        inside = masked_i.eq(float(A))

        data = _cnf_from_sample(
            masked_i, complement_i, A, inside, forbidden_pairs_idx
        )
        dataset.append(data)

    return dataset

# ------------------------------ Data loading API ---------------------------- #

def _load_sft_spec(original_cwd: str, subshift_name: str):
    """
    Load alphabet and forbidden_pairs for the given subshift from samples.json.
    Returns (alphabet: list[str], forbidden_pairs_idx: list[((int,int), str)]).
    """
    with open(os.path.join(original_cwd, "samples.json"), "r") as f:
        spec = json.load(f)

    if subshift_name not in spec:
        raise KeyError(f"Subshift '{subshift_name}' not found in samples.json")

    entry = spec[subshift_name]
    alphabet = entry["alphabet"]
    fpairs = entry["forbidden_pairs"]  # [ [[a,b], "horizontal"|"vertical"], ... ]

    # Map letters to indices
    idx = {a: i for i, a in enumerate(alphabet)}
    forbidden_pairs_idx = [((idx[a], idx[b]), direction) for [a, b], direction in fpairs]
    return alphabet, forbidden_pairs_idx

def get_data(cfg):
    """
    Load arrays, convert symbols, and return dict with lists of Data.
    Chooses the backend based on cfg.model.name:
      - "bipartite" -> CNF bipartite data for GNN_SAT (x_l, x_c, adj_t_lit, y_l, y_l_mask)
      - otherwise   -> original grid data (x, y, edge_index)
    """
    NAME = cfg.data.name

    # Hydra may change runtime CWD; recover the original project root
    original_cwd = os.getcwd().split('/outputs')[0] if '/outputs' in os.getcwd() else os.getcwd()

    # Load SFT spec (alphabet and forbidden pairs)
    alphabet, forbidden_pairs_idx = _load_sft_spec(original_cwd, NAME)

    # Load arrays
    complement_patterns = np.load(f"{original_cwd}/subbox_masked_patterns/{NAME}/all_patterns.npy")
    complement_patterns = convert_string_patterns_to_float(complement_patterns, mask_value=-100)

    masked_patterns = np.load(f"{original_cwd}/outside_subbox_masked_patterns/{NAME}/all_patterns.npy")
    masked_patterns = convert_string_patterns_to_float(masked_patterns, mask_value=len(alphabet))

    # Select backend
    model_name = getattr(getattr(cfg, "model", {}), "name", "gnn")

    if model_name == "bipartite":
        dataset = make_dataset_cnf(
            masked_patterns,
            complement_patterns,
            alphabet_size=len(alphabet),
            forbidden_pairs_idx=forbidden_pairs_idx,
        )
    else:
        dataset = make_dataset_grid(complement_patterns, masked_patterns)

    train_set, test_set = train_test_split(dataset, test_size=cfg.data.test_ratio, random_state=42)
    data = {'train': train_set, 'val': test_set, 'test': test_set}
    return data

class SFTDataModule(pl.LightningDataModule):
    """
    LightningDataModule supporting both grid and CNF bipartite backends.

    For cfg.model.name == "bipartite":
      - Each item has x_l, x_c, adj_t_lit (SparseTensor), y_l, y_l_mask.
      - Loaders use follow_batch=['x_l','x_c'] so GNN_SAT receives x_l_batch.
    Else (grid):
      - Each item has x, y, edge_index (as before).
    """
    def __init__(self, data, batch_size=32, num_workers=0, model_name: str = "bipartite"):
        super().__init__()
        self.train_batch_size = batch_size
        self.val_batch_size = batch_size
        self.test_batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset, self.val_dataset, self.test_dataset = data['train'], data['val'], data['test']
        self.model_name = model_name

        self.follow_batch = ['x_l', 'x_c'] if self.model_name == "bipartite" else None

    def train_dataloader(self):
        return GeoDataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            follow_batch=self.follow_batch,
        )

    def val_dataloader(self):
        return GeoDataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            follow_batch=self.follow_batch,
        )

    def test_dataloader(self):
        return GeoDataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            follow_batch=self.follow_batch,
        )
