"""
model.py — in-context Transformer for opaque T1 validity classification.

Input: K context configs (admissible) + 1 query config, each an N x N grid.
Tokenisation (per cell): symbol embedding + shared 2D cell-position embedding +
role embedding (context vs query). No per-config index embedding, so the token
multiset is invariant to the ordering of context configs -> the readout is
permutation-invariant over context, as it should be for an i.i.d. sample.

A learned [READ] token (symbol id = q, its own position slot, query role) is
appended; its final hidden state is decoded to a binary admissible/not logit.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


class InContextT1(nn.Module):
    def __init__(self, q, N, K, d_model=96, nhead=4, num_layers=3, dim_ff=256,
                 dropout=0.0):
        super().__init__()
        self.q, self.N, self.K = q, N, K
        self.cells = N * N
        self.read_pos = self.cells            # extra position slot for [READ]
        self.read_sym = q                     # extra symbol id for [READ]

        self.sym_emb = nn.Embedding(q + 1, d_model)      # symbols + READ
        self.pos_emb = nn.Embedding(self.cells + 1, d_model)
        self.role_emb = nn.Embedding(2, d_model)         # 0 context, 1 query/read

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2)

        # precompute static position / role id templates (depend only on K,N)
        pos, role = self._templates(K)
        self.register_buffer('pos_ids', pos, persistent=False)
        self.register_buffer('role_ids', role, persistent=False)

    def _templates(self, K):
        cell_pos = np.arange(self.cells)
        pos = np.concatenate([np.tile(cell_pos, K),   # K context configs
                              cell_pos,                # query config
                              [self.read_pos]])        # READ token
        role = np.concatenate([np.zeros(self.cells * K, dtype=np.int64),
                               np.ones(self.cells, dtype=np.int64),
                               [1]])
        return torch.tensor(pos, dtype=torch.long), torch.tensor(role, dtype=torch.long)

    def _sym_ids(self, context, query):
        """Assemble symbol-id sequences. context (B,K,N,N), query (B,N,N)."""
        B = query.shape[0]
        ctx = context.reshape(B, -1)               # (B, K*cells)
        qry = query.reshape(B, -1)                 # (B, cells)
        read = torch.full((B, 1), self.read_sym, dtype=torch.long, device=qry.device)
        return torch.cat([ctx, qry, read], dim=1)  # (B, L)

    def forward(self, context, query):
        sym = self._sym_ids(context, query)        # (B, L)
        B, L = sym.shape
        pos = self.pos_ids.unsqueeze(0).expand(B, L)
        role = self.role_ids.unsqueeze(0).expand(B, L)
        h = self.sym_emb(sym) + self.pos_emb(pos) + self.role_emb(role)
        h = self.encoder(h)
        h_read = self.norm(h[:, -1])               # [READ] token
        return self.head(h_read)                   # (B, 2)
