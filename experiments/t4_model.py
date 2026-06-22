"""
t4_model.py — rule-conditioned Transformer for transparent T4 (completion).

Input tokens:
  - D = 2*q^2 "rule" tokens, one per possible domino, each tagged allowed/forbidden
    (this is how the world's rules F enter the model);
  - N*N "grid" tokens, each a cell value (symbol or MASK) + its 2D position.
A segment embedding separates rule tokens from grid tokens. Full self-attention
lets every grid cell attend to the rule tokens and to other cells (constraint
propagation). Per-cell heads on the grid tokens output symbol logits.

Feeding an all-zeros rule vector yields the "rule-blind" baseline (the model is
told nothing about which dominoes are forbidden).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


class RuleCompleter(nn.Module):
    def __init__(self, q, N, d_model=96, nhead=4, num_layers=4, dim_ff=256, dropout=0.0):
        super().__init__()
        self.q, self.N = q, N
        self.D = 2 * q * q          # number of possible dominoes / rule tokens
        self.cells = N * N
        self.mask_token = q

        self.domino_emb = nn.Embedding(self.D, d_model)      # which domino
        self.forbid_emb = nn.Embedding(2, d_model)           # allowed / forbidden
        self.sym_emb = nn.Embedding(q + 1, d_model)          # symbols + MASK
        self.pos_emb = nn.Embedding(self.cells, d_model)     # 2D cell position
        self.seg_emb = nn.Embedding(2, d_model)              # 0 rule, 1 grid

        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                           dim_feedforward=dim_ff, dropout=dropout,
                                           batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, q)

        self.register_buffer('domino_ids', torch.arange(self.D), persistent=False)
        self.register_buffer('cell_ids', torch.arange(self.cells), persistent=False)

    def forward(self, rule, grid):
        """rule (B, D) int64 {0,1}; grid (B, N, N) int64 in 0..q (q = MASK)."""
        B = grid.shape[0]
        # rule tokens
        dom = self.domino_emb(self.domino_ids).unsqueeze(0).expand(B, -1, -1)
        rule_tok = dom + self.forbid_emb(rule) + self.seg_emb(
            torch.zeros(1, dtype=torch.long, device=rule.device))
        # grid tokens
        g = grid.reshape(B, -1)
        grid_tok = (self.sym_emb(g)
                    + self.pos_emb(self.cell_ids).unsqueeze(0)
                    + self.seg_emb(torch.ones(1, dtype=torch.long, device=g.device)))
        h = torch.cat([rule_tok, grid_tok], dim=1)           # (B, D+cells, d)
        h = self.encoder(h)
        h_grid = self.norm(h[:, self.D:])                    # grid tokens only
        return self.head(h_grid).reshape(B, self.N, self.N, self.q)
