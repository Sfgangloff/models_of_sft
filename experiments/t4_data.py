"""
t4_data.py — data for the TRANSPARENT T4 (completion) task.

Transparent mode: the world's rules F are given to the model as input (a binary
vector over all possible dominoes). The model sees a partially-masked admissible
grid and must fill the masked cells so the completed grid is admissible.

Because each masked grid is produced by masking a genuinely admissible grid, a
valid completion always exists (the original), so no SAT call is needed at eval:
we simply check whether the model's completed grid avoids all forbidden dominoes.
"""
from __future__ import annotations
import numpy as np
from worlds import is_admissible


def domino_list(q):
    """Canonical order of all 2*q^2 possible dominoes ((a,b), orient)."""
    out = []
    for orient in ('h', 'v'):
        for a in range(q):
            for b in range(q):
                out.append(((a, b), orient))
    return out


def forbidden_vector(F, q):
    """Binary vector over domino_list(q): 1 if that domino is forbidden in F."""
    idx = {d: i for i, d in enumerate(domino_list(q))}
    v = np.zeros(len(idx), dtype=np.int64)
    for d in F:
        v[idx[d]] = 1
    return v


def make_masked(grid, p, rng, mask_token):
    """Randomly mask each cell with prob p (>=1 cell guaranteed). Returns
    (masked_grid with mask_token in holes, boolean mask)."""
    N = grid.shape[0]
    mask = np.zeros((N, N), dtype=bool)
    for i in range(N):
        for j in range(N):
            if rng.random() < p:
                mask[i, j] = True
    if not mask.any():                      # guarantee at least one hole
        mask[rng.randrange(N), rng.randrange(N)] = True
    masked = grid.copy()
    masked[mask] = mask_token
    return masked, mask


def build_batch_t4(worlds, B, q, p, rng, world_indices=None, mismatch=False):
    """Build a completion batch.

    Returns:
      rule  (B, D) int64   forbidden-domino vector given to the model
      masked(B, N, N) int8 grid with holes = q (MASK token)
      target(B, N, N) int8 original grid (ground truth for masked cells)
      mask  (B, N, N) bool which cells are holes
      Fs    list[frozenset] the TRUE rules of each item's world (for eval)
    mismatch=True gives the model another world's rule vector while target/Fs
    stay from the item's own world (the mismatched-rule control).
    """
    pool = list(world_indices) if world_indices is not None else list(range(len(worlds)))
    N = worlds[0]['configs'].shape[1]
    D = 2 * q * q
    mask_token = q
    rule = np.zeros((B, D), dtype=np.int64)
    masked = np.zeros((B, N, N), dtype=np.int8)
    target = np.zeros((B, N, N), dtype=np.int8)
    holes = np.zeros((B, N, N), dtype=bool)
    Fs = []
    for b in range(B):
        w = worlds[pool[rng.randrange(len(pool))]]
        cfgs = w['configs']
        grid = cfgs[rng.randrange(cfgs.shape[0])]
        m_grid, m = make_masked(grid, p, rng, mask_token)
        masked[b] = m_grid
        target[b] = grid
        holes[b] = m
        Fs.append(w['F'])
        if mismatch:
            w2 = worlds[pool[rng.randrange(len(pool))]]
            rule[b] = forbidden_vector(w2['F'], q)   # WRONG rules
        else:
            rule[b] = forbidden_vector(w['F'], q)
    return rule, masked, target, holes, Fs


def _forbidden_sets(F):
    """Return (H, V) dicts of forbidden ordered pairs for quick lookup."""
    H = set(); V = set()
    for (a, b), o in F:
        (H if o == 'h' else V).add((a, b))
    return H, V


def forced_cell_stats_split(pred_full, masked, holes, Fs, q, U):
    """Forced-cell accuracy split by whether the forcing depends on a held-out
    ('U') domino.

    A forced cell (unique locally-admissible symbol s*) is **U-critical** if some
    other symbol s' is excluded *only* by U-dominoes (without the held-out rules the
    cell would not be forced to s*). It is **S-sufficient** otherwise. Tests whether
    the model can act on rule components never exercised in training.
    Returns accuracies and counts for all / S-sufficient / U-critical.
    """
    B, N, _ = pred_full.shape
    agg = {'all': [0, 0], 'S': [0, 0], 'U': [0, 0]}
    for b in range(B):
        F = Fs[b]; g = masked[b]; hole = holes[b]
        for i in range(N):
            for j in range(N):
                if not hole[i, j]:
                    continue
                adm = []
                excl_by_S = {}   # s' -> excluded by at least one S-domino?
                for s in range(q):
                    nbrs = []
                    if j > 0 and not hole[i, j-1]:   nbrs.append(((int(g[i, j-1]), s), 'h'))
                    if j < N-1 and not hole[i, j+1]: nbrs.append(((s, int(g[i, j+1])), 'h'))
                    if i > 0 and not hole[i-1, j]:   nbrs.append(((int(g[i-1, j]), s), 'v'))
                    if i < N-1 and not hole[i+1, j]: nbrs.append(((s, int(g[i+1, j])), 'v'))
                    excluders = [d for d in nbrs if d in F]
                    if excluders:
                        excl_by_S[s] = any(d not in U for d in excluders)
                    else:
                        adm.append(s)
                if len(adm) != 1:
                    continue
                s_star = adm[0]
                correct = 1 if int(pred_full[b, i, j]) == s_star else 0
                u_critical = any(not excl_by_S[sp] for sp in range(q) if sp != s_star)
                agg['all'][0] += correct; agg['all'][1] += 1
                k = 'U' if u_critical else 'S'
                agg[k][0] += correct; agg[k][1] += 1
    def acc(k):
        return agg[k][0] / agg[k][1] if agg[k][1] else float('nan')
    return {'all': acc('all'), 'S_sufficient': acc('S'), 'U_critical': acc('U'),
            'n_all': agg['all'][1], 'n_S': agg['S'][1], 'n_U': agg['U'][1]}


def forced_cell_stats(pred_full, masked, holes, Fs, q, mask_token=None):
    """Accuracy on *locally-forced* masked cells.

    A hole cell is locally-forced if, given its OBSERVED neighbours and the rules,
    exactly one symbol is locally admissible (forms no forbidden domino with any
    observed neighbour). That unique symbol must be the true value (the original is
    locally admissible). A rule-blind model cannot compute it, so this isolates
    genuine rule-use. Returns (accuracy, n_forced).
    """
    if mask_token is None:
        mask_token = q
    B, N, _ = pred_full.shape
    correct = 0; total = 0
    for b in range(B):
        H, V = _forbidden_sets(Fs[b])
        g = masked[b]; hole = holes[b]
        for i in range(N):
            for j in range(N):
                if not hole[i, j]:
                    continue
                admissible = []
                for s in range(q):
                    ok = True
                    if j > 0 and not hole[i, j-1] and (int(g[i, j-1]), s) in H: ok = False      # left
                    if ok and j < N-1 and not hole[i, j+1] and (s, int(g[i, j+1])) in H: ok = False  # right
                    if ok and i > 0 and not hole[i-1, j] and (int(g[i-1, j]), s) in V: ok = False     # up
                    if ok and i < N-1 and not hole[i+1, j] and (s, int(g[i+1, j])) in V: ok = False   # down
                    if ok:
                        admissible.append(s)
                if len(admissible) == 1:
                    total += 1
                    if int(pred_full[b, i, j]) == admissible[0]:
                        correct += 1
    return (correct / total if total else float('nan')), total


def completion_success(pred_full, masked, holes, Fs, q):
    """Fraction of items whose completed grid is admissible.

    pred_full (B,N,N) argmax predictions; we keep observed cells and fill holes
    with predictions, then check admissibility against each item's true F.
    """
    B = pred_full.shape[0]
    ok = 0
    for b in range(B):
        g = masked[b].copy()
        g[holes[b]] = pred_full[b][holes[b]]
        if is_admissible(g, Fs[b], q):
            ok += 1
    return ok / B
