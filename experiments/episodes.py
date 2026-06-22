"""
episodes.py — T1 (validity classification) episode construction.

An episode is (K admissible context configs, 1 query config, label), where the
query is either a held-out admissible config (label 1) or a hard negative
(label 0): an admissible config minimally corrupted until it violates F.
"""

from __future__ import annotations
import numpy as np
from worlds import is_admissible


def make_hard_negative(config, F, q, rng, max_single=40):
    """Corrupt an admissible config until it is inadmissible, changing as few
    cells as possible (prefer a single-cell change that creates one violation).
    """
    N = config.shape[0]
    # try single-cell corruptions first (hard negatives near the manifold)
    for _ in range(max_single):
        y = config.copy()
        i, j = rng.randrange(N), rng.randrange(N)
        choices = [a for a in range(q) if a != config[i, j]]
        y[i, j] = choices[rng.randrange(len(choices))]
        if not is_admissible(y, F, q):
            return y
    # fall back: corrupt increasing numbers of cells until inadmissible
    for k in range(2, N * N + 1):
        y = config.copy()
        idx = rng.sample(range(N * N), k)
        for t in idx:
            i, j = divmod(t, N)
            choices = [a for a in range(q) if a != config[i, j]]
            y[i, j] = choices[rng.randrange(len(choices))]
        if not is_admissible(y, F, q):
            return y
    return None  # world has no inadmissible configs at all (degenerate; filtered out)


def build_episode(world, K, q, rng):
    """Return (context (K,N,N) int8, query (N,N) int8, label int).

    Context configs and the positive-query config are sampled without replacement
    from the world's config pool, so the query grid is never identical to a
    context grid (they share dominoes — that is the signal, not the leak).
    """
    configs = world['configs']
    n = configs.shape[0]
    need = K + 1
    if n < need:
        idx = list(rng.choices(range(n), k=need))  # with replacement if tiny
    else:
        idx = rng.sample(range(n), need)
    context = configs[idx[:K]]
    base = configs[idx[K]]
    if rng.random() < 0.5:
        return context, base.copy(), 1
    neg = make_hard_negative(base, world['F'], q, rng)
    if neg is None:                      # extremely rare; resample as positive
        return context, base.copy(), 1
    return context, neg, 0


def build_batch(worlds, B, K, q, rng, world_indices=None, mismatch=False):
    """Build a batch of B episodes drawn from the given list of worlds.

    world_indices : optional iterable restricting which worlds to draw from.
    mismatch      : if True, the query comes from a DIFFERENT world than the
                    context (the mismatched-context control); label is computed
                    against the context world's F, so a context-using model
                    should collapse to chance.
    Returns (context (B,K,N,N) int8, query (B,N,N) int8, labels (B,) int).
    """
    pool = world_indices if world_indices is not None else range(len(worlds))
    pool = list(pool)
    N = worlds[0]['configs'].shape[1]
    ctx = np.zeros((B, K, N, N), dtype=np.int8)
    qry = np.zeros((B, N, N), dtype=np.int8)
    lab = np.zeros((B,), dtype=np.int64)
    for b in range(B):
        w = worlds[pool[rng.randrange(len(pool))]]
        if not mismatch:
            c, qr, y = build_episode(w, K, q, rng)
        else:
            # context from w; query from another world w2, labelled under w's F
            other = pool[rng.randrange(len(pool))]
            w2 = worlds[other]
            cfgs = w['configs']
            ci = (rng.sample(range(cfgs.shape[0]), K)
                  if cfgs.shape[0] >= K else rng.choices(range(cfgs.shape[0]), k=K))
            c = cfgs[ci]
            qc = w2['configs'][rng.randrange(w2['configs'].shape[0])]
            if rng.random() < 0.5:
                qr = qc.copy()
            else:
                neg = make_hard_negative(qc, w2['F'], q, rng)
                qr = neg if neg is not None else qc.copy()
            y = int(is_admissible(qr, w['F'], q))   # label under the CONTEXT world
        ctx[b] = c
        qry[b] = qr
        lab[b] = y
    return ctx, qry, lab
