"""
worlds.py — domino-SFT world generation and SAT-based config sampling.

A *world* is a nearest-neighbour subshift of finite type (SFT) over alphabet
{0, ..., q-1} on an N x N window, defined by a set of forbidden dominoes:
((a, b), 'h')  forbids symbol a immediately left of b (horizontal pair),
((a, b), 'v')  forbids symbol a immediately above b (vertical pair).

This module:
  - samples random worlds with forbidden-pattern density rho,
  - filters them (non-empty AND >= 2 distinct admissible configs),
  - samples diverse admissible configurations via a SAT solver with randomized
    variable phases (decorrelated, non-uniform sampling; bias is declared).
"""

from __future__ import annotations
import itertools
import numpy as np
from pysat.solvers import Cadical153
from pysat.card import CardEnc, EncType


# ----------------------------------------------------------------------------
# World = forbidden domino set
# ----------------------------------------------------------------------------

def all_dominoes(q):
    """Every possible domino ((a, b), orient) over a q-symbol alphabet."""
    out = []
    for a, b in itertools.product(range(q), repeat=2):
        out.append(((a, b), 'h'))
        out.append(((a, b), 'v'))
    return out


def sample_world(q, rho, rng):
    """Sample a forbidden domino set; each domino included i.i.d. with prob rho.

    Returns a frozenset of ((a, b), orient) tuples (canonical, hashable).
    """
    F = [d for d in all_dominoes(q) if rng.random() < rho]
    return frozenset(F)


def world_key(F):
    """Canonical hashable key for a world (for dedup)."""
    return tuple(sorted(F))


# ----------------------------------------------------------------------------
# Admissibility check (vectorized over a stack of configs)
# ----------------------------------------------------------------------------

def split_forbidden(F, q):
    """Return (H, V) boolean lookup arrays; H[a,b]/V[a,b] True iff forbidden.

    q must be the true alphabet size: F may omit the top symbol, so q cannot be
    inferred from F alone (configs may still contain that symbol).
    """
    H = np.zeros((q, q), dtype=bool)
    V = np.zeros((q, q), dtype=bool)
    for (a, b), o in F:
        (H if o == 'h' else V)[a, b] = True
    return H, V


def is_admissible(configs, F, q=None):
    """Vectorized admissibility test.

    configs : (..., N, N) int array. Returns boolean array of shape (...,).
    A config is admissible iff it contains no forbidden horizontal or vertical
    pair of adjacent cells. q is the alphabet size; if None it is taken large
    enough to cover both the configs and the forbidden set.
    """
    configs = np.asarray(configs)
    if configs.ndim == 2:
        configs = configs[None]
        squeeze = True
    else:
        squeeze = False
    if q is None:
        qF = 1 + max((max(a, b) for (a, b), _ in F), default=0)
        qC = int(configs.max()) + 1 if configs.size else 1
        q = max(qF, qC)
    H, V = split_forbidden(F, q)
    N = configs.shape[-1]
    bad = np.zeros(configs.shape[:-2], dtype=bool)
    # horizontal pairs (cell, cell-to-the-right)
    left = configs[..., :, :-1]
    right = configs[..., :, 1:]
    if H.any():
        bad |= H[left, right].any(axis=(-2, -1))
    # vertical pairs (cell, cell-below)
    top = configs[..., :-1, :]
    bot = configs[..., 1:, :]
    if V.any():
        bad |= V[top, bot].any(axis=(-2, -1))
    ok = ~bad
    return ok[0] if squeeze else ok


def observed_dominoes(configs, q):
    """Set of dominoes that actually occur in a stack of configs.

    Used by the frequency-learner baseline. Returns a frozenset in the same
    ((a, b), orient) format as a forbidden set.
    """
    configs = np.asarray(configs)
    if configs.ndim == 2:
        configs = configs[None]
    seen = set()
    left = configs[..., :, :-1].reshape(-1)
    right = configs[..., :, 1:].reshape(-1)
    for a, b in zip(left.tolist(), right.tolist()):
        seen.add(((a, b), 'h'))
    top = configs[..., :-1, :].reshape(-1)
    bot = configs[..., 1:, :].reshape(-1)
    for a, b in zip(top.tolist(), bot.tolist()):
        seen.add(((a, b), 'v'))
    return frozenset(seen)


def config_dominoes(config):
    """All dominoes present in a single config (with multiplicity collapsed)."""
    return observed_dominoes(config[None], None)


# ----------------------------------------------------------------------------
# SAT encoding and diverse config sampling
# ----------------------------------------------------------------------------

def _var(i, j, a, q, N):
    return i * N * q + j * q + a + 1


def _base_clauses(q, N, F):
    """Exactly-one-per-cell + forbidden-domino clauses (CNF, 1-indexed vars)."""
    clauses = []
    for i in range(N):
        for j in range(N):
            lits = [_var(i, j, a, q, N) for a in range(q)]
            clauses += CardEnc.atleast(lits=lits, bound=1, encoding=EncType.pairwise).clauses
            clauses += CardEnc.atmost(lits=lits, bound=1, encoding=EncType.pairwise).clauses
    for (a, b), o in F:
        if o == 'h':
            for i in range(N):
                for j in range(N - 1):
                    clauses.append([-_var(i, j, a, q, N), -_var(i, j + 1, b, q, N)])
        else:
            for i in range(N - 1):
                for j in range(N):
                    clauses.append([-_var(i, j, a, q, N), -_var(i + 1, j, b, q, N)])
    return clauses


def _decode(model, q, N):
    mset = set(l for l in model if l > 0)
    cfg = np.zeros((N, N), dtype=np.int8)
    for i in range(N):
        for j in range(N):
            for a in range(q):
                if _var(i, j, a, q, N) in mset:
                    cfg[i, j] = a
                    break
    return cfg


def count_at_least_two(q, N, F):
    """True iff the world has >= 2 distinct admissible configs (diversity filter)."""
    clauses = _base_clauses(q, N, F)
    with Cadical153(bootstrap_with=clauses) as s:
        if not s.solve():
            return False
        m1 = s.get_model()
        cfg = _decode(m1, q, N)
        # block this exact config and re-solve
        block = []
        for i in range(N):
            for j in range(N):
                block.append(-_var(i, j, int(cfg[i, j]), q, N))
        s.add_clause(block)
        return bool(s.solve())


def sample_configs(q, N, F, n_configs, rng, max_tries_factor=6):
    """Sample up to n_configs distinct admissible configs with decorrelation.

    Each config is produced by a fresh solver with randomized variable phases,
    then rejected if already seen. Non-uniform but diverse; the solver protocol
    is part of the declared benchmark distribution.
    """
    clauses = _base_clauses(q, N, F)
    seen = set()
    out = []
    n_vars = N * N * q
    tries = 0
    max_tries = max(n_configs * max_tries_factor, n_configs + 20)
    while len(out) < n_configs and tries < max_tries:
        tries += 1
        s = Cadical153(bootstrap_with=clauses)
        # randomize preferred phases -> different solution each time
        phases = [(v if rng.random() < 0.5 else -v) for v in range(1, n_vars + 1)]
        try:
            s.set_phases(literals=phases)
        except Exception:
            pass
        if not s.solve():
            s.delete()
            break
        cfg = _decode(s.get_model(), q, N)
        s.delete()
        key = cfg.tobytes()
        if key in seen:
            continue
        seen.add(key)
        out.append(cfg)
    if not out:
        return np.zeros((0, N, N), dtype=np.int8)
    return np.stack(out)


# ----------------------------------------------------------------------------
# World-pool construction
# ----------------------------------------------------------------------------

def build_world_pool(q, rho, N, n_worlds, n_configs, rng, exclude_keys=None,
                     min_configs=None, log_every=50, label=""):
    """Build a list of worlds, each {'F', 'key', 'configs'}.

    Keeps sampling random worlds until n_worlds pass the filters:
      - non-empty and >= 2 distinct admissible configs,
      - at least min_configs distinct configs actually sampled,
      - F not in exclude_keys and not a duplicate within the pool.
    """
    if min_configs is None:
        min_configs = max(4, n_configs // 4)
    exclude = set(exclude_keys or [])
    pool = []
    attempts = 0
    while len(pool) < n_worlds:
        attempts += 1
        F = sample_world(q, rho, rng)
        k = world_key(F)
        if k in exclude:
            continue
        if not count_at_least_two(q, N, F):
            continue
        configs = sample_configs(q, N, F, n_configs, rng)
        if configs.shape[0] < min_configs:
            continue
        exclude.add(k)
        pool.append({'F': F, 'key': k, 'configs': configs})
        if log_every and len(pool) % log_every == 0:
            print(f"  [{label}] {len(pool)}/{n_worlds} worlds "
                  f"({attempts} attempts, last had {configs.shape[0]} configs)",
                  flush=True)
    return pool


if __name__ == "__main__":
    import random
    rng = random.Random(0)
    F = sample_world(q=3, rho=0.3, rng=rng)
    print("forbidden:", sorted(F))
    print("diverse?", count_at_least_two(3, 8, F))
    cfgs = sample_configs(3, 8, F, 20, rng)
    print("sampled configs:", cfgs.shape)
    print("all admissible?", bool(is_admissible(cfgs, F).all()))
    print("#distinct observed dominoes:", len(observed_dominoes(cfgs, 3)))
