"""
causal_complexity.py — computable proxies for a world's causal complexity κ.

The thesis: a world's training-informativeness tracks the complexity of its causal
(forcing) structure. κ is the minimal number of generators of the forcing relation
(a shift-invariant analogue of the Duquenne–Guigues implication base). Exact κ on
the lattice is intractable; here we compute a **radius-1 forcing-generator count**
κ̂₁ — the deterministic local edges — which reproduces the canonical values
(full shift → 0; two-uniform-configs → 4). Documented convention; validate with the
domain expert before relying on it.

A radius-1 generator (forward convention): for an orientation (h/v) and a source
symbol `a`, if `a` has a UNIQUE allowed successor `b` (exactly one `b` with the
domino (a,b) not forbidden), then "a forces b in that orientation" is one generator.
κ̂₁ counts these over both orientations. (κ̂₁_fb additionally counts unique
*predecessors*; reported as a richer variant.)
"""
from __future__ import annotations
import numpy as np
from worlds import is_admissible


def _forbidden_by_orient(F):
    H = set(); V = set()
    for (a, b), o in F:
        (H if o == 'h' else V).add((a, b))
    return H, V


def kappa1(F, q, include_backward=False):
    """Radius-1 forcing-generator count (forward convention). See module doc."""
    H, V = _forbidden_by_orient(F)
    gens = 0
    for forb in (H, V):
        for a in range(q):                      # forward: a -> unique successor?
            succ = [b for b in range(q) if (a, b) not in forb]
            if len(succ) == 1:
                gens += 1
        if include_backward:
            for b in range(q):                  # backward: unique predecessor of b?
                pred = [a for a in range(q) if (a, b) not in forb]
                if len(pred) == 1:
                    gens += 1
    return gens


def forcing_density(world, q, n_masked=200, p=0.4, seed=0):
    """Empirical fraction of masked cells that are locally forced (unique admissible
    symbol given observed neighbours), over random masks of the world's configs.
    Complementary to κ̂₁: measures how MUCH forcing happens (captures multi-cell
    forcing), not the generator count."""
    import random
    rng = random.Random(seed)
    F = world['F']; cfgs = world['configs']; N = cfgs.shape[1]
    forced = 0; total = 0
    H, V = _forbidden_by_orient(F)
    for _ in range(n_masked):
        g = cfgs[rng.randrange(cfgs.shape[0])]
        # random mask
        mask = np.zeros((N, N), bool)
        for i in range(N):
            for j in range(N):
                if rng.random() < p:
                    mask[i, j] = True
        for i in range(N):
            for j in range(N):
                if not mask[i, j]:
                    continue
                adm = []
                for s in range(q):
                    ok = True
                    if j > 0 and not mask[i, j-1] and (int(g[i, j-1]), s) in H: ok = False
                    if ok and j < N-1 and not mask[i, j+1] and (s, int(g[i, j+1])) in H: ok = False
                    if ok and i > 0 and not mask[i-1, j] and (int(g[i-1, j]), s) in V: ok = False
                    if ok and i < N-1 and not mask[i+1, j] and (s, int(g[i+1, j])) in V: ok = False
                    if ok:
                        adm.append(s)
                total += 1
                if len(adm) == 1:
                    forced += 1
    return forced / total if total else 0.0


def _admissible_patch(F, q, h, w):
    """All h×w patterns avoiding forbidden dominoes within the patch."""
    import itertools
    H, V = _forbidden_by_orient(F)
    pats = []
    for flat in itertools.product(range(q), repeat=h * w):
        g = np.array(flat, dtype=np.int8).reshape(h, w)
        ok = True
        for i in range(h):
            for j in range(w):
                if j < w - 1 and (int(g[i, j]), int(g[i, j+1])) in H: ok = False; break
                if i < h - 1 and (int(g[i, j]), int(g[i+1, j])) in V: ok = False; break
            if not ok:
                break
        if ok:
            pats.append(flat)
    return pats, h, w


def count_forcing_implications(F, q, h=3, w=3, max_premise=2):
    """Richer causal-complexity κ̂₂: number of IRREDUCIBLE forcing implications
    (premise on <=max_premise cells -> a forced target cell value) within an h×w
    patch, counted up to translation.

    Local approximation of the forcing relation (uses patch-admissible patterns).
    Convention: counts all irreducible forcing implications (both directions), so it
    is a *superset* of the author's forward-only hand count — same spirit (0 for the
    full shift; grows with forcing structure), validated below. Returns dict with the
    count by premise size and the total (= κ̂₂).
    """
    import itertools
    pats, h, w = _admissible_patch(F, q, h, w)
    if not pats:
        return {'by_size': {}, 'kappa2': 0}
    P = np.array(pats, dtype=np.int8)               # (npat, h*w)
    cells = [(i, j) for i in range(h) for j in range(w)]
    idx = {c: k for k, c in enumerate(cells)}
    gens = set()
    found_for_target = {}   # (norm premise) cache not needed; track irreducibility per (premise-cells,target)

    def normalize(prem_items, tcell, tval):
        allc = [c for c, _ in prem_items] + [tcell]
        mr = min(c[0] for c in allc); mc = min(c[1] for c in allc)
        pn = frozenset(((c[0]-mr, c[1]-mc), v) for c, v in prem_items)
        return (pn, (tcell[0]-mr, tcell[1]-mc), tval)

    # for irreducibility we record forcing facts of smaller premises
    forced_by = {}   # (frozenset premise-cells-with-values) -> set of (tcell,tval) forced
    for size in range(1, max_premise + 1):
        for prem_cells in itertools.combinations(cells, size):
            pcols = [idx[c] for c in prem_cells]
            # group patterns by their values on prem_cells
            from collections import defaultdict
            groups = defaultdict(list)
            for r in range(P.shape[0]):
                key = tuple(int(P[r, col]) for col in pcols)
                groups[key].append(r)
            for key, rows in groups.items():
                prem_items = tuple((prem_cells[t], key[t]) for t in range(size))
                prem_fs = frozenset(prem_items)
                sub = P[rows]
                for tcell in cells:
                    if tcell in prem_cells:
                        continue
                    # local only: target must be adjacent to some premise cell
                    # (avoids counting long-range / transitively-composed forces)
                    if not any(abs(tcell[0]-pc[0]) + abs(tcell[1]-pc[1]) == 1 for pc in prem_cells):
                        continue
                    tcol = idx[tcell]
                    vals = set(int(x) for x in sub[:, tcol])
                    if len(vals) == 1:
                        tval = vals.pop()
                        # nontrivial: target not globally constant across all pats
                        if len(set(int(x) for x in P[:, tcol])) == 1:
                            continue
                        # irreducible: no proper subset of prem already forces (tcell,tval)
                        irred = True
                        for ssize in range(1, size):
                            for sub_cells in itertools.combinations(prem_items, ssize):
                                if (tcell, tval) in forced_by.get(frozenset(sub_cells), set()):
                                    irred = False; break
                            if not irred:
                                break
                        forced_by.setdefault(prem_fs, set()).add((tcell, tval))
                        if irred:
                            gens.add(normalize(prem_items, tcell, tval))
    by_size = {}
    for g in gens:
        s = len(g[0]); by_size[s] = by_size.get(s, 0) + 1
    return {'by_size': by_size, 'kappa2': len(gens)}


def kappa2(F, q):
    return count_forcing_implications(F, q)['kappa2']


if __name__ == "__main__":
    import itertools
    # ---- validation on the canonical examples ----
    def all_dominoes(q):
        out = []
        for a, b in itertools.product(range(q), repeat=2):
            out.append(((a, b), 'h')); out.append(((a, b), 'v'))
        return out

    # full shift: no forbidden dominoes
    print("full shift      kappa1 =", kappa1(frozenset(), 2), "(expect 0)")

    # two-uniform configs over {0,1}: forbid every domino except (0,0) and (1,1)
    keep = {((0, 0), 'h'), ((1, 1), 'h'), ((0, 0), 'v'), ((1, 1), 'v')}
    F2 = frozenset(d for d in all_dominoes(2) if d not in keep)
    print("two-uniform     kappa1 =", kappa1(F2, 2), "(expect 4)")
    print("two-uniform  kappa1_fb =", kappa1(F2, 2, include_backward=True), "(forward+backward)")

    # a checkerboard-ish q=2 world: forbid (0,0) and (1,1) both orientations
    Fcheck = frozenset({((0, 0), 'h'), ((1, 1), 'h'), ((0, 0), 'v'), ((1, 1), 'v')})
    print("anti-uniform    kappa1 =", kappa1(Fcheck, 2),
          "(each symbol has unique successor -> 4)")

    # richer kappa2 (multi-cell forcing implications, up to translation)
    print("full shift      kappa2 =", count_forcing_implications(frozenset(), 2), "(expect 0)")
    print("two-uniform     kappa2 =", count_forcing_implications(F2, 2))
    print("anti-uniform    kappa2 =", count_forcing_implications(Fcheck, 2))
