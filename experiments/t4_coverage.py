"""
t4_coverage.py — Experiment C (Coverage).

Hold out a subset U of dominoes: training worlds forbid ONLY S-dominoes (the
complement), so the model never sees any U-domino exercised. Then test on worlds
that forbid k dominoes total, varying j = how many come from U. Measure forced-cell
accuracy split by whether each cell's forcing depends on a held-out (U) domino.

Reads out the mechanism:
  - per-domino lookup  -> U-critical accuracy ~ chance/blind (coverage strict)
  - general operation  -> U-critical accuracy stays high (coverage violated)

Safe envelope: CPU, 2 threads, niced, watchdog; results saved per-j.
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
try: os.nice(10)
except Exception: pass
import json, random, sys, time, gc
import numpy as np, torch
torch.set_num_threads(2)
sys.path.insert(0, os.path.dirname(__file__))
from worlds import count_at_least_two, sample_configs
from t4_data import domino_list, build_batch_t4, forced_cell_stats_split, forced_cell_stats
from t4_model import RuleCompleter
from t4_run import masked_loss

HERE = os.path.dirname(__file__); RES = os.path.join(HERE, "results"); FIG = os.path.join(HERE, "figs")
for d in (RES, FIG): os.makedirs(d, exist_ok=True)

Q, N = 3, 6
RHO_TRAIN = 0.35           # density over S-dominoes for training worlds
K_TEST = 4                 # forbidden dominoes per test world
N_TRAIN, N_TEST_PER_J = 64, 14
N_CFG, MINC = 40, 22
D_MODEL, LAYERS, NHEAD, DIMFF = 80, 3, 4, 256
STEPS, BATCH, LR, P = 2000, 32, 3e-4, 0.4
N_EVAL = 600
PART_SEED, U_SIZE = 7, 6
OUT = os.path.join(RES, "t4_coverage.json")


def make_world(F, rng):
    """Filter + sample configs for forbidden set F; return world dict or None."""
    F = frozenset(F)
    if not count_at_least_two(Q, N, F):
        return None
    cfgs = sample_configs(Q, N, F, N_CFG, rng)
    if cfgs.shape[0] < MINC:
        return None
    return {'F': F, 'configs': cfgs}


def main():
    dominoes = domino_list(Q)
    rp = random.Random(PART_SEED); order = dominoes[:]; rp.shuffle(order)
    U = frozenset(order[:U_SIZE]); S = [d for d in dominoes if d not in U]
    print(f"q={Q} N={N} |dominoes|={len(dominoes)}  |U|={len(U)} |S|={len(S)}", flush=True)
    print(f"U (held out): {sorted(U)}", flush=True)

    rng = random.Random(123)
    # training worlds: forbid only S-dominoes
    train = []
    attempts = 0
    while len(train) < N_TRAIN and attempts < N_TRAIN * 40:
        attempts += 1
        F = [d for d in S if rng.random() < RHO_TRAIN]
        w = make_world(F, rng)
        if w is not None:
            train.append(w)
    print(f"built {len(train)} training worlds (all F subset of S)", flush=True)

    # test worlds per j = #forbidden dominoes drawn from U
    test_by_j = {}
    for j in range(0, K_TEST + 1):
        worlds_j = []; att = 0
        while len(worlds_j) < N_TEST_PER_J and att < N_TEST_PER_J * 80:
            att += 1
            Fs = random.sample(S, K_TEST - j) + random.sample(list(U), j)
            w = make_world(Fs, rng)
            if w is not None:
                worlds_j.append(w)
        test_by_j[j] = worlds_j
        print(f"  j={j}: {len(worlds_j)} test worlds", flush=True)

    # train rule-conditioned model on training worlds
    torch.manual_seed(0); rtr = random.Random(0)
    model = RuleCompleter(Q, N, d_model=D_MODEL, nhead=NHEAD, num_layers=LAYERS, dim_ff=DIMFF)
    opt = torch.optim.Adam(model.parameters(), lr=LR); model.train()
    t0 = time.time()
    for step in range(1, STEPS + 1):
        rule, masked, target, holes, _ = build_batch_t4(train, BATCH, Q, P, rtr)
        logits = model(torch.from_numpy(rule), torch.from_numpy(masked.astype(np.int64)))
        opt.zero_grad(); masked_loss(logits, torch.from_numpy(target.astype(np.int64)), torch.from_numpy(holes)).backward(); opt.step()
        if step % 500 == 0:
            print(f"  train step {step}  {step/(time.time()-t0):.1f} st/s", flush=True)

    def predict(rule, masked, blind=False):
        model.eval()
        with torch.no_grad():
            r = torch.zeros_like(torch.from_numpy(rule)) if blind else torch.from_numpy(rule)
            p = model(r, torch.from_numpy(masked.astype(np.int64))).argmax(-1).numpy().astype(np.int8)
        model.train(); return p

    res = {"config": dict(q=Q, N=N, rho_train=RHO_TRAIN, k_test=K_TEST, U_size=U_SIZE,
                          steps=STEPS, d_model=D_MODEL, layers=LAYERS, p_mask=P),
           "U": sorted([list(map(list, [d[0]])) + [d[1]] for d in U], key=str),
           "chance": 1.0 / Q, "by_j": []}
    for j in range(0, K_TEST + 1):
        worlds_j = test_by_j[j]
        if not worlds_j:
            continue
        rule, masked, target, holes, Fs = build_batch_t4(worlds_j, N_EVAL, Q, P, random.Random(900 + j))
        pred = predict(rule, masked)
        blind = predict(rule, masked, blind=True)
        sp = forced_cell_stats_split(pred, masked, holes, Fs, Q, U)
        sp_blind = forced_cell_stats_split(blind, masked, holes, Fs, Q, U)
        rec = dict(j=j, n_worlds=len(worlds_j),
                   acc_all=sp['all'], acc_S=sp['S_sufficient'], acc_U=sp['U_critical'],
                   n_S=sp['n_S'], n_U=sp['n_U'],
                   blind_all=sp_blind['all'], blind_U=sp_blind['U_critical'])
        res["by_j"].append(rec)
        json.dump(res, open(OUT, "w"), indent=2)        # incremental
        print(f"j={j}  acc_all {sp['all']:.3f}  S-suff {sp['S_sufficient']:.3f} (n={sp['n_S']})  "
              f"U-crit {sp['U_critical']:.3f} (n={sp['n_U']})  | blind U-crit {sp_blind['U_critical']:.3f}  [saved]", flush=True)
    del model; gc.collect()

    # plot: accuracy vs j for S-sufficient vs U-critical cells, + chance
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        js = [r['j'] for r in res['by_j']]
        plt.figure(figsize=(6.5, 4.5))
        plt.plot(js, [r['acc_S'] for r in res['by_j']], 'o-', label='S-sufficient cells (seen rules)')
        plt.plot(js, [r['acc_U'] for r in res['by_j']], 's-', c='crimson', label='U-critical cells (held-out rule)')
        plt.plot(js, [r['blind_U'] for r in res['by_j']], '^--', c='gray', label='rule-blind, U-critical')
        plt.axhline(res['chance'], ls=':', c='k', label='Chance (1/q)')
        plt.xlabel('j = # forbidden dominoes from held-out set U'); plt.ylabel('Forced-cell accuracy')
        plt.title(f'Coverage: can the model use rules it never saw exercised?  (q={Q}, N={N})')
        plt.ylim(0, 1.02); plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(FIG, "t4_coverage.png"), dpi=140)
        print("saved figs/t4_coverage.png", flush=True)
    except Exception as e:
        print("plot skipped:", e, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
