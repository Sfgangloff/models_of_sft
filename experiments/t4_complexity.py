"""
t4_complexity.py — Exp D (first look): does a world's training-informativeness
track its causal complexity κ̂₁?

Generate a bank of worlds with a FIXED number of forbidden dominoes (|F|=k, so they
differ in causal structure, not constraint count). Bin by κ̂₁ (low/mid/high). Train a
rule-conditioned completer on each bin; measure OOW forced-cell accuracy on a fixed
neutral test bank. Report per-pool κ̂, rule-space coverage, and #configs so the
coverage/diversity confounds are visible.

Safe envelope: CPU/2-thread/niced, watchdog, incremental saves.
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
from t4_data import domino_list, build_batch_t4, forced_cell_stats
from t4_model import RuleCompleter
from t4_run import masked_loss
from causal_complexity import kappa1, forcing_density

HERE = os.path.dirname(__file__); RES = os.path.join(HERE, "results"); FIG = os.path.join(HERE, "figs")
for d in (RES, FIG): os.makedirs(d, exist_ok=True)

Q, N = 3, 6
K_FORB = 7                 # fixed |F| for bank worlds (gives kappa1 spread 0..3)
N_BANK = 160
M = 18                     # worlds per pool
N_CFG, MINC = 40, 16
N_TEST, RHO_TEST = 16, 0.3
D_MODEL, LAYERS, NHEAD, DIMFF = 80, 3, 4, 256
STEPS, BATCH, LR, P, N_EVAL = 1500, 32, 3e-4, 0.4, 600
OUT = os.path.join(RES, "t4_complexity.json")


def make_world(F, rng):
    F = frozenset(F)
    if not count_at_least_two(Q, N, F):
        return None
    c = sample_configs(Q, N, F, N_CFG, rng)
    return {'F': F, 'configs': c} if c.shape[0] >= MINC else None


def coverage(worlds):
    s = set()
    for w in worlds:
        s |= set(w['F'])
    return len(s)


def train(pool, seed=0):
    torch.manual_seed(seed); rng = random.Random(1000 + seed)
    m = RuleCompleter(Q, N, d_model=D_MODEL, nhead=NHEAD, num_layers=LAYERS, dim_ff=DIMFF)
    opt = torch.optim.Adam(m.parameters(), lr=LR); m.train()
    for _ in range(STEPS):
        rule, masked, target, holes, _ = build_batch_t4(pool, BATCH, Q, P, rng)
        logits = m(torch.from_numpy(rule), torch.from_numpy(masked.astype(np.int64)))
        opt.zero_grad(); masked_loss(logits, torch.from_numpy(target.astype(np.int64)), torch.from_numpy(holes)).backward(); opt.step()
    return m


def oow_forced(model, test, seed=7):
    rule, masked, target, holes, Fs = build_batch_t4(test, N_EVAL, Q, P, random.Random(seed))
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(rule), torch.from_numpy(masked.astype(np.int64))).argmax(-1).numpy().astype(np.int8)
    model.train()
    a, n = forced_cell_stats(pred, masked, holes, Fs, Q)
    return a, n


def main():
    dom = domino_list(Q)
    rng = random.Random(2025)
    # bank: worlds with exactly K_FORB forbidden dominoes
    bank = []; att = 0
    while len(bank) < N_BANK and att < N_BANK * 60:
        att += 1
        F = random.sample(dom, K_FORB)
        w = make_world(F, rng)
        if w is not None:
            w['kappa1'] = kappa1(w['F'], Q)
            bank.append(w)
    ks = np.array([w['kappa1'] for w in bank])
    print(f"bank: {len(bank)} worlds, |F|={K_FORB}; kappa1 range {ks.min()}..{ks.max()}, "
          f"hist {np.bincount(ks)}", flush=True)

    # sort by kappa1, take low / mid / high pools of size M
    bank.sort(key=lambda w: w['kappa1'])
    n = len(bank)
    pools = {'low': bank[:M], 'mid': bank[(n-M)//2:(n-M)//2 + M], 'high': bank[-M:]}

    # neutral test bank (random rho), disjoint from bank by F
    bank_keys = {w['F'] for w in bank}
    test = []; att = 0
    while len(test) < N_TEST and att < N_TEST * 80:
        att += 1
        F = frozenset(d for d in dom if random.random() < RHO_TEST)
        if F in bank_keys:
            continue
        w = make_world(F, rng)
        if w is not None:
            test.append(w)
    print(f"test bank: {len(test)} neutral worlds", flush=True)

    res = {"config": dict(q=Q, N=N, k_forb=K_FORB, M=M, steps=STEPS), "pools": []}
    for name in ['low', 'mid', 'high']:
        pool = pools[name]
        kmean = float(np.mean([w['kappa1'] for w in pool]))
        cov = coverage(pool)
        cfg_mean = float(np.mean([w['configs'].shape[0] for w in pool]))
        fdens = float(np.mean([forcing_density(w, Q, n_masked=60) for w in pool]))
        t0 = time.time()
        model = train(pool, seed=0)
        acc, nforced = oow_forced(model, test)
        del model; gc.collect()
        rec = dict(pool=name, kappa1_mean=kmean, coverage=cov, configs_mean=cfg_mean,
                   forcing_density=fdens, oow_forced_acc=acc, n_forced=nforced, secs=time.time()-t0)
        res["pools"].append(rec)
        json.dump(res, open(OUT, "w"), indent=2)
        print(f"[{name}] kappa1~{kmean:.2f}  cov={cov}/18  cfgs~{cfg_mean:.0f}  "
              f"fdens={fdens:.2f}  -> OOW forced-acc {acc:.3f}  ({rec['secs']:.0f}s) [saved]", flush=True)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        ks = [r['kappa1_mean'] for r in res['pools']]; accs = [r['oow_forced_acc'] for r in res['pools']]
        plt.figure(figsize=(6, 4.3))
        plt.plot(ks, accs, 'o-', ms=9)
        for r in res['pools']:
            plt.annotate(f"{r['pool']}\ncov={r['coverage']}/18\ncfgs~{r['configs_mean']:.0f}",
                         (r['kappa1_mean'], r['oow_forced_acc']), fontsize=7,
                         textcoords="offset points", xytext=(6, 6))
        plt.xlabel('mean causal complexity κ̂₁ of training pool'); plt.ylabel('OOW forced-cell accuracy')
        plt.title(f'Informativeness vs causal complexity (|F|={K_FORB} fixed, q={Q}, N={N})')
        plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(FIG, "t4_complexity.png"), dpi=140)
        print("saved figs/t4_complexity.png", flush=True)
    except Exception as e:
        print("plot skipped:", e, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
