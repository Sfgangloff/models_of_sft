"""
t4_informativeness.py — is a world's training-value anything beyond coverage?

Tests the hypothesis (Gangloff): maybe all worlds are equally informative — the
model just learns to apply whatever rules it is given, independent of which.

Method: sample many random training sets (varied composition). Train one model per
set; measure OOW forced-cell accuracy on a fixed neutral test bank. Record per-set
coverage (#dominoes exercised), mean kappa1, mean forcing_density, mean #configs.
Plus a SEED-NOISE baseline: one fixed set trained with several seeds.

Readout:
 - OOW vs coverage: is coverage the driver?
 - Among high-coverage sets: does OOW vary with complexity (kappa1 / forcing_density),
   or is it flat?
 - across-set variance vs within-set (seed) variance: if comparable, worlds are
   equally informative given coverage.

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
POOL_SIZE = 80
FMIN, FMAX = 3, 8          # per-world |F| range (varied coverage/complexity)
M = 10                     # worlds per training set
R = 10                     # number of random training sets
SEED_SET_SEEDS = [0, 1, 2] # seed-noise baseline on one fixed set
N_CFG, MINC = 40, 16
N_TEST, RHO_TEST = 16, 0.3
D_MODEL, LAYERS, NHEAD, DIMFF = 64, 3, 4, 256
STEPS, BATCH, LR, P, N_EVAL = 900, 32, 3e-4, 0.4, 600
OUT = os.path.join(RES, "t4_informativeness.json")


def make_world(F, rng):
    F = frozenset(F)
    if not count_at_least_two(Q, N, F):
        return None
    c = sample_configs(Q, N, F, N_CFG, rng)
    return {'F': F, 'configs': c} if c.shape[0] >= MINC else None


def train(pool, seed):
    torch.manual_seed(seed); rng = random.Random(1000 + seed)
    m = RuleCompleter(Q, N, d_model=D_MODEL, nhead=NHEAD, num_layers=LAYERS, dim_ff=DIMFF)
    opt = torch.optim.Adam(m.parameters(), lr=LR); m.train()
    for _ in range(STEPS):
        rule, masked, target, holes, _ = build_batch_t4(pool, BATCH, Q, P, rng)
        logits = m(torch.from_numpy(rule), torch.from_numpy(masked.astype(np.int64)))
        opt.zero_grad(); masked_loss(logits, torch.from_numpy(target.astype(np.int64)), torch.from_numpy(holes)).backward(); opt.step()
    return m


def oow(model, test):
    rule, masked, target, holes, Fs = build_batch_t4(test, N_EVAL, Q, P, random.Random(7))
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(rule), torch.from_numpy(masked.astype(np.int64))).argmax(-1).numpy().astype(np.int8)
    model.train()
    a, _ = forced_cell_stats(pred, masked, holes, Fs, Q)
    return a


def set_features(pool_idx, pool):
    ws = [pool[i] for i in pool_idx]
    cov = len(set().union(*[set(w['F']) for w in ws]))
    return dict(coverage=cov,
                kappa1_mean=float(np.mean([w['kappa1'] for w in ws])),
                fdens_mean=float(np.mean([w['fdens'] for w in ws])),
                cfgs_mean=float(np.mean([w['configs'].shape[0] for w in ws])))


def main():
    dom = domino_list(Q); rng = random.Random(31)
    pool = []; att = 0
    while len(pool) < POOL_SIZE and att < POOL_SIZE * 60:
        att += 1
        k = rng.randint(FMIN, FMAX)
        w = make_world(rng.sample(dom, k), rng)
        if w is not None:
            w['kappa1'] = kappa1(w['F'], Q)
            w['fdens'] = forcing_density(w, Q, n_masked=40)
            pool.append(w)
    print(f"pool: {len(pool)} worlds (|F| in [{FMIN},{FMAX}])", flush=True)

    # fixed neutral test bank (disjoint by F)
    pool_keys = {w['F'] for w in pool}
    test = []; att = 0
    while len(test) < N_TEST and att < N_TEST * 80:
        att += 1
        F = frozenset(d for d in dom if random.random() < RHO_TEST)
        if F in pool_keys:
            continue
        w = make_world(F, rng)
        if w is not None:
            test.append(w)
    print(f"test bank: {len(test)} neutral worlds", flush=True)

    res = {"config": dict(q=Q, N=N, M=M, R=R, steps=STEPS, fmin=FMIN, fmax=FMAX),
           "random_sets": [], "seed_baseline": []}

    # R random training sets, 1 seed each
    for r in range(R):
        idx = rng.sample(range(len(pool)), M)
        feat = set_features(idx, pool)
        t0 = time.time()
        model = train([pool[i] for i in idx], seed=0)
        acc = oow(model, test); del model; gc.collect()
        rec = dict(set=r, oow=acc, **feat, secs=time.time()-t0)
        res["random_sets"].append(rec)
        json.dump(res, open(OUT, "w"), indent=2)
        print(f"set {r:2d}  cov={feat['coverage']:2d}/18  k1~{feat['kappa1_mean']:.2f}  "
              f"fdens~{feat['fdens_mean']:.2f}  cfgs~{feat['cfgs_mean']:.0f}  OOW {acc:.3f}  [saved]", flush=True)

    # seed-noise baseline: one fixed set, several seeds
    fixed_idx = rng.sample(range(len(pool)), M)
    ffeat = set_features(fixed_idx, pool)
    for s in SEED_SET_SEEDS:
        model = train([pool[i] for i in fixed_idx], seed=s)
        acc = oow(model, test); del model; gc.collect()
        res["seed_baseline"].append(dict(seed=s, oow=acc, **ffeat))
        json.dump(res, open(OUT, "w"), indent=2)
        print(f"seed-baseline (fixed set, cov={ffeat['coverage']}/18) seed={s}  OOW {acc:.3f}  [saved]", flush=True)

    # summary stats
    accs = np.array([r['oow'] for r in res['random_sets']])
    seed_accs = np.array([r['oow'] for r in res['seed_baseline']])
    res["summary"] = dict(across_set_mean=float(accs.mean()), across_set_std=float(accs.std()),
                          seed_std=float(seed_accs.std()),
                          corr_oow_coverage=float(np.corrcoef([r['coverage'] for r in res['random_sets']], accs)[0, 1]),
                          corr_oow_kappa1=float(np.corrcoef([r['kappa1_mean'] for r in res['random_sets']], accs)[0, 1]),
                          corr_oow_fdens=float(np.corrcoef([r['fdens_mean'] for r in res['random_sets']], accs)[0, 1]))
    json.dump(res, open(OUT, "w"), indent=2)
    print("\nSUMMARY:", json.dumps(res["summary"], indent=2), flush=True)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
        cov = [r['coverage'] for r in res['random_sets']]
        ax[0].scatter(cov, accs, c='steelblue'); ax[0].set_xlabel('coverage (#dominoes)'); ax[0].set_ylabel('OOW forced-acc')
        ax[0].set_title(f"OOW vs coverage  (r={res['summary']['corr_oow_coverage']:.2f})")
        k1 = [r['kappa1_mean'] for r in res['random_sets']]
        ax[1].scatter(k1, accs, c='crimson'); ax[1].set_xlabel('mean κ̂₁'); ax[1].set_ylabel('OOW forced-acc')
        ax[1].set_title(f"OOW vs complexity  (r={res['summary']['corr_oow_kappa1']:.2f})")
        ax[1].axhspan(seed_accs.mean()-seed_accs.std(), seed_accs.mean()+seed_accs.std(), color='gray', alpha=0.2,
                      label='seed-noise band')
        ax[1].legend(fontsize=8)
        plt.tight_layout(); plt.savefig(os.path.join(FIG, "t4_informativeness.png"), dpi=140)
        print("saved figs/t4_informativeness.png", flush=True)
    except Exception as e:
        print("plot skipped:", e, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
