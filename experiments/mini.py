"""
mini.py — tiny, CPU-only, memory-light mini M-ablation for T1.

Demonstrates the headline in miniature: train on M worlds, evaluate in-world
(IW) and out-of-world (OOW); OOW should rise with M while IW stays high.
Deliberately small (N=6, tiny model, 2 threads, niced) to be safe on a
RAM-constrained laptop.
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
try:
    os.nice(10)
except Exception:
    pass

import time, random, pickle, sys, gc, json
import numpy as np
import torch
torch.set_num_threads(2)
sys.path.insert(0, os.path.dirname(__file__))
from worlds import build_world_pool
from episodes import build_batch
from baselines import frequency_learner_batch
from model import InContextT1
from sklearn.metrics import balanced_accuracy_score

HERE = os.path.dirname(__file__)
CACHE = os.path.join(HERE, "cache"); RES = os.path.join(HERE, "results"); FIG = os.path.join(HERE, "figs")
for d in (CACHE, RES, FIG): os.makedirs(d, exist_ok=True)

# ---- tiny config ----
Q, N, RHO, K = 3, 6, 0.3, 4
N_TRAIN, N_OOW, N_CFG, EVAL_HOLD = 32, 12, 45, 12
D_MODEL, LAYERS, NHEAD = 48, 2, 4
M_LIST = [1, 8, 32]
STEPS, BATCH, LR = 700, 16, 3e-4
N_EVAL = 300
DEVICE = "cpu"


def view(w, sl):
    return {'F': w['F'], 'key': w['key'], 'configs': w['configs'][sl]}


def get_pool():
    path = os.path.join(CACHE, f"minipool_q{Q}_N{N}_rho{RHO}.pkl")
    if os.path.exists(path):
        print(f"load {path}", flush=True); return pickle.load(open(path, "rb"))
    print("building tiny N=6 pool (single-threaded SAT)...", flush=True)
    rng = random.Random(11); minc = EVAL_HOLD + K + 6
    tr = build_world_pool(Q, RHO, N, N_TRAIN, N_CFG, rng, min_configs=minc, label="train", log_every=16)
    excl = {w['key'] for w in tr}
    oo = build_world_pool(Q, RHO, N, N_OOW, N_CFG, rng, exclude_keys=excl, min_configs=minc, label="oow", log_every=12)
    pools = {'train': tr, 'oow': oo}; pickle.dump(pools, open(path, "wb")); print("saved", flush=True)
    return pools


def bacc_probs(model, ctx, qry):
    model.eval()
    with torch.no_grad():
        c = torch.from_numpy(ctx.astype(np.int64)); q = torch.from_numpy(qry.astype(np.int64))
        p = torch.softmax(model(c, q), -1)[:, 1].numpy()
    return p


def train_eval(train_views, eval_views, oow_views, world_idx, seed, fixed_evals):
    torch.manual_seed(seed); rng = random.Random(1000 + seed)
    model = InContextT1(Q, N, K, d_model=D_MODEL, nhead=NHEAD, num_layers=LAYERS, dim_ff=4*D_MODEL)
    opt = torch.optim.Adam(model.parameters(), lr=LR); lf = torch.nn.CrossEntropyLoss()
    model.train()
    for step in range(STEPS):
        ctx, qry, lab = build_batch(train_views, BATCH, K, Q, rng, world_indices=world_idx)
        opt.zero_grad()
        out = model(torch.from_numpy(ctx.astype(np.int64)), torch.from_numpy(qry.astype(np.int64)))
        lf(out, torch.from_numpy(lab)).backward(); opt.step()
    # IW eval: held-out queries from the M training worlds
    iw_views = [eval_views[i] for i in world_idx]
    ic, iq, il = build_batch(iw_views, N_EVAL, K, Q, random.Random(555))
    iw = balanced_accuracy_score(il, (bacc_probs(model, ic, iq) >= 0.5).astype(int))
    # OOW eval: fixed held-out worlds
    oc, oq, ol = fixed_evals['oow']
    oo = balanced_accuracy_score(ol, (bacc_probs(model, oc, oq) >= 0.5).astype(int))
    return model, float(iw), float(oo)


def main():
    print(f"device={DEVICE} N={N} K={K} d={D_MODEL} L={LAYERS} M={M_LIST} steps={STEPS}", flush=True)
    pools = get_pool(); tr_all, oo_all = pools['train'], pools['oow']
    tr_train = [view(w, slice(0, w['configs'].shape[0] - EVAL_HOLD)) for w in tr_all]
    tr_eval = [view(w, slice(w['configs'].shape[0] - EVAL_HOLD, None)) for w in tr_all]

    oc, oq, ol = build_batch(oo_all, N_EVAL, K, Q, random.Random(2))
    fixed = {'oow': (oc, oq, ol)}
    freq = balanced_accuracy_score(ol, frequency_learner_batch(oc, oq, Q))
    print(f"frequency-learner OOW bacc = {freq:.3f}", flush=True)

    order = list(range(N_TRAIN)); random.Random(42).shuffle(order)
    results = {'config': dict(Q=Q, N=N, K=K, rho=RHO, d_model=D_MODEL, layers=LAYERS,
                              M_list=M_LIST, steps=STEPS, batch=BATCH),
               'frequency_learner_oow': float(freq), 'runs': []}
    last_model = None
    for M in M_LIST:
        t0 = time.time()
        model, iw, oo = train_eval(tr_train, tr_eval, oo_all, order[:M], seed=0, fixed_evals=fixed)
        print(f"M={M:3d}  IW {iw:.3f}  OOW {oo:.3f}  (freq {freq:.3f})  {time.time()-t0:.0f}s", flush=True)
        results['runs'].append(dict(M=M, iw=iw, oow=oo))
        if M == M_LIST[-1]:
            last_model = model
        else:
            del model; gc.collect()

    # mismatched-context control on the largest-M model
    mc, mq, mlab = build_batch(oo_all, N_EVAL, K, Q, random.Random(3), mismatch=True)
    mm = balanced_accuracy_score(mlab, (bacc_probs(last_model, mc, mq) >= 0.5).astype(int))
    results['mismatched_context_oow'] = float(mm)
    print(f"mismatched-context control (M={M_LIST[-1]}): {mm:.3f}  (should be ~0.5)", flush=True)

    json.dump(results, open(os.path.join(RES, "mini.json"), "w"), indent=2)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    Ms = [r['M'] for r in results['runs']]
    plt.figure(figsize=(6, 4.2))
    plt.plot(Ms, [r['iw'] for r in results['runs']], 'o-', label='In-world (IW)')
    plt.plot(Ms, [r['oow'] for r in results['runs']], 's-', label='Out-of-world (OOW)')
    plt.axhline(freq, ls='--', c='gray', label='Frequency learner (OOW)')
    plt.axhline(0.5, ls=':', c='k', label='Chance')
    plt.xscale('log', base=2); plt.xlabel('Number of training worlds  M'); plt.ylabel('Balanced accuracy')
    plt.title(f'Mini cross-world ablation (T1, q={Q}, N={N}, K={K})')
    plt.ylim(0.45, 1.02); plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "mini.png"), dpi=130)
    print("saved results/mini.json and figs/mini.png\nDONE", flush=True)


if __name__ == "__main__":
    main()
