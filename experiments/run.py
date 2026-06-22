"""
run.py — headline cross-world generalization experiment (opaque in-context T1).

Pipeline:
  1. Build (and cache) a training master pool of worlds + a disjoint OOW pool.
  2. For each M in the task-diversity ladder and each seed: train an identical
     in-context Transformer on M worlds for a FIXED number of steps, then
     evaluate balanced accuracy / AUROC in-world (IW) and out-of-world (OOW).
  3. Compute non-neural baselines (majority, frequency learner) and the
     mismatched-context control on the fixed eval sets.
  4. Save results JSON + the headline figure (accuracy vs M, IW & OOW).

Usage:
  python experiments/run.py --smoke          # tiny end-to-end validation
  python experiments/run.py                  # full run
"""

from __future__ import annotations
import argparse, json, os, pickle, random, time
import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, os.path.dirname(__file__))
from worlds import build_world_pool
from episodes import build_batch
from baselines import frequency_learner_batch
from model import InContextT1

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figs")
CACHE = os.path.join(HERE, "cache")
for d in (RES, FIG, CACHE):
    os.makedirs(d, exist_ok=True)

try:
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
except Exception:
    balanced_accuracy_score = roc_auc_score = None


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

def get_config(smoke):
    if smoke:
        return dict(
            q=3, N=8, rho=0.3, K=6,
            n_train_worlds=16, n_oow_worlds=8,
            n_configs=40, eval_hold=10,
            M_list=[1, 4, 16], seeds=[0],
            steps=150, batch=32, lr=3e-4,
            d_model=64, nhead=4, layers=2, dim_ff=128,
            n_eval_episodes=400,
        )
    return dict(
        q=3, N=8, rho=0.3, K=6,
        n_train_worlds=512, n_oow_worlds=128,
        n_configs=110, eval_hold=30,
        M_list=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512], seeds=[0, 1, 2],
        steps=4000, batch=32, lr=3e-4,
        d_model=96, nhead=4, layers=3, dim_ff=256,
        n_eval_episodes=3000,
    )


# ----------------------------------------------------------------------------
# Pools (cached)
# ----------------------------------------------------------------------------

def view(world, sl):
    return {'F': world['F'], 'key': world['key'], 'configs': world['configs'][sl]}


def build_or_load_pools(cfg):
    tag = f"q{cfg['q']}_N{cfg['N']}_rho{cfg['rho']}_tr{cfg['n_train_worlds']}_oow{cfg['n_oow_worlds']}_nc{cfg['n_configs']}"
    path = os.path.join(CACHE, f"pools_{tag}.pkl")
    if os.path.exists(path):
        print(f"Loading cached pools: {path}", flush=True)
        with open(path, "rb") as f:
            return pickle.load(f)
    print("Building world pools (SAT generation)...", flush=True)
    rng = random.Random(12345)
    t0 = time.time()
    # require enough distinct configs that BOTH the training slice and the
    # held-out eval slice (eval_hold) are well populated after splitting.
    min_cfg = cfg['eval_hold'] + cfg['K'] + 10
    train = build_world_pool(cfg['q'], cfg['rho'], cfg['N'], cfg['n_train_worlds'],
                             cfg['n_configs'], rng, min_configs=min_cfg, label="train")
    excl = {w['key'] for w in train}
    oow = build_world_pool(cfg['q'], cfg['rho'], cfg['N'], cfg['n_oow_worlds'],
                           cfg['n_configs'], rng, exclude_keys=excl,
                           min_configs=min_cfg, label="oow")
    pools = {'train': train, 'oow': oow}
    with open(path, "wb") as f:
        pickle.dump(pools, f)
    print(f"Pools built in {time.time()-t0:.1f}s, cached to {path}", flush=True)
    return pools


# ----------------------------------------------------------------------------
# Eval sets (fixed across all M and seeds for comparability)
# ----------------------------------------------------------------------------

def make_eval_batch(worlds, n, K, q, seed, mismatch=False):
    rng = random.Random(seed)
    ctx, qry, lab = build_batch(worlds, n, K, q, rng, mismatch=mismatch)
    return ctx, qry, lab


def metrics(labels, probs):
    preds = (probs >= 0.5).astype(int)
    if balanced_accuracy_score is not None:
        bacc = balanced_accuracy_score(labels, preds)
        try:
            auc = roc_auc_score(labels, probs)
        except Exception:
            auc = float('nan')
    else:
        tp = ((preds == 1) & (labels == 1)).sum(); p = (labels == 1).sum()
        tn = ((preds == 0) & (labels == 0)).sum(); ne = (labels == 0).sum()
        bacc = 0.5 * (tp / max(p, 1) + tn / max(ne, 1)); auc = float('nan')
    return float(bacc), float(auc)


@torch.no_grad()
def eval_neural(model, ctx, qry, device, batch=256):
    model.eval()
    probs = []
    for i in range(0, qry.shape[0], batch):
        c = torch.from_numpy(ctx[i:i+batch].astype(np.int64)).to(device)
        q = torch.from_numpy(qry[i:i+batch].astype(np.int64)).to(device)
        logits = model(c, q)
        probs.append(torch.softmax(logits, -1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------

def train_one(train_worlds, world_idx, cfg, device, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    rng = random.Random(1000 + seed)
    model = InContextT1(cfg['q'], cfg['N'], cfg['K'], d_model=cfg['d_model'],
                        nhead=cfg['nhead'], num_layers=cfg['layers'],
                        dim_ff=cfg['dim_ff']).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    lossfn = nn.CrossEntropyLoss()
    model.train()
    for step in range(cfg['steps']):
        ctx, qry, lab = build_batch(train_worlds, cfg['batch'], cfg['K'], cfg['q'],
                                    rng, world_indices=world_idx)
        c = torch.from_numpy(ctx.astype(np.int64)).to(device)
        q = torch.from_numpy(qry.astype(np.int64)).to(device)
        y = torch.from_numpy(lab).to(device)
        opt.zero_grad()
        loss = lossfn(model(c, q), y)
        loss.backward(); opt.step()
    return model


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    cfg = get_config(args.smoke)

    device = args.device or ("mps" if torch.backends.mps.is_available()
                             else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  config={json.dumps(cfg)}", flush=True)

    pools = build_or_load_pools(cfg)
    train_all, oow_all = pools['train'], pools['oow']

    # split each world's configs into train / held-out-eval halves
    R = cfg['eval_hold']
    train_views = [view(w, slice(0, max(1, w['configs'].shape[0] - R))) for w in train_all]
    train_eval_views = [view(w, slice(max(1, w['configs'].shape[0] - R), None)) for w in train_all]
    # OOW worlds are never trained on -> use all their configs for eval
    oow_eval_views = oow_all

    # fixed OOW eval set (shared across all M, seeds)
    K, q = cfg['K'], cfg['q']
    oow_ctx, oow_qry, oow_lab = make_eval_batch(oow_eval_views, cfg['n_eval_episodes'], K, q, seed=777)

    # ---- baselines on the fixed OOW eval set ----
    freq_pred = frequency_learner_batch(oow_ctx, oow_qry, q)
    freq_bacc, _ = metrics(oow_lab, freq_pred.astype(float))
    majority_bacc = 0.5
    print(f"OOW baselines: majority={majority_bacc:.3f}  frequency_learner={freq_bacc:.3f}", flush=True)

    results = {'config': cfg, 'baselines': {'majority': majority_bacc,
               'frequency_learner_oow': freq_bacc}, 'runs': []}

    nt = len(train_views)
    for M in cfg['M_list']:
        if M > nt:
            continue
        for seed in cfg['seeds']:
            t0 = time.time()
            # nested subset: first M worlds of a seed-shuffled master order
            order = list(range(nt)); random.Random(424242).shuffle(order)
            world_idx = order[:M]
            model = train_one(train_views, world_idx, cfg, device, seed)

            # IW eval: held-out queries from the M TRAINING worlds
            iw_views = [train_eval_views[i] for i in world_idx]
            iw_ctx, iw_qry, iw_lab = make_eval_batch(iw_views, cfg['n_eval_episodes'], K, q, seed=555 + seed)
            iw_probs = eval_neural(model, iw_ctx, iw_qry, device)
            iw_bacc, iw_auc = metrics(iw_lab, iw_probs)

            # OOW eval: fixed held-out world set
            oow_probs = eval_neural(model, oow_ctx, oow_qry, device)
            oow_bacc, oow_auc = metrics(oow_lab, oow_probs)

            rec = dict(M=M, seed=seed, iw_bacc=iw_bacc, iw_auc=iw_auc,
                       oow_bacc=oow_bacc, oow_auc=oow_auc, secs=time.time() - t0)
            results['runs'].append(rec)
            print(f"M={M:4d} seed={seed}  IW bacc={iw_bacc:.3f} auc={iw_auc:.3f} | "
                  f"OOW bacc={oow_bacc:.3f} auc={oow_auc:.3f}  ({rec['secs']:.0f}s)", flush=True)

    # ---- mismatched-context control on the largest-M, seed-0 model ----
    Mmax = max(m for m in cfg['M_list'] if m <= nt)
    order = list(range(nt)); random.Random(424242).shuffle(order)
    model = train_one(train_views, order[:Mmax], cfg, device, cfg['seeds'][0])
    mm_ctx, mm_qry, mm_lab = make_eval_batch(oow_eval_views, cfg['n_eval_episodes'], K, q, seed=999, mismatch=True)
    mm_probs = eval_neural(model, mm_ctx, mm_qry, device)
    mm_bacc, _ = metrics(mm_lab, mm_probs)
    results['mismatched_context_control'] = dict(M=Mmax, oow_matched_ref=None, mismatched_bacc=mm_bacc)
    print(f"Mismatched-context control (M={Mmax}): bacc={mm_bacc:.3f} (should be ~0.5)", flush=True)

    out = os.path.join(RES, "headline_smoke.json" if args.smoke else "headline.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out}", flush=True)

    make_plot(results, cfg, args.smoke)


def make_plot(results, cfg, smoke):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import defaultdict
    iw = defaultdict(list); oow = defaultdict(list)
    for r in results['runs']:
        iw[r['M']].append(r['iw_bacc']); oow[r['M']].append(r['oow_bacc'])
    Ms = sorted(iw)
    def ms_err(d): return ([np.mean(d[m]) for m in Ms], [np.std(d[m]) for m in Ms])
    iw_m, iw_e = ms_err(iw); oow_m, oow_e = ms_err(oow)
    plt.figure(figsize=(6.5, 4.5))
    plt.errorbar(Ms, iw_m, yerr=iw_e, marker='o', capsize=3, label='In-world (IW)')
    plt.errorbar(Ms, oow_m, yerr=oow_e, marker='s', capsize=3, label='Out-of-world (OOW)')
    plt.axhline(results['baselines']['frequency_learner_oow'], ls='--', c='gray',
                label='Frequency learner (OOW)')
    plt.axhline(0.5, ls=':', c='k', label='Majority / chance')
    plt.xscale('log', base=2)
    plt.xlabel('Number of training worlds  M'); plt.ylabel('Balanced accuracy')
    plt.title(f"Cross-world generalization (T1, q={cfg['q']}, N={cfg['N']}, "
              f"ρ={cfg['rho']}, K={cfg['K']})")
    plt.ylim(0.45, 1.02); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    out = os.path.join(FIG, "headline_smoke.png" if smoke else "headline.png")
    plt.savefig(out, dpi=150)
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
